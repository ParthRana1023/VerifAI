import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from app import run_news_analysis, get_report_as_markdown
from reddit import scrape_reddit_data, extract_keywords, is_reddit_url
import traceback
from setup import setup_crewai_config, setup_api_keys, check_gemini_status

st.set_page_config(
    page_title="VerifAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gemini_connection():
    """Check if Gemini API key is set"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return False, "GEMINI_API_KEY not found. Please set it in your environment variables or .env file."
    return True, "Gemini API key is set."



def display_report(report):
    """Display the news analysis report in the Streamlit interface"""
    if not report:
        st.error("No report to display")
        return
    
    # Handle string reports (most common case with your current setup)
    if isinstance(report, str):
        st.markdown("## Analysis Report")
        st.markdown(report)
        return
        
    try:
        # Handle structured reports (if available)
        st.title(f"News Analysis Report: {getattr(report, 'query_summary', 'Unknown Topic')}")
        
        st.header("Key Findings & Summary")
        st.write(getattr(report, 'key_findings', 'No key findings available'))
        
        # Related Articles
        st.header("Related Articles")
        related_articles = getattr(report, 'related_articles', [])
        if related_articles:
            for article in related_articles:
                if isinstance(article, dict):
                    for title, url in article.items():
                        st.markdown(f"- [{title}]({url})")
        else:
            st.write("No related articles found")
        
        # Related Words (wordcloud)
        st.header("Related Words")
        st.write("*Keywords found in the analysis:*")
        related_words = getattr(report, 'related_words', [])
        if related_words:
            st.write(", ".join(related_words))
        else:
            st.write("No related words found")
        
        # Topic Clusters
        st.header("Related Topic Clusters")
        topic_clusters = getattr(report, 'topic_clusters', [])
        if topic_clusters:
            for cluster in topic_clusters:
                if isinstance(cluster, dict):
                    st.markdown(f"- **{cluster.get('topic', 'N/A')}** (Size: {cluster.get('size', 'N/A')})")
                    related_narratives = cluster.get('related_narratives', [])
                    if related_narratives:
                        st.markdown("  - Related narratives: " + ", ".join(related_narratives))
        else:
            st.write("No topic clusters found")
        
        # Top Sources
        st.header("List of Top Sources")
        top_sources = getattr(report, 'top_sources', [])
        if top_sources:
            sources_data = []
            for source in top_sources:
                sources_data.append({
                    "Domain": getattr(source, 'domain', 'N/A'),
                    "Factual Rating": getattr(source, 'factual_rating', 'N/A'),
                    "Articles Count": getattr(source, 'articles_count', 0),
                    "Engagement": getattr(source, 'engagement', 0)
                })
            st.dataframe(pd.DataFrame(sources_data))
            plot_source_reliability(top_sources)
        else:
            st.write("No source data available")

        # Top Hashtags
        st.header("Top Hashtags")
        top_hashtags = getattr(report, 'top_hashtags', [])
        if top_hashtags:
            hashtags_data = []
            for hashtag in top_hashtags:
                hashtags_data.append({
                    "Hashtag": getattr(hashtag, 'hashtag', 'N/A'),
                    "Engagement Rate": getattr(hashtag, 'engagement_rate', 0.0),
                    "Reach": getattr(hashtag, 'reach', 0),
                    "Sentiment": getattr(hashtag, 'sentiment', 'N/A')
                })
            st.dataframe(pd.DataFrame(hashtags_data))
            plot_social_media_metrics(top_hashtags)
        else:
            st.write("No hashtag data available")

        # Similar Posts Time Series
        st.header("Similar Posts Over Time")
        similar_posts_time_series = getattr(report, 'similar_posts_time_series', [])
        if similar_posts_time_series:
            time_series_data = []
            for ts_entry in similar_posts_time_series:
                time_series_data.append({
                    "Date": getattr(ts_entry, 'date', 'N/A'),
                    "Count": getattr(ts_entry, 'count', 0)
                })
            df_time_series = pd.DataFrame(time_series_data)
            df_time_series['Date'] = pd.to_datetime(df_time_series['Date'])
            df_time_series = df_time_series.sort_values(by='Date')
            st.dataframe(df_time_series)
            plot_time_series_data(df_time_series)
        else:
            st.write("No time series data available")

        # Propaganda Analysis
        st.header("Propaganda Analysis")
        propaganda_analysis = getattr(report, 'propaganda_analysis', None)
        if propaganda_analysis:
            st.subheader("Overall Reliability Score")
            st.write(f"{getattr(propaganda_analysis, 'overall_reliability_score', 'N/A')}/100")

            propaganda_techniques = getattr(propaganda_analysis, 'propaganda_techniques', [])
            if propaganda_techniques:
                st.subheader("Propaganda Techniques Detected")
                tech_data = []
                for tech in propaganda_techniques:
                    tech_data.append({
                        "Technique": getattr(tech, 'technique_name', 'N/A'),
                        "Frequency": getattr(tech, 'frequency', 0),
                        "Severity": getattr(tech, 'severity', 0.0),
                        "Example": getattr(tech, 'example', 'N/A'),
                        "Explanation": getattr(tech, 'explanation', 'N/A')
                    })
                st.dataframe(pd.DataFrame(tech_data))
                plot_propaganda_techniques(propaganda_techniques)
            else:
                st.write("No propaganda techniques detected.")

            misinformation_indicators = getattr(propaganda_analysis, 'misinformation_indicators', [])
            if misinformation_indicators:
                st.subheader("Misinformation Indicators")
                for indicator in misinformation_indicators:
                    st.markdown(f"- **Type**: {getattr(indicator, 'indicator_type', 'N/A')}")
                    st.markdown(f"  **Confidence**: {getattr(indicator, 'confidence', 'N/A')}")
                    st.markdown(f"  **Correction**: {getattr(indicator, 'correction', 'N/A')}")
                    st.markdown(f"  **Source Verification**: {', '.join(getattr(indicator, 'source_verification', []))}")
            else:
                st.write("No misinformation indicators detected.")

            fake_news_sites = getattr(propaganda_analysis, 'fake_news_sites', [])
            if fake_news_sites:
                st.subheader("Associated Fake News Sites")
                fake_news_data = []
                for site in fake_news_sites:
                    fake_news_data.append({
                        "Domain": getattr(site, 'domain', 'N/A'),
                        "Shares": getattr(site, 'shares', 0),
                        "Engagement": getattr(site, 'engagement', 0),
                        "Known False Stories": getattr(site, 'known_false_stories', 0)
                    })
                st.dataframe(pd.DataFrame(fake_news_data))
                plot_fake_news_sites(fake_news_sites)
            else:
                st.write("No fake news sites identified.")

        else:
            st.write("No propaganda analysis available.")

    except Exception as e:
        st.error(f"Error displaying structured report: {str(e)}")
        st.markdown("## Raw Report")
        st.text(str(report))

def plot_source_reliability(sources):
    if not sources:
        return
    df = pd.DataFrame([{"domain": s.domain, "factual_rating": s.factual_rating, "articles_count": s.articles_count, "engagement": s.engagement} for s in sources])
    
    st.subheader("Source Reliability: Articles Count by Factual Rating")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='factual_rating', y='articles_count', data=df, ax=ax, palette='viridis')
    ax.set_title('Articles Count by Factual Rating')
    ax.set_xlabel('Factual Rating')
    ax.set_ylabel('Number of Articles')
    st.pyplot(fig)

    st.subheader("Source Reliability: Engagement by Factual Rating")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='factual_rating', y='engagement', data=df, ax=ax, palette='magma')
    ax.set_title('Engagement by Factual Rating')
    ax.set_xlabel('Factual Rating')
    ax.set_ylabel('Engagement')
    st.pyplot(fig)

def plot_social_media_metrics(hashtags):
    if not hashtags:
        return
    df = pd.DataFrame([{"hashtag": h.hashtag, "engagement_rate": h.engagement_rate, "reach": h.reach, "sentiment": h.sentiment} for h in hashtags])

    st.subheader("Social Media: Engagement Rate by Hashtag")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='hashtag', y='engagement_rate', data=df, ax=ax, palette='cubehelix')
    ax.set_title('Engagement Rate by Hashtag')
    ax.set_xlabel('Hashtag')
    ax.set_ylabel('Engagement Rate (%)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("Social Media: Reach by Hashtag")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='hashtag', y='reach', data=df, ax=ax, palette='rocket')
    ax.set_title('Reach by Hashtag')
    ax.set_xlabel('Hashtag')
    ax.set_ylabel('Reach')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("Social Media: Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(8, 8))
    sentiment_counts = df['sentiment'].value_counts()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax.set_title('Sentiment Distribution')
    st.pyplot(fig)

def plot_time_series_data(df_time_series):
    if df_time_series.empty:
        return
    st.subheader("Time Series of Similar Posts")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Date', y='Count', data=df_time_series, marker='o', ax=ax)
    ax.set_title('Count of Similar Posts Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_propaganda_techniques(techniques):
    if not techniques:
        return
    df = pd.DataFrame([{"technique_name": t.technique_name, "frequency": t.frequency, "severity": t.severity} for t in techniques])

    st.subheader("Propaganda Techniques: Frequency")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='technique_name', y='frequency', data=df, ax=ax, palette='mako')
    ax.set_title('Frequency of Propaganda Techniques')
    ax.set_xlabel('Technique')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("Propaganda Techniques: Severity")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='technique_name', y='severity', data=df, ax=ax, palette='flare')
    ax.set_title('Severity of Propaganda Techniques')
    ax.set_xlabel('Technique')
    ax.set_ylabel('Severity (0-10)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def plot_fake_news_sites(sites):
    if not sites:
        return
    df = pd.DataFrame([{"domain": s.domain, "shares": s.shares, "engagement": s.engagement, "known_false_stories": s.known_false_stories} for s in sites])

    st.subheader("Fake News Sites: Shares and Engagement")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_melted = df.melt(id_vars=['domain'], value_vars=['shares', 'engagement'], var_name='Metric', value_name='Value')
    sns.barplot(x='domain', y='Value', hue='Metric', data=df_melted, ax=ax, palette='crest')
    ax.set_title('Shares and Engagement for Fake News Sites')
    ax.set_xlabel('Domain')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("Fake News Sites: Known False Stories")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='domain', y='known_false_stories', data=df, ax=ax, palette='viridis')
    ax.set_title('Number of Known False Stories by Fake News Site')
    ax.set_xlabel('Domain')
    ax.set_ylabel('Known False Stories')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def analyze_reddit_post(url):
    """
    Analyze a Reddit post and return the news analysis report
    """
    # Scrape Reddit data
    with st.spinner("Scraping Reddit post..."):
        reddit_data = scrape_reddit_data(url)
    
    if "error" in reddit_data:
        st.error(f"Error: {reddit_data['error']}")
        return None
    
    # Extract keywords
    with st.spinner("Extracting keywords..."):
        keywords = extract_keywords(reddit_data)
    
    if not keywords:
        st.error("No keywords extracted from the post.")
        return None
    
    # Convert keywords to list
    keyword_list = [kw['text'] for kw in keywords]
    
    # Display extracted information
    st.subheader("Reddit Post Information")
    st.write(f"**Title:** {reddit_data['title']}")
    st.write(f"**Subreddit:** r/{reddit_data['subreddit']}")
    st.write(f"**Author:** u/{reddit_data['author']}")
    st.write(f"**Score:** {reddit_data['score']} (Upvote ratio: {reddit_data['upvote_ratio']})")
    st.write(f"**Comments:** {reddit_data['num_comments']}")
    
    # Display content if not empty
    if reddit_data['selftext']:
        with st.expander("Post Content"):
            st.write(reddit_data['selftext'])
    
    # Display top keywords
    st.subheader("Top Keywords")
    keywords_data = []
    for kw in keywords[:20]:  # Show top 20 keywords
        keywords_data.append({
            "Keyword": kw['text'],
            "Frequency": kw['frequency']
        })
    st.dataframe(pd.DataFrame(keywords_data))
    
    # User query
    user_query = f"News analysis for: {reddit_data['title']}"
    keywords = keyword_list[:5]
    st.info(f"Running analysis for: {user_query}")
    
    # Run analysis
    with st.spinner("Running news analysis... This may take several minutes."):
        report = run_news_analysis(
            user_query=user_query,
            keywords=keywords
        )
    
    return report

def manual_analysis():
    """Manual news analysis without Reddit integration"""
    st.subheader("Manual News Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_query = st.text_input(
    "News Topic to Analyze:",
    placeholder="Enter the news topic you want to analyze...",
    key="user_query_1"
)
        
        keywords_input = st.text_area(
            "Additional Keywords (one per line):",
            placeholder="keyword1\nkeyword2\nkeyword3",
            key="keywords_input_1"
        )
        
    with col2:
        urls_input = st.text_area(
            "Specific URLs to analyze (one per line):",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            key="urls_input_1"
        )
        
        hashtags_input = st.text_input(
            "Hashtags to track (comma-separated):",
            placeholder="#news, #breaking, #analysis",
            key="hashtags_input_1"
        )
    
    if st.button("Run Manual Analysis", type="primary", key="run_manual_analysis_1"):
        if not user_query.strip():
            st.error("Please enter a news topic to analyze.")
            return None
        
        # Parse inputs
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()] if keywords_input.strip() else None
        urls = [u.strip() for u in urls_input.split('\n') if u.strip()] if urls_input.strip() else None
        hashtags = [h.strip() for h in hashtags_input.split(',') if h.strip()] if hashtags_input.strip() else None
        
        # Run analysis
        with st.spinner("Running news analysis... This may take several minutes."):
            report = run_news_analysis(
                user_query=user_query.strip(),
                urls=urls,
                hashtags=hashtags,
                keywords=keywords
            )
        
        return report
    
    return None

def main():
    try:
        st.title("VerifAI: News Analysis Tool")
        
        # Sidebar
        st.sidebar.header("About")
        st.sidebar.markdown(
            """
            This tool analyzes news content to extract insights, detect propaganda techniques,
            and identify misinformation patterns. You can analyze Reddit posts to generate
            comprehensive reports.
            """
        )
        st.sidebar.expander("Configure API Keys").markdown(
            """
            To use this tool, you need to configure your Gemini API key.
            You can get one at [Google AI Studio](https://aistudio.google.com/app/apikey).
            """
        )
        # Check Gemini status
        gemini_status, gemini_msg = check_gemini_connection()
        if gemini_status:
            st.success(f"✅ {gemini_msg}")
        else:
            st.error(f"❌ {gemini_msg}")
            st.info("Please set your GEMINI_API_KEY in your environment variables or .env file.")
        
        # API key setup in the sidebar
        with st.sidebar.expander("Configure API Keys"):
            current_key = os.environ.get("SERPER_API_KEY", "")
            serper_api_key = st.text_input(
                "Serper API Key", 
                value=current_key if current_key != "dummy-key-for-ollama" else "",
                type="password",
                help="Required for web search functionality",
                key="serper_api_key_1"
            )
            
            if st.button("Save API Keys", key="save_api_keys_1"):
                if serper_api_key.strip():
                    os.environ["SERPER_API_KEY"] = serper_api_key.strip()
                    st.success("API key saved!")
                else:
                    st.error("Please enter a valid Serper API key")
        
        # Reddit Analysis interface
        st.header("Reddit Post Analysis")
        st.markdown("Analyze a Reddit post to understand news patterns and credibility.")
        
        url = st.text_input(
            "Enter a Reddit URL:", 
            placeholder="https://www.reddit.com/r/news/comments/...",
            help="Paste a link to a Reddit post you want to analyze",
            key="url_1"
        )
        
        if st.button("Analyze Reddit Post", type="primary", key="analyze_reddit_post_1"):
            if not url:
                st.error("Please enter a Reddit URL.")
            elif not is_reddit_url(url):
                st.error("Invalid Reddit URL. Please enter a valid Reddit URL.")
            elif not setup_api_keys():
                st.error("API keys not set or invalid. Please set valid API keys in the sidebar.")
            else:
                try:
                    report = analyze_reddit_post(url)
                    
                    if report:
                        st.success("Analysis completed!")
                        
                        # Display report
                        st.divider()
                        display_report(report)
                        
                        # Download button
                        st.divider()
                        markdown_report = get_report_as_markdown(report)
                        st.download_button(
                            label="📥 Download Report as Markdown",
                            data=markdown_report,
                            file_name=f"reddit_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error("Failed to generate report.")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error("If this error persists, check your API keys and Ollama setup.")
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

if __name__ == "__main__":
    main()
