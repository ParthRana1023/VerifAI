import streamlit as st
import pandas as pd
import os
from datetime import datetime
import logging
from app import run_news_analysis, setup_api_keys
from reddit import scrape_reddit_data, extract_keywords, is_reddit_url

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_report(report):
    """
    Display the news analysis report in the Streamlit interface
    """
    # Key Findings & Summary
    st.title(f"News Analysis Report: {report.query_summary}")
    
    st.header("Key Findings & Summary")
    st.write(report.key_findings)
    
    # Related Articles
    st.header("Related Articles")
    for article in report.related_articles:
        for title, url in article.items():
            st.markdown(f"- [{title}]({url})")
    
    # Related Words (wordcloud)
    st.header("Related Words")
    st.write("*Wordcloud visualization would show these terms with size relative to frequency:*")
    st.write(", ".join(report.related_words))
    
    # Topic Clusters
    st.header("Related Topic Clusters")
    st.write("*Visualization would show bubbles with sizes relative to prevalence:*")
    for cluster in report.topic_clusters:
        st.markdown(f"- **{cluster.get('topic', 'N/A')}** (Size: {cluster.get('size', 'N/A')})")
        related_narratives = cluster.get('related_narratives', [])
        if related_narratives:
            st.markdown("  - Related narratives: " + ", ".join(related_narratives))
    
    # Top Sources
    st.header("List of Top Sources")
    sources_data = []
    for source in report.top_sources:
        sources_data.append({
            "Domain": source.domain,
            "Factual Rating": source.factual_rating,
            "Articles Count": source.articles_count,
            "Engagement": source.engagement
        })
    if sources_data:
        st.dataframe(pd.DataFrame(sources_data))
    
    # Top Hashtags
    st.header("Top Hashtags")
    hashtags_data = []
    for hashtag in report.top_hashtags:
        hashtags_data.append({
            "Hashtag": hashtag.hashtag,
            "Engagement Rate (%)": hashtag.engagement_rate,
            "Reach": hashtag.reach,
            "Sentiment": hashtag.sentiment
        })
    if hashtags_data:
        st.dataframe(pd.DataFrame(hashtags_data))
    
    # Time Series Graph
    st.header("Similar Posts Spread Over Time")
    st.write("*Time series visualization would show:*")
    time_series_data = []
    for data_point in report.similar_posts_time_series:
        time_series_data.append({
            "Date": data_point.date,
            "Count": data_point.count
        })
    if time_series_data:
        st.dataframe(pd.DataFrame(time_series_data))
        # Optional: Create a line chart if there's enough data
        if len(time_series_data) > 1:
            df = pd.DataFrame(time_series_data)
            st.line_chart(df.set_index("Date"))
    
    # Fake News Sites
    st.header("Most Shared Fake News Sites")
    st.write("*Line chart visualization would show:*")
    fake_news_data = []
    for site in report.fake_news_sites:
        site_name = site.get('site', 'N/A')
        shares = site.get('shares', 0)
        fake_news_data.append({
            "Site": site_name,
            "Shares": shares
        })
    if fake_news_data:
        st.dataframe(pd.DataFrame(fake_news_data))
    
    # Content Analysis Metrics
    st.header("Content Analysis Metrics")
    st.write("*Percentage bars visualization would show:*")
    
    # Create a DataFrame for the metrics
    metrics_data = {
        'Metric': ['Language', 'Coordination', 'Source', 'Bot-like activity'],
        'Percentage': [
            report.content_analysis.language_percentage,
            report.content_analysis.coordination_percentage,
            report.content_analysis.source_percentage,
            report.content_analysis.bot_like_activity_percentage
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display as a bar chart
    st.bar_chart(metrics_df.set_index('Metric'))
    
    # Propaganda News
    st.header("Propaganda and Misinformation Analysis")
    
    st.subheader(f"Overall Reliability Score: {report.propaganda_analysis.overall_reliability_score}/100")
    
    # Propaganda Techniques
    st.subheader("Propaganda Techniques Detected")
    techniques_data = []
    for technique in report.propaganda_analysis.propaganda_techniques:
        techniques_data.append({
            "Technique": technique.technique_name,
            "Frequency": technique.frequency,
            "Severity (0-10)": technique.severity,
            "Example": technique.example
        })
    if techniques_data:
        st.dataframe(pd.DataFrame(techniques_data))
    
    st.write("*Explanation of techniques:*")
    for technique in report.propaganda_analysis.propaganda_techniques:
        st.markdown(f"- **{technique.technique_name}**: {technique.explanation}")
    
    # Misinformation Indicators
    st.subheader("Misinformation Indicators")
    indicators_data = []
    for indicator in report.propaganda_analysis.misinformation_indicators:
        sources = ", ".join(indicator.source_verification)
        indicators_data.append({
            "Type": indicator.indicator_type,
            "Confidence": f"{indicator.confidence*100:.1f}%",
            "Correction": indicator.correction,
            "Verification Sources": sources
        })
    if indicators_data:
        st.dataframe(pd.DataFrame(indicators_data))
    
    # Coordination Patterns
    st.subheader("Coordination Patterns")
    for pattern in report.propaganda_analysis.coordination_patterns:
        st.markdown(f"**{pattern.pattern_type}** (Strength: {pattern.strength*100:.1f}%)")
        st.markdown(f"- Entities involved: {', '.join(pattern.entities_involved)}")
        st.markdown(f"- Timeline: {pattern.timeline}")
    
    # Bot Activity Metrics
    st.subheader("Bot Activity Metrics")
    bot_metrics = report.propaganda_analysis.bot_activity_metrics
    st.markdown(f"**Bot Likelihood Score: {bot_metrics.bot_likelihood_score*100:.1f}%**")
    st.markdown(f"Account Creation Patterns: {bot_metrics.account_creation_patterns}")
    
    st.markdown("Behavioral Indicators:")
    for indicator in bot_metrics.behavioral_indicators:
        st.markdown(f"- {indicator}")
    
    st.markdown(f"Network Analysis: {bot_metrics.network_analysis}")
    
    # Fake News Sites
    st.subheader("Most Shared Fake News Sites")
    fake_sites_data = []
    for site in report.propaganda_analysis.fake_news_sites:
        failures = ", ".join(site.verification_failures[:2]) + (", ..." if len(site.verification_failures) > 2 else "")
        fake_sites_data.append({
            "Domain": site.domain,
            "Shares": site.shares,
            "Engagement": site.engagement,
            "Known False Stories": site.known_false_stories,
            "Verification Failures": failures
        })
    if fake_sites_data:
        st.dataframe(pd.DataFrame(fake_sites_data))
    
    # Deceptive Practices
    st.subheader("Deceptive Practices by Domain")
    for site in report.propaganda_analysis.fake_news_sites:
        st.markdown(f"**{site.domain}**:")
        for practice in site.deceptive_practices:
            st.markdown(f"- {practice}")
    
    # Information Manipulation Timeline
    st.subheader("Information Manipulation Timeline")
    st.write("*Timeline showing how information evolved and spread:*")
    for entry in report.propaganda_analysis.manipulation_timeline:
        st.markdown(f"- **{entry.get('date', 'N/A')}**: {entry.get('event', 'N/A')}")
    
    # Narrative Fingerprint
    st.subheader("Narrative Fingerprint")
    st.write("*Distinctive narrative patterns and their strength:*")
    narrative_data = []
    for narrative, strength in report.propaganda_analysis.narrative_fingerprint.items():
        narrative_data.append({
            "Narrative": narrative,
            "Strength": f"{strength*100:.1f}%"
        })
    if narrative_data:
        st.dataframe(pd.DataFrame(narrative_data))
    
    # Verification Steps
    st.subheader("How to Verify This Information")
    for i, step in enumerate(report.propaganda_analysis.recommended_verification_steps, 1):
        st.markdown(f"{i}. {step}")
    
    # Facts Comparison
    st.header("Facts Gathered from Platform")
    for fact in report.platform_facts:
        st.markdown(f"- {fact}")
    
    st.header("Facts Gathered from Relevant Sources")
    for fact in report.cross_source_facts:
        st.markdown(f"- {fact}")

def get_report_as_markdown(report):
    """
    Convert the report to markdown format
    """
    markdown = []
    
    # Key Findings & Summary
    markdown.append(f"# News Analysis Report: {report.query_summary}\n\n")
    markdown.append("## Key Findings & Summary\n\n")
    markdown.append(f"{report.key_findings}\n\n")
    
    # Related Articles
    markdown.append("## Related Articles\n\n")
    for article in report.related_articles:
        for title, url in article.items():
            markdown.append(f"- [{title}]({url})\n")
    markdown.append("\n")
    
    # Related Words (wordcloud)
    markdown.append("## Related Words\n\n")
    markdown.append("*Wordcloud visualization would show these terms with size relative to frequency:*\n\n")
    markdown.append(", ".join(report.related_words))
    markdown.append("\n\n")
    
    # Topic Clusters
    markdown.append("## Related Topic Clusters\n\n")
    markdown.append("*Visualization would show bubbles with sizes relative to prevalence:*\n\n")
    for cluster in report.topic_clusters:
        markdown.append(f"- **{cluster.get('topic', 'N/A')}** (Size: {cluster.get('size', 'N/A')})\n")
        related_narratives = cluster.get('related_narratives', [])
        if related_narratives:
            markdown.append("  - Related narratives: " + ", ".join(related_narratives) + "\n")
    markdown.append("\n")
    
    # Add the rest of the report sections...
    # This is truncated for brevity, but should include all sections from the display_report function
    
    return "".join(markdown)

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
    user_query = "News analysis for: " + ", ".join(keyword_list[:5])  # Top 5 keywords
    
    # Run analysis
    with st.spinner("Running news analysis... This may take several minutes."):
        report = run_news_analysis(
            user_query=user_query,
            keywords=keyword_list
        )
    
    return report

def main():
    st.set_page_config(
        page_title="VerifAI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("VerifAI: A Reddit News Analysis Tool")
    
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This tool analyzes news-related Reddit posts to extract insights, detect propaganda techniques, 
    and identify misinformation patterns. Enter a Reddit post URL below to generate a comprehensive 
    news analysis report.
    """)
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Enter a Reddit post URL in the text box
    2. Click "Analyze" to start the analysis
    3. Wait for the results (this may take several minutes)
    4. View the analysis report directly in the interface
    5. Download the report as a markdown file if needed
    """)
    
    # API key setup in the sidebar
    with st.sidebar.expander("Configure API Keys"):
        llm_provider = st.radio("LLM Provider", ["Groq", "OpenAI"])
        
        if llm_provider == "Groq":
            groq_api_key = st.text_input("Groq API Key", type="password")
        else:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            
        serper_api_key = st.text_input("Serper API Key", type="password")
        
        if st.button("Save API Keys"):
            if llm_provider == "Groq":
                os.environ["GROQ_API_KEY"] = groq_api_key
            else:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                
            os.environ["SERPER_API_KEY"] = serper_api_key
            st.success("API keys saved!")
    
    # URL Input
    url = st.text_input("Enter a Reddit URL:", placeholder="https://www.reddit.com/r/news/comments/...")
    
    if st.button("Analyze"):
        if not url:
            st.error("Please enter a Reddit URL.")
            return
        
        if not is_reddit_url(url):
            st.error("Invalid Reddit URL. Please enter a valid URL.")
            return
        
        # Setup API keys
        if not setup_api_keys():
            st.error("API keys not set or invalid. Please set valid API keys in the sidebar.")
            return
        
        # Start analysis
        try:
            report = analyze_reddit_post(url)
            
            if report:
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Report View", "Raw Report"])
                
                with tab1:
                    display_report(report)
                
                with tab2:
                    st.text_area("Raw Report", str(report), height=400)
                
                # Create markdown for download
                markdown_report = get_report_as_markdown(report)
                st.download_button(
                    label="Download Report as Markdown",
                    data=markdown_report,
                    file_name=f"reddit_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            else:
                st.error("Failed to generate report.")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.error("If this error persists, check your API keys and try again.")

if __name__ == "__main__":
    main()