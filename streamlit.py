import streamlit as st
import pandas as pd
import os
from datetime import datetime
import logging
from app import run_news_analysis, setup_api_keys, setup_crewai_config
from reddit import scrape_reddit_data, extract_keywords, is_reddit_url

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Ollama is running"
        else:
            return False, f"Ollama returned status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"

def display_report(report):
    """Display the news analysis report in the Streamlit interface"""
    if not report:
        st.error("No report to display")
        return
        
    try:
        # Key Findings & Summary
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
        st.write("*Wordcloud visualization would show these terms with size relative to frequency:*")
        related_words = getattr(report, 'related_words', [])
        if related_words:
            st.write(", ".join(related_words))
        else:
            st.write("No related words found")
        
        # Topic Clusters
        st.header("Related Topic Clusters")
        st.write("*Visualization would show bubbles with sizes relative to prevalence:*")
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
                    "Engagement Rate (%)": getattr(hashtag, 'engagement_rate', 0),
                    "Reach": getattr(hashtag, 'reach', 0),
                    "Sentiment": getattr(hashtag, 'sentiment', 'N/A')
                })
            st.dataframe(pd.DataFrame(hashtags_data))
        else:
            st.write("No hashtag data available")
        
        # Time Series Graph
        st.header("Similar Posts Spread Over Time")
        st.write("*Time series visualization would show:*")
        time_series = getattr(report, 'similar_posts_time_series', [])
        if time_series:
            time_series_data = []
            for data_point in time_series:
                time_series_data.append({
                    "Date": getattr(data_point, 'date', 'N/A'),
                    "Count": getattr(data_point, 'count', 0)
                })
            st.dataframe(pd.DataFrame(time_series_data))
            
            # Optional: Create a line chart if there's enough data
            if len(time_series_data) > 1:
                try:
                    df = pd.DataFrame(time_series_data)
                    st.line_chart(df.set_index("Date"))
                except Exception as e:
                    st.write(f"Could not create chart: {e}")
        else:
            st.write("No time series data available")
        
        # Fake News Sites
        st.header("Most Shared Fake News Sites")
        st.write("*Line chart visualization would show:*")
        fake_news_sites = getattr(report, 'fake_news_sites', [])
        if fake_news_sites:
            fake_news_data = []
            for site in fake_news_sites:
                if isinstance(site, dict):
                    site_name = site.get('site', 'N/A')
                    shares = site.get('shares', 0)
                    fake_news_data.append({
                        "Site": site_name,
                        "Shares": shares
                    })
            if fake_news_data:
                st.dataframe(pd.DataFrame(fake_news_data))
        else:
            st.write("No fake news site data available")
        
        # Content Analysis Metrics
        st.header("Content Analysis Metrics")
        content_analysis = getattr(report, 'content_analysis', None)
        if content_analysis:
            st.write("*Percentage bars visualization would show:*")
            
            # Create a DataFrame for the metrics
            metrics_data = {
                'Metric': ['Language', 'Coordination', 'Source', 'Bot-like activity'],
                'Percentage': [
                    getattr(content_analysis, 'language_percentage', 0),
                    getattr(content_analysis, 'coordination_percentage', 0),
                    getattr(content_analysis, 'source_percentage', 0),
                    getattr(content_analysis, 'bot_like_activity_percentage', 0)
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            
            # Display as a bar chart
            st.bar_chart(metrics_df.set_index('Metric'))
        else:
            st.write("No content analysis data available")
        
        # Propaganda Analysis
        st.header("Propaganda and Misinformation Analysis")
        propaganda_analysis = getattr(report, 'propaganda_analysis', None)
        if propaganda_analysis:
            reliability_score = getattr(propaganda_analysis, 'overall_reliability_score', 0)
            st.subheader(f"Overall Reliability Score: {reliability_score}/100")
            
            # Propaganda Techniques
            st.subheader("Propaganda Techniques Detected")
            techniques = getattr(propaganda_analysis, 'propaganda_techniques', [])
            if techniques:
                techniques_data = []
                for technique in techniques:
                    techniques_data.append({
                        "Technique": getattr(technique, 'technique_name', 'N/A'),
                        "Frequency": getattr(technique, 'frequency', 0),
                        "Severity (0-10)": getattr(technique, 'severity', 0),
                        "Example": getattr(technique, 'example', 'N/A')
                    })
                st.dataframe(pd.DataFrame(techniques_data))
                
                st.write("*Explanation of techniques:*")
                for technique in techniques:
                    name = getattr(technique, 'technique_name', 'N/A')
                    explanation = getattr(technique, 'explanation', 'No explanation available')
                    st.markdown(f"- **{name}**: {explanation}")
            else:
                st.write("No propaganda techniques detected")
        else:
            st.write("No propaganda analysis data available")
        
        # Facts Comparison
        st.header("Facts Gathered from Platform")
        platform_facts = getattr(report, 'platform_facts', [])
        if platform_facts:
            for fact in platform_facts:
                st.markdown(f"- {fact}")
        else:
            st.write("No platform facts available")
        
        st.header("Facts Gathered from Relevant Sources")
        cross_source_facts = getattr(report, 'cross_source_facts', [])
        if cross_source_facts:
            for fact in cross_source_facts:
                st.markdown(f"- {fact}")
        else:
            st.write("No cross-source facts available")
            
    except Exception as e:
        st.error(f"Error displaying report: {str(e)}")
        st.write("Raw report data:")
        st.json(str(report))

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
        serper_api_key = st.text_input("Serper API Key", type="password")
        
        if st.button("Save API Keys"):
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