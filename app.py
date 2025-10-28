from crewai import Crew, Process
import streamlit as st
from agents import create_news_analysis_agents
from tasks import create_news_analysis_tasks
from setup import setup_crewai_config, setup_api_keys, check_gemini_status
import time
import traceback
import json
from models import NewsAnalysisReport
import os

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

def create_news_analysis_crew(user_query, urls=None, hashtags=None, keywords=None):
    # Setup CrewAI configuration
    setup_crewai_config()

    # Check Gemini API key
    gemini_ok, gemini_msg = check_gemini_status()
    if not gemini_ok:
        st.error(f"Gemini API Key Error: {gemini_msg}")
    
    agents = create_news_analysis_agents()
    if not agents:
        st.error("Failed to create agents")
        return None
        
    tasks = create_news_analysis_tasks(agents, user_query, urls, hashtags, keywords)
    if not tasks:
        st.error("Failed to create tasks")
        return None
        
    try:
        return Crew(
            agents=agents, 
            tasks=tasks, 
            process=Process.sequential,
            memory=False,  # Disable memory to avoid potential issues
            verbose=True,
            # Reduced timeout - optimized tasks should complete faster
            max_execution_time=600,  # 10 minutes max (reduced from 15)
            # Disable planning which can cause issues with Ollama
            # planning=False,
            # Disable embedder which can cause issues
            embedder=None
        )
    except Exception as e:
        st.error(f"Failed to create crew: {e}")
        return None

def run_news_analysis(user_query, urls=None, hashtags=None, keywords=None):
    try:
        # Ensure configuration is set up
        setup_crewai_config()
        
        # Validate inputs
        if not user_query or len(user_query.strip()) < 3:
            st.error("Please provide a valid query (at least 3 characters)")
            return None
        
        # Show progress with more detailed steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Initializing analysis system...")
        progress_bar.progress(5)
        
        # Create crew with timeout handling
        crew = create_news_analysis_crew(user_query, urls, hashtags, keywords)
        if not crew:
            st.error("Failed to create analysis crew")
            return None
        
        status_text.text("Crew created successfully. Starting analysis...")
        progress_bar.progress(15)
        
        # Prepare inputs
        inputs = {
            'query': user_query.strip(), 
            'urls': urls or [], 
            'hashtags': hashtags or [], 
            'keywords': keywords or []
        }
        
        # Show estimated time
        status_text.text("Running optimized news analysis (estimated 5-8 minutes)...")
        progress_bar.progress(20)
        
        # Run the crew with better error handling
        start_time = time.time()
        
        # Add progress updates during execution
        try:
            st.info("🔍 Phase 1: Searching for news articles...")
            progress_bar.progress(30)
            
            result = crew.kickoff(inputs=inputs)
            
            elapsed_time = time.time() - start_time
            st.success(f"Analysis completed in {elapsed_time:.1f} seconds!")
            
        except TimeoutError as te:
            st.error("Analysis timed out. This can happen with complex queries or network issues.")
            st.info("Try simplifying your query or checking your internet connection.")
            
            # Provide partial results if possible
            with st.expander("Troubleshooting Tips"):
                st.write("""
                **Common timeout causes:**
                - Complex or very specific queries
                - Network connectivity issues
                - Too many web scraping requests
                
                **Try these solutions:**
                - Use simpler, more general queries
                - Check internet connection
                - Try again in a few minutes
                """)
            return None
        
        progress_bar.progress(80)
        status_text.text("Processing and formatting results...")
        
        # Handle the result with improved error handling and JSON extraction
        try:
            # Get the raw result
            if hasattr(result, 'raw'):
                raw_result = result.raw
            elif hasattr(result, 'json'):
                raw_result = result.json
            else:
                raw_result = str(result)

            # Extract JSON from potentially markdown-wrapped response
            json_string = extract_json_from_response(raw_result)
            
            if not json_string:
                raise ValueError("No JSON content found in response")
            
            # Clean the JSON string
            json_string = clean_json_string(json_string)
            
            # Debug: Show what we're trying to parse
            with st.expander("Debug: Raw JSON being parsed"):
                st.code(json_string[:500] + "..." if len(json_string) > 500 else json_string)
            
            # Try to parse as JSON
            try:
                parsed_json = json.loads(json_string)
                final_result = NewsAnalysisReport.model_validate(parsed_json)
                st.success("✅ Successfully parsed structured report!")
                
            except json.JSONDecodeError as je:
                st.warning(f"JSON parsing failed: {je}")
                st.info("Attempting alternative parsing methods...")
                
                # Try using model_validate_json directly
                try:
                    final_result = NewsAnalysisReport.model_validate_json(json_string)
                    st.success("✅ Successfully parsed with alternative method!")
                except Exception as e2:
                    st.warning(f"Alternative parsing also failed: {e2}")
                    raise e2
            
        except Exception as e:
            st.warning(f"Could not parse report into structured format: {e}")
            st.info("Creating fallback report from raw analysis results...")
            
            # Show the raw result for debugging
            with st.expander("Debug: Raw Result"):
                st.text(str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result))
            
            # Create a simple fallback report structure
            final_result = {
                'query_summary': f"Analysis for: {user_query}",
                'key_findings': str(result)[:500] + "..." if len(str(result)) > 500 else str(result),
                'related_articles': [],
                'related_words': user_query.split(),
                'topic_clusters': [{'topic': 'General Analysis', 'size': 1, 'related_narratives': ['Raw analysis output']}],
                'top_sources': [],
                'top_hashtags': [],
                'similar_posts_time_series': [],
                'fake_news_sites': [],
                'content_analysis': {
                    'language_percentage': 0.0,
                    'coordination_percentage': 0.0,
                    'source_percentage': 0.0,
                    'bot_like_activity_percentage': 0.0
                },
                'propaganda_analysis': {
                    'overall_reliability_score': 50.0,
                    'propaganda_techniques': [],
                    'misinformation_indicators': [],
                    'coordination_patterns': [],
                    'bot_activity_metrics': {
                        'bot_likelihood_score': 0.0,
                        'account_creation_patterns': 'Unknown',
                        'behavioral_indicators': [],
                        'network_analysis': 'Analysis incomplete'
                    },
                    'fake_news_sites': [],
                    'manipulation_timeline': [],
                    'narrative_fingerprint': {},
                    'cross_verification_results': {},
                    'recommended_verification_steps': ['Check multiple sources', 'Verify with fact-checkers']
                },
                'platform_facts': ['Analysis incomplete due to parsing issues'],
                'cross_source_facts': ['Please refer to raw output above'],
                'analysis_note': f'Raw output due to parsing issues: {str(e)}'
            }
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Clear progress indicators after a moment
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return final_result
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        
        # Provide more specific error guidance
        if "TimeoutError" in str(e):
            st.info("💡 **Timeout occurred.** Try these solutions:")
            st.write("- Use a simpler, more specific query")
            st.write("- Check your internet connection")
            st.write("- Try again in a few minutes")
        elif "connection" in str(e).lower():
            st.info("💡 **Connection issues detected.** Check:")
            st.write("- Internet connectivity")
            st.write("- API key configurations")
        elif "json" in str(e).lower():
            st.info("💡 **JSON parsing issues detected.** This usually means:")
            st.write("- The AI model returned malformed JSON")
            st.write("- Try running the analysis again")
            st.write("- Consider simplifying your query")
        
        # Print detailed error info for debugging
        with st.expander("Detailed Error Information (for debugging)"):
            st.code(traceback.format_exc())
        return None

def get_report_as_markdown(report):
    """Convert report to markdown format for download"""
    if not report:
        return "# No Report Generated\n\nThe analysis did not produce a report."
    
    # If report is just a string, return it as-is
    if isinstance(report, str):
        return f"# News Analysis Report\n\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n{report}"
    
    # Handle dictionary fallback format
    if isinstance(report, dict) and 'analysis_note' in report:
        markdown_content = []
        markdown_content.append(f"# News Analysis Report: {report.get('query_summary', 'Unknown')}")
        markdown_content.append(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        markdown_content.append("---\n")
        markdown_content.append("## Key Findings\n")
        markdown_content.append(f"{report.get('key_findings', 'No findings available')}\n")
        markdown_content.append("---\n")
        markdown_content.append(f"*Note: {report.get('analysis_note', 'Standard analysis')}*")
        return "\n".join(markdown_content)
    
    try:
        # Try to format structured report
        markdown_content = []
        markdown_content.append(f"# News Analysis Report: {getattr(report, 'query_summary', 'Unknown Topic')}")
        markdown_content.append(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        markdown_content.append("---\n")
        
        # Key Findings & Summary
        markdown_content.append("## Key Findings & Summary\n")
        markdown_content.append(f"{getattr(report, 'key_findings', 'No key findings available')}\n")
        
        # Related Articles
        markdown_content.append("## Related Articles\n")
        related_articles = getattr(report, 'related_articles', [])
        if related_articles:
            for article in related_articles:
                if isinstance(article, dict):
                    title = article.get('title', 'Unknown Title')
                    url = article.get('url', '#')
                    markdown_content.append(f"- [{title}]({url})")
        else:
            markdown_content.append("No related articles found")
        markdown_content.append("")
        
        # Related Words
        markdown_content.append("## Related Keywords\n")
        related_words = getattr(report, 'related_words', [])
        if related_words:
            markdown_content.append(", ".join(related_words))
        else:
            markdown_content.append("No related words found")
        markdown_content.append("")
        
        # Topic Clusters
        markdown_content.append("## Topic Analysis\n")
        topic_clusters = getattr(report, 'topic_clusters', [])
        if topic_clusters:
            for cluster in topic_clusters:
                if isinstance(cluster, dict):
                    topic = cluster.get('topic', 'Unknown Topic')
                    size = cluster.get('size', 0)
                    markdown_content.append(f"- **{topic}** (Relevance Score: {size})")
        else:
            markdown_content.append("No topic analysis available")
        markdown_content.append("")
        
        # Top Sources
        markdown_content.append("## Source Analysis\n")
        top_sources = getattr(report, 'top_sources', [])
        if top_sources:
            markdown_content.append("| Domain | Reliability | Articles | Engagement |\n")
            markdown_content.append("|--------|-------------|----------|------------|\n")
            for source in top_sources:
                domain = getattr(source, 'domain', 'N/A')
                factual = getattr(source, 'factual_rating', 'N/A')
                articles = getattr(source, 'articles_count', 0)
                engagement = getattr(source, 'engagement', 0)
                markdown_content.append(f"| {domain} | {factual} | {articles} | {engagement} |\n")
        else:
            markdown_content.append("No source analysis available")
        markdown_content.append("")
        
        # Analysis Summary
        markdown_content.append("## Summary\n")
        markdown_content.append("This report was generated using automated AI analysis tools. ")
        markdown_content.append("Results should be verified with additional sources for critical decisions.")
        
        return "\n".join(markdown_content)
        
    except Exception as e:
        # Fallback to string representation
        return f"# News Analysis Report\n\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n{str(report)}\n\n---\n\n*Note: Error formatting structured report: {str(e)}*"

def save_report_to_file(report, user_query):
    """Save the report to a markdown file"""
    try:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"news_analysis_report_{timestamp}.md"
        
        # Use the improved get_report_as_markdown function
        formatted_report = get_report_as_markdown(report)
        
        # For Streamlit, we'll provide a download button
        return formatted_report, filename
    except Exception as e:
        st.error(f"Failed to prepare report for download: {e}")
        return None, None