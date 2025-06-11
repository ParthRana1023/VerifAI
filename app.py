from crewai import Crew, Process
import streamlit as st
from agents import create_news_analysis_agents
from tasks import create_news_analysis_tasks
from models import NewsAnalysisReport
from save_report import save_report_to_file
from setup import setup_crewai_config, setup_api_keys, check_ollama_status
import time

def create_news_analysis_crew(user_query, urls=None, hashtags=None, keywords=None):
    # Setup CrewAI configuration
    setup_crewai_config()
    
    # Check Ollama first
    ollama_ok, ollama_msg = check_ollama_status()
    if not ollama_ok:
        st.error(f"Ollama Error: {ollama_msg}")
        return None
    
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
            # Add timeout to prevent hanging
            max_execution_time=1800,  # 30 minutes max
            # Disable planning which can cause issues
            planning=False
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
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Setting up analysis crew...")
        progress_bar.progress(10)
        
        crew = create_news_analysis_crew(user_query, urls, hashtags, keywords)
        if not crew:
            st.error("Failed to create analysis crew")
            return None
        
        status_text.text("Starting analysis...")
        progress_bar.progress(20)
        
        # Prepare inputs
        inputs = {
            'query': user_query.strip(), 
            'urls': urls or [], 
            'hashtags': hashtags or [], 
            'keywords': keywords or []
        }
        
        status_text.text("Running news analysis (this may take several minutes)...")
        progress_bar.progress(30)
        
        # Run the crew with timeout
        start_time = time.time()
        result = crew.kickoff(inputs=inputs)
        
        progress_bar.progress(80)
        status_text.text("Processing results...")
        
        # Handle different result types
        if hasattr(result, 'pydantic') and result.pydantic:
            final_result = result.pydantic
        elif hasattr(result, 'raw') and result.raw:
            # Try to parse raw result
            try:
                # If it's already a NewsAnalysisReport instance
                if isinstance(result.raw, NewsAnalysisReport):
                    final_result = result.raw
                else:
                    # Try to create from string
                    final_result = create_fallback_report(user_query, str(result.raw))
            except Exception as e:
                st.warning(f"Could not parse structured result: {e}")
                final_result = create_fallback_report(user_query, str(result.raw))
        else:
            # Fallback for any other result type
            final_result = create_fallback_report(user_query, str(result))
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Clear progress indicators after a moment
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return final_result
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        # Print more detailed error info for debugging
        import traceback
        with st.expander("Detailed Error Information"):
            st.code(traceback.format_exc())
        return None

def create_fallback_report(user_query, raw_output):
    """Create a fallback report when Pydantic parsing fails"""
    try:
        return NewsAnalysisReport(
            query_summary=user_query,
            key_findings=f"Analysis completed for: {user_query}\n\nRaw output:\n{raw_output[:1000]}...",
            related_articles=[{"Fallback Article": "No structured articles available"}],
            related_words=["analysis", "news", "content"],
            topic_clusters=[{"topic": "General Analysis", "size": 1, "related_narratives": ["Basic analysis completed"]}],
            top_sources=[],
            top_hashtags=[],
            similar_posts_time_series=[],
            fake_news_sites=[],
            content_analysis=ContentAnalysisMetrics(
                language_percentage=0.0,
                coordination_percentage=0.0,
                source_percentage=0.0,
                bot_like_activity_percentage=0.0
            ),
            propaganda_analysis=EnhancedPropagandaAnalysis(
                overall_reliability_score=50.0,
                propaganda_techniques=[],
                misinformation_indicators=[],
                coordination_patterns=[],
                bot_activity_metrics=BotActivityMetrics(
                    bot_likelihood_score=0.0,
                    account_creation_patterns="No patterns detected",
                    behavioral_indicators=[],
                    network_analysis="No network analysis available"
                ),
                fake_news_sites=[],
                manipulation_timeline=[],
                narrative_fingerprint={},
                cross_verification_results={},
                recommended_verification_steps=["Check multiple sources", "Verify with fact-checkers"]
            ),
            platform_facts=["Analysis was completed with basic tools"],
            cross_source_facts=["Cross-referencing may be limited due to technical constraints"]
        )
    except Exception as e:
        st.error(f"Could not create fallback report: {e}")
        return None

def main():
    setup_crewai_config()
    
    if not setup_api_keys():
        print("Invalid API keys. Exiting.")
        return
    
    user_query = input("Enter news topic to analyze: ")
    urls = input("Enter news URLs (comma-separated, optional): ").split(',') if input("Include specific news URLs? (y/n): ").lower() == 'y' else None
    hashtags = input("Enter hashtags to track (comma-separated, optional): ").split(',') if input("Track specific hashtags? (y/n): ").lower() == 'y' else None
    keywords = input("Enter additional keywords (comma-separated, optional): ").split(',') if input("Include additional keywords? (y/n): ").lower() == 'y' else None
    
    report = run_news_analysis(user_query, urls, hashtags, keywords)
    if report:
        save_report_to_file(report)
    else:
        print("Failed to generate report")

if __name__ == "__main__":
    main()