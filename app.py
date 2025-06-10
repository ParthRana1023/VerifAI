from crewai import Crew, Process
import streamlit as st
from agents import create_news_analysis_agents
from tasks import create_news_analysis_tasks
from models import SourceReliability, SocialMediaMetrics, ContentAnalysisMetrics, TimeSeriesData, PropagandaTechnique, MisinformationIndicator, CoordinationPattern, BotActivityMetrics, FakeNewsSite, EnhancedPropagandaAnalysis, NewsAnalysisReport
from save_report import save_report_to_file
from setup import setup_crewai_config, setup_api_keys

def create_news_analysis_crew(user_query, urls=None, hashtags=None, keywords=None):
    # Setup CrewAI configuration
    setup_crewai_config()
    
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
            verbose=True
        )
    except Exception as e:
        st.error(f"Failed to create crew: {e}")
        return None

def run_news_analysis(user_query, urls=None, hashtags=None, keywords=None):
    try:
        # Ensure configuration is set up
        setup_crewai_config()
        
        crew = create_news_analysis_crew(user_query, urls, hashtags, keywords)
        if not crew:
            st.error("Failed to create analysis crew")
            return None
            
        result = crew.kickoff(inputs={
            'query': user_query, 
            'urls': urls or [], 
            'hashtags': hashtags or [], 
            'keywords': keywords or []
        })
        
        return result.pydantic if hasattr(result, 'pydantic') else result
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        # Print more detailed error info
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
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