from crewai import Crew, Process
import streamlit as st
from agents import create_news_analysis_agents
from tasks import create_news_analysis_tasks
from setup import setup_crewai_config, setup_api_keys, check_ollama_status
import time
import traceback

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
            max_execution_time=900,  # 15 minutes max
            # Disable planning which can cause issues with Ollama
            planning=False,
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
        
        # Handle the result more simply - just use the raw text output
        if hasattr(result, 'raw'):
            final_result = str(result.raw)
        else:
            final_result = str(result)
        
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
        with st.expander("Detailed Error Information"):
            st.code(traceback.format_exc())
        return None

def format_report_for_display(raw_report, user_query):
    """Format the raw text report for better display"""
    if not raw_report:
        return "No report generated"
    
    # Add a header
    formatted_report = f"# News Analysis Report: {user_query}\n\n"
    formatted_report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    formatted_report += "---\n\n"
    formatted_report += raw_report
    
    return formatted_report

def save_report_to_file(report_text, user_query):
    """Save the report to a markdown file"""
    try:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"news_analysis_report_{timestamp}.md"
        
        formatted_report = format_report_for_display(report_text, user_query)
        
        # For Streamlit, we'll provide a download button
        return formatted_report, filename
    except Exception as e:
        st.error(f"Failed to prepare report for download: {e}")
        return None, None

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
        formatted_report, filename = save_report_to_file(report, user_query)
        if formatted_report:
            print(f"Report generated successfully!")
            print(f"Report content:\n{formatted_report}")
        else:
            print("Failed to format report")
    else:
        print("Failed to generate report")

if __name__ == "__main__":
    main()