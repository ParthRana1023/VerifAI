from crewai import Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from setup import setup_crewai_config, check_gemini_status, get_llm
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

def create_news_analysis_agents():
    # Setup CrewAI configuration first
    setup_crewai_config()
    
    # Check Gemini status
    gemini_ok, gemini_msg = check_gemini_status()
    if not gemini_ok:
        st.error(f"Gemini issue: {gemini_msg}")
        return None
    
    # Initialize LLM with proper error handling
    llm = get_llm()
    if not llm:
        if 'st' in globals():
            st.error("Failed to initialize LLM")
        else:
            print("Failed to initialize LLM")
        return None

    try:
        # Initialize tools with error handling and timeout configurations
        serper_tool = SerperDevTool()
        scrape_tool = ScrapeWebsiteTool()
        search_tool = WebsiteSearchTool()
        
        return [
            Agent(
                role="Web Crawler",
                goal="Quickly find 3-5 recent news articles about the query using search tools only",
                backstory="An efficient web crawler that focuses on finding the most relevant recent articles quickly without deep scraping.",
                tools=[serper_tool, scrape_tool, search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                step_callback=None,
                system_message="Focus only on finding article titles, URLs, and sources. Do not analyze content deeply. Limit to 3-5 articles maximum."
            ),
            Agent(
                role="News Content Analyst",
                goal="Quickly analyze the main themes from article titles and summaries only",
                backstory="A fast content analyst who works with article titles, headlines, and brief summaries to extract key themes without deep content analysis.",
                tools=[serper_tool, scrape_tool, search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                step_callback=None,
                system_message="Analyze only headlines and brief summaries. Do not scrape full article content. Focus on identifying 5-7 key themes quickly."
            ),
            Agent(
                role="Social Media Tracking Specialist", 
                goal="Quickly identify trending hashtags and basic sentiment using search only",
                backstory="A social media expert who uses search tools to quickly identify popular hashtags and general sentiment without deep analysis.",
                tools=[serper_tool, scrape_tool, search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                step_callback=None,
                system_message="Find 3-5 popular hashtags and general sentiment quickly. Do not perform deep social media analysis."
            ),
            Agent(
                role="Data Organizer",
                goal="Organize the collected information into structured format", 
                backstory="A data organization specialist who structures information efficiently without additional research.",
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                step_callback=None,
                system_message="Only organize and structure data provided by other agents. Do not conduct additional research."
            ),
            Agent(
                role="Basic Reliability Assessor",
                goal="Provide basic reliability assessment of sources without deep investigation",
                backstory="A reliability assessor who provides quick, basic credibility checks based on well-known source reputations.",
                tools=[serper_tool, scrape_tool, search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                step_callback=None,
                system_message="Provide basic reliability scores based on common knowledge of source credibility. Do not conduct deep verification research."
            ),
            Agent(
                role="Report Compiler",
                goal="Compile all findings into the required JSON report format",
                backstory="A report writer who efficiently compiles analysis into structured JSON format without additional research.",
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                step_callback=None,
                system_message="Compile provided information into the required JSON schema. Do not conduct additional research or analysis."
            )
        ]
    except Exception as e:
        if 'st' in globals():
            st.error(f"Failed to create agents: {e}")
        else:
            print(f"Failed to create agents: {e}")
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_news_api(query):
    try:
        response = requests.get(API_ENDPOINT, 
            params={'q': query},
            headers={'Authorization': f'Bearer {os.getenv("API_KEY")}'},
            timeout=15
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"API Error: {str(e)}", exc_info=True)
        return None