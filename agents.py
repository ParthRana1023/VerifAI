from crewai import Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from setup import setup_crewai_config, get_llm, check_ollama_status
import streamlit as st

def create_news_analysis_agents():
    # Setup CrewAI configuration first
    setup_crewai_config()
    
    # Check Ollama status
    ollama_ok, ollama_msg = check_ollama_status()
    if not ollama_ok:
        if 'st' in globals():
            st.error(f"Ollama issue: {ollama_msg}")
        else:
            print(f"Ollama issue: {ollama_msg}")
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
        # Initialize tools with error handling
        serper_tool = SerperDevTool()
        scrape_tool = ScrapeWebsiteTool()
        search_tool = WebsiteSearchTool()
        
        return [
            Agent(
                role="Web Crawler",
                goal="Extract news data for the query",
                backstory="An expert web crawler specialized in news sites, capable of identifying reliable sources and extracting relevant articles efficiently.",
                tools=[scrape_tool, search_tool, serper_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                # Disable memory to avoid issues
                memory=False,
                # Add max_iter to prevent infinite loops
                max_iter=3,
                # Set max execution time
                max_execution_time=300  # 5 minutes
            ),
            Agent(
                role="News Content Analyst",
                goal="Analyze news content in depth",
                backstory="A seasoned journalist with expertise in fact-checking, source reliability assessment, and content analysis who can identify credible sources, biases, and trends in news articles.",
                tools=[scrape_tool, search_tool, serper_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=3,
                max_execution_time=300
            ),
            Agent(
                role="Social Media Tracking Specialist", 
                goal="Track news spread on social media",
                backstory="A social media expert who specializes in tracking how news spreads across platforms, identifying trending hashtags, measuring engagement, and analyzing sentiment related to news topics.",
                tools=[search_tool, serper_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=3,
                max_execution_time=300
            ),
            Agent(
                role="News Data Visualization Expert",
                goal="Create data visualizations from news analysis", 
                backstory="A data visualization specialist who transforms news analysis data into meaningful visual representations including topic clusters, wordclouds, time series graphs, and reliability charts.",
                tools=[serper_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=3,
                max_execution_time=300
            ),
            Agent(
                role="Propaganda & Misinformation Analyst",
                goal="Identify and quantify propaganda, misinformation, and coordinated inauthentic behavior in news content",
                backstory="""An expert with advanced training in computational propaganda detection, 
                           misinformation analysis, and network forensics. Specialized in identifying 
                           manipulation techniques, assessing credibility signals, detecting narrative 
                           manipulation, and tracing the spread of false information across media ecosystems.
                           Has experience working with fact-checking organizations and research institutions 
                           on digital media literacy.""",
                tools=[scrape_tool, serper_tool, search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=3,
                max_execution_time=300
            ),
            Agent(
                role="News Report Generator",
                goal="Compile findings into a comprehensive news analysis report",
                backstory="A professional report writer specialized in organizing complex news analysis data into structured, insightful, and actionable reports with clear visualizations and fact comparisons.",
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=3,
                max_execution_time=300
            )
        ]
    except Exception as e:
        if 'st' in globals():
            st.error(f"Failed to create agents: {e}")
        else:
            print(f"Failed to create agents: {e}")
        return None