from crewai import Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from setup import setup_crewai_config, get_llm
import streamlit as st

def create_news_analysis_agents():
    # Setup CrewAI configuration first
    setup_crewai_config()
    
    # Initialize LLM with proper error handling
    llm = get_llm()
    if not llm:
        st.error("Failed to initialize LLM")
        return None

    try:
        return [
            Agent(
                role="Web Crawler",
                goal="Extract news data for the query",
                backstory="An expert web crawler specialized in news sites, capable of identifying reliable sources and extracting relevant articles efficiently.",
                tools=[ScrapeWebsiteTool(), WebsiteSearchTool(), SerperDevTool()],
                llm=llm,
                verbose=True,
                allow_delegation=False
            ),
            Agent(
                role="News Content Analyst",
                goal="Analyze news content in depth",
                backstory="A seasoned journalist with expertise in fact-checking, source reliability assessment, and content analysis who can identify credible sources, biases, and trends in news articles.",
                tools=[ScrapeWebsiteTool(), WebsiteSearchTool(), SerperDevTool()],
                llm=llm,
                verbose=True,
                allow_delegation=False
            ),
            Agent(
                role="Social Media Tracking Specialist",
                goal="Track news spread on social media",
                backstory="A social media expert who specializes in tracking how news spreads across platforms, identifying trending hashtags, measuring engagement, and analyzing sentiment related to news topics.",
                tools=[WebsiteSearchTool(), SerperDevTool()],
                llm=llm,
                verbose=True,
                allow_delegation=False
            ),
            Agent(
                role="News Data Visualization Expert",
                goal="Create data visualizations from news analysis",
                backstory="A data visualization specialist who transforms news analysis data into meaningful visual representations including topic clusters, wordclouds, time series graphs, and reliability charts.",
                tools=[SerperDevTool()],
                llm=llm,
                verbose=True,
                allow_delegation=False
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
                tools=[ScrapeWebsiteTool(), SerperDevTool(), WebsiteSearchTool()],
                llm=llm,
                verbose=True,
                allow_delegation=False
            ),
            Agent(
                role="News Report Generator",
                goal="Compile findings into a comprehensive news analysis report",
                backstory="A professional report writer specialized in organizing complex news analysis data into structured, insightful, and actionable reports with clear visualizations and fact comparisons.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )
        ]
    except Exception as e:
        st.error(f"Failed to create agents: {e}")
        return None