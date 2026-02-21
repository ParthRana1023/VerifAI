from crewai import Agent
from crewai.tools import tool
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from setup import setup_crewai_config, check_llm_status, get_llm
from cache_service import search_cache, get_cache_stats, cache_articles
import streamlit as st
import logging

logger = logging.getLogger(__name__)


@tool("Search Local Cache")
def cached_search_tool(search_query: str) -> str:
    """Search the local article cache for previously scraped articles matching the query.
    Use this tool FIRST before searching the web. Returns cached articles if available."""
    try:
        cached = search_cache(search_query, n_results=5)
        if not cached:
            return "No cached articles found. Please use the web search tool to find articles."

        result_lines = [f"Found {len(cached)} cached articles:\n"]
        for i, article in enumerate(cached, 1):
            result_lines.append(
                f"{i}. Title: {article['title']} | "
                f"Source: {article['source']} | "
                f"URL: {article['url']} | "
                f"Similarity: {article['similarity']}"
            )
        result_lines.append(
            f"\n{'These cached results are sufficient.' if len(cached) >= 3 else 'Fewer than 3 cached results. Consider searching the web for more.'}"
        )
        return "\n".join(result_lines)
    except Exception as e:
        logger.warning("Cache search failed: %s", e)
        return "Cache search failed. Please use the web search tool instead."


class RealtimeCachedScrapeTool(ScrapeWebsiteTool):
    """Scrapes a website and immediately caches the content in ChromaDB."""
    def _run(self, **kwargs) -> str:
        content = super()._run(**kwargs)
        website_url = kwargs.get('website_url', '')
        if website_url and content:
            try:
                source = website_url.split('/')[2] if '//' in website_url else "Web"
                cache_articles("Realtime Web Scrape", [{
                    "title": website_url,
                    "url": website_url,
                    "source": source,
                    "content_snippet": content[:500]
                }])
                logger.info("Cached scraped content from %s", website_url)
            except Exception as e:
                logger.warning("Cache failed for %s: %s", website_url, e)
        return content


def create_news_analysis_agents():
    """Create 4 optimized agents for news analysis with minimal token usage."""
    setup_crewai_config()

    llm_ok, llm_msg = check_llm_status()
    if not llm_ok:
        st.error(f"LLM issue: {llm_msg}")
        return None

    llm = get_llm()
    if not llm:
        st.error("Failed to initialize LLM")
        return None

    # Log cache status
    stats = get_cache_stats()
    logger.info("Cache: %s (%d articles)", stats["status"], stats["count"])

    try:
        serper_tool = SerperDevTool()
        scrape_tool = RealtimeCachedScrapeTool()

        return [
            # Agent 0: News Research Agent (merges Web Crawler + News Content Analyst)
            Agent(
                role="News Researcher",
                goal="Find and analyze 3-5 recent news articles about the query",
                backstory="A skilled news researcher who efficiently finds relevant articles, "
                          "extracts key themes from headlines, and rates source reliability.",
                tools=[cached_search_tool, serper_tool, scrape_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=2,
            ),
            # Agent 1: Social Media Analyst (kept separate per user request)
            Agent(
                role="Social Media Analyst",
                goal="Find trending hashtags and assess public sentiment about the topic",
                backstory="A social media specialist who quickly identifies relevant hashtags, "
                          "engagement levels, and overall public sentiment on a topic.",
                tools=[cached_search_tool, serper_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=2,
            ),
            # Agent 2: Analyst (merges Data Organizer + Reliability Assessor)
            Agent(
                role="Data Analyst",
                goal="Organize findings by reliability and assess overall information quality",
                backstory="An analyst who structures data by source reliability, identifies "
                          "patterns, flags red flags, and suggests verification steps.",
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=2,
            ),
            # Agent 3: Report Compiler
            Agent(
                role="Report Compiler",
                goal="Compile all findings into a structured JSON report",
                backstory="A report writer who efficiently compiles analysis results into "
                          "the required JSON schema format without additional research.",
                llm=llm,
                verbose=True,
                allow_delegation=False,
                memory=False,
                max_iter=2,
            ),
        ]
    except Exception as e:
        st.error(f"Failed to create agents: {e}")
        return None