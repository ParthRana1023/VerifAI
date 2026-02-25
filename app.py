from crewai import Crew, Process
import streamlit as st
from agents import create_news_analysis_agents
from tasks import create_news_analysis_tasks
from setup import setup_crewai_config, setup_api_keys, check_llm_status
import time
import traceback
import json
import re
from logger_config import get_logger
from models import NewsAnalysisReport
from cache_service import cache_articles
import os

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

_logger = get_logger(__name__)


def normalize_report_scores(report):
    """Fix common LLM scale mismatches in the parsed report.

    Works on both dict and Pydantic model instances.
    """
    def _set(obj, key, value):
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # --- Propaganda analysis fixes ---
    propaganda = _get(report, "propaganda_analysis")
    if propaganda:
        # overall_reliability_score: expected 0-100, LLM may return 0-1
        reliability = _get(propaganda, "overall_reliability_score", 0)
        if isinstance(reliability, (int, float)) and 0 < reliability <= 1.0:
            _set(propaganda, "overall_reliability_score", round(reliability * 100, 1))

        # bot_likelihood_score: expected 0-1, LLM may return 0-100
        bot_metrics = _get(propaganda, "bot_activity_metrics")
        if bot_metrics:
            bot_score = _get(bot_metrics, "bot_likelihood_score", 0)
            if isinstance(bot_score, (int, float)) and bot_score > 1.0:
                _set(bot_metrics, "bot_likelihood_score", round(bot_score / 100, 4))

        # narrative_fingerprint: expected 0-1, LLM may return 0-100
        fingerprint = _get(propaganda, "narrative_fingerprint", {})
        if fingerprint:
            corrected = {}
            for key, val in (fingerprint.items() if isinstance(fingerprint, dict) else []):
                if isinstance(val, (int, float)) and val > 1.0:
                    corrected[key] = round(val / 100, 4)
                else:
                    corrected[key] = val
            _set(propaganda, "narrative_fingerprint", corrected)

    return report


def inject_platform_metrics(report, platform_metrics):
    """Overwrite LLM-generated zero metrics with real platform data.

    ``platform_metrics`` is expected to come from
    ``reddit.compute_reddit_engagement()`` and/or ``trends_service``.
    """
    if not platform_metrics:
        return report

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _set(obj, key, value):
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)

    # Inject engagement into top_hashtags that have zero values
    engagement_rate = platform_metrics.get("engagement_rate", 0)
    reach = platform_metrics.get("reach", 0)
    sentiment = platform_metrics.get("sentiment_from_ratio", "Neutral")

    top_hashtags = _get(report, "top_hashtags", [])
    if top_hashtags:
        for ht in top_hashtags:
            if _get(ht, "engagement_rate", 0) == 0 and engagement_rate > 0:
                _set(ht, "engagement_rate", engagement_rate)
            if _get(ht, "reach", 0) == 0 and reach > 0:
                _set(ht, "reach", reach)
            if _get(ht, "sentiment", "Neutral") == "Neutral" and sentiment != "Neutral":
                _set(ht, "sentiment", sentiment)

    # Inject into source engagement
    total_interactions = platform_metrics.get("total_interactions", 0)
    top_sources = _get(report, "top_sources", [])
    if top_sources and total_interactions > 0:
        per_source = total_interactions // max(len(top_sources), 1)
        for source in top_sources:
            if _get(source, "engagement", 0) == 0:
                _set(source, "engagement", per_source)

    # Inject Google Trends time series if available
    trends_ts = platform_metrics.get("trends_time_series", [])
    if trends_ts:
        existing_ts = _get(report, "similar_posts_time_series", [])
        if not existing_ts:
            _set(report, "similar_posts_time_series", trends_ts)

    return report


def extract_json_from_response(response_text: str) -> str | None:
    """Extract JSON content from a potentially markdown-wrapped LLM response.
    
    Handles responses wrapped in ```json ... ``` code fences as well as
    bare JSON objects.
    """
    if not response_text:
        return None

    text = response_text.strip()

    # 1. Try to extract from markdown code fences  (```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # 2. Try to find the outermost { ... } block
    first_brace = text.find("{")
    if first_brace != -1:
        # Walk backwards from end to find the matching closing brace
        last_brace = text.rfind("}")
        if last_brace > first_brace:
            return text[first_brace : last_brace + 1]

    return None


def clean_json_string(json_string: str) -> str:
    """Fix common JSON formatting issues produced by LLMs.

    Applies multiple repair passes:
    1. Remove trailing commas before } or ]
    2. Add missing commas between adjacent elements
    3. Strip control characters
    4. Fix unescaped quotes inside string values
    5. Replace single quotes with double quotes (outside strings)
    """
    if not json_string:
        return json_string

    # Pass 1: Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", json_string)

    # Pass 2: Replace literal \r\n with \\n (unescaped newlines in strings)
    cleaned = cleaned.replace("\r\n", "\\n").replace("\r", "\\n")

    # Pass 3: Add missing commas between adjacent JSON elements.
    # This fixes the exact error: "Expecting ',' delimiter"
    # Pattern: "value"\n"key" → "value",\n"key"  (string → string)
    cleaned = re.sub(r'"\s*\n(\s*")', r'",\n\1', cleaned)
    # Pattern: }\n{ or ]\n{ or }\n" etc.
    cleaned = re.sub(r'(\})\s*\n(\s*\{)', r'\1,\n\2', cleaned)
    cleaned = re.sub(r'(\])\s*\n(\s*[\{"])', r'\1,\n\2', cleaned)
    cleaned = re.sub(r'(\})\s*\n(\s*")', r'\1,\n\2', cleaned)
    # Pattern: number/true/false/null followed by newline then "key"
    cleaned = re.sub(r'(\d|true|false|null)\s*\n(\s*")', r'\1,\n\2', cleaned)

    # Pass 4: Remove any trailing commas we may have just re-introduced
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Pass 5: Remove non-printable control characters (except \n, \t)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)

    return cleaned


def _try_repair_json(json_string: str) -> dict | None:
    """Attempt to repair and parse malformed JSON using multiple strategies."""

    # Strategy 1: Try the json_repair library if available
    try:
        import json_repair
        repaired = json_repair.loads(json_string)
        if isinstance(repaired, dict):
            _logger.info("JSON repaired using json_repair library")
            return repaired
    except ImportError:
        pass
    except Exception:
        pass

    # Strategy 2: Try ast.literal_eval (handles single quotes, trailing commas)
    try:
        import ast
        result = ast.literal_eval(json_string)
        if isinstance(result, dict):
            _logger.info("JSON parsed using ast.literal_eval")
            return result
    except Exception:
        pass

    # Strategy 3: Try fixing truncated JSON by closing open brackets
    try:
        fixed = json_string.rstrip()
        # Count open vs close braces/brackets
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')
        # Remove any trailing comma
        fixed = re.sub(r',\s*$', '', fixed)
        # Close any unclosed brackets/braces
        fixed += ']' * max(open_brackets, 0)
        fixed += '}' * max(open_braces, 0)
        result = json.loads(fixed)
        if isinstance(result, dict):
            _logger.info("JSON repaired by closing %d braces and %d brackets",
                        open_braces, open_brackets)
            return result
    except Exception:
        pass

    return None


def create_news_analysis_crew(user_query, urls=None, hashtags=None, keywords=None, llm_provider="gemini", model_name=None):
    # Setup CrewAI configuration
    setup_crewai_config(llm_provider)

    # Check LLM API key
    llm_ok, llm_msg = check_llm_status(llm_provider)
    if not llm_ok:
        st.error(f"LLM API Key Error: {llm_msg}")
    
    agents = create_news_analysis_agents(llm_provider, model_name=model_name)
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
            verbose=False,
            # Reduced timeout - optimized tasks should complete faster
            max_execution_time=300,  # 5 minutes max (reduced from 10)
            # Disable planning which can cause issues with Ollama
            # planning=False,
            # Disable embedder which can cause issues
            embedder=None
        )
    except Exception as e:
        st.error(f"Failed to create crew: {e}")
        return None


def _cache_result_articles(query: str, result):
    """Extract articles from analysis result and cache them in ChromaDB."""
    try:
        articles_to_cache = []

        if isinstance(result, dict):
            # Handle dict-format results
            for article in result.get("related_articles", []):
                if isinstance(article, dict) and article.get("url"):
                    articles_to_cache.append({
                        "title": article.get("title", article.get("source", "Unknown")),
                        "url": article["url"],
                        "source": article.get("source", "Unknown"),
                        "content_snippet": article.get("summary", ""),
                    })
            # Also try key_findings as general content
            if result.get("key_findings") and not articles_to_cache:
                articles_to_cache.append({
                    "title": result.get("query_summary", query),
                    "url": f"analysis://{query.replace(' ', '-')}",
                    "source": "VerifAI Analysis",
                    "content_snippet": str(result["key_findings"])[:500],
                })
        elif hasattr(result, "related_articles"):
            # Handle Pydantic model results
            for article in (result.related_articles or []):
                url = getattr(article, "url", "") or ""
                if url:
                    articles_to_cache.append({
                        "title": getattr(article, "title", "Unknown"),
                        "url": url,
                        "source": getattr(article, "source", "Unknown"),
                        "content_snippet": getattr(article, "summary", ""),
                    })

        if articles_to_cache:
            cached = cache_articles(query, articles_to_cache)
            if cached > 0:
                _logger.info(
                    "Cached %d articles from analysis for '%s'", cached, query
                )
    except Exception as e:
        # Caching failure should never break the analysis
        _logger.warning("Failed to cache articles: %s", e)


def run_news_analysis(user_query, urls=None, hashtags=None, keywords=None, llm_provider="gemini", model_name=None, platform_metrics=None):
    try:
        # Ensure configuration is set up
        setup_crewai_config(llm_provider)
        
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
        crew = create_news_analysis_crew(user_query, urls, hashtags, keywords, llm_provider, model_name=model_name)
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
            
            time.sleep(1)  # Brief buffer before crew kickoff
            
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
                _logger.warning("JSON parsing failed: %s", je)
                st.warning(f"JSON parsing failed: {je}")
                st.info("Attempting repair...")
                
                # Try model_validate_json directly
                try:
                    final_result = NewsAnalysisReport.model_validate_json(json_string)
                    st.success("✅ Successfully parsed with alternative method!")
                except Exception:
                    # Try the multi-strategy repair
                    repaired = _try_repair_json(json_string)
                    if repaired:
                        try:
                            final_result = NewsAnalysisReport.model_validate(repaired)
                            st.success("✅ Successfully repaired and parsed JSON!")
                        except Exception as e3:
                            _logger.warning("Repair succeeded but validation failed: %s", e3)
                            raise e3
                    else:
                        raise je
            
        except Exception as e:
            st.warning(f"Could not parse report into structured format: {e}")
            st.info("Creating fallback report from raw analysis results...")
            
            # Show the raw result for debugging
            with st.expander("Debug: Raw Result"):
                st.text(str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result))
            
            # Create a simple fallback report structure matching NewsAnalysisReport model
            final_result = {
                'query_summary': f"Analysis for: {user_query}",
                'key_findings': str(result)[:500] + "..." if len(str(result)) > 500 else str(result),
                'related_articles': [],
                'related_words': user_query.split(),
                'topic_clusters': [{'cluster_name': 'General Analysis', 'keywords': user_query.split(), 'article_count': 1}],
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
                    'overall_reliability_score': 0.0,
                    'propaganda_techniques': [],
                    'misinformation_indicators': [],
                    'coordination_patterns': [],
                    'bot_activity_metrics': {},
                    'fake_news_sites': [],
                    'manipulation_timeline': [],
                    'narrative_fingerprint': {},
                    'cross_verification_results': {},
                    'recommended_verification_steps': []
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
        
        # Normalize LLM scale mismatches and inject real platform metrics
        final_result = normalize_report_scores(final_result)
        final_result = inject_platform_metrics(final_result, platform_metrics)

        # Cache article data for future queries
        _cache_result_articles(user_query, final_result)

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
    
    try:
        from save_report import generate_report_markdown
        return generate_report_markdown(report)
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