import json
from crewai import Task
from models import NewsAnalysisReport
from typing import List


def model_to_json_template(model_class: type) -> str:
    """Generates a JSON schema template from a Pydantic model."""
    return json.dumps(model_class.model_json_schema(), indent=2)


def create_news_analysis_tasks(agents: List, user_query: str,
                               urls: List[str] = None,
                               hashtags: List[str] = None,
                               keywords: List[str] = None) -> List[Task]:
    if not agents or len(agents) < 4:
        print(f"Expected 4 agents, got {len(agents) if agents else 0}")
        return None

    json_schema_template = model_to_json_template(NewsAnalysisReport)

    return [
        # Task 0: News Research (assigned to Research Agent)
        Task(
            description=f"""Find and analyze 3-5 recent news articles about: {user_query}

            Steps:
            1. Check the local cache first using "Search Local Cache" tool
            2. If fewer than 3 cached articles, search the web for more
            3. For each article, note: title, source domain, URL
            4. Rate each source reliability as High/Medium/Low
            5. Identify 3-5 key themes from headlines
            6. List 5-8 important keywords

            Output format:
            ARTICLES:
            1. Title: [title] | Source: [domain] | URL: [url] | Reliability: [H/M/L]
            (repeat for each article)

            THEMES: [theme1, theme2, theme3]
            KEYWORDS: [word1, word2, word3, word4, word5]
            SUMMARY: [One sentence overview]""",
            agent=agents[0],
            expected_output="Articles list with reliability ratings, themes, keywords, and summary."
        ),

        # Task 1: Social Media Analysis (assigned to Social Media Analyst)
        Task(
            description=f"""Analyze social media presence for: {user_query}

            Steps:
            1. Search for 3-5 relevant hashtags about this topic
            2. Assess engagement level as High/Medium/Low
            3. Determine overall sentiment as Positive/Negative/Neutral/Mixed
            4. Note if the topic is trending

            Output format:
            HASHTAGS: #tag1, #tag2, #tag3
            ENGAGEMENT: [High/Medium/Low]
            SENTIMENT: [Positive/Negative/Neutral/Mixed]
            TRENDING: [Yes/No]""",
            agent=agents[1],
            expected_output="Hashtags, engagement level, sentiment, and trending status."
        ),

        # Task 2: Analysis & Reliability (assigned to Analyst Agent)
        Task(
            description=f"""Organize and assess the findings for: {user_query}

            Using the data from previous tasks:
            1. Group articles by reliability (High/Medium/Low)
            2. Identify patterns across sources
            3. Rate overall information reliability from 1-10
            4. Flag any red flags or contradictions
            5. Suggest 2-3 verification steps

            Output format:
            RELIABILITY GROUPS:
            - High: [sources]
            - Medium: [sources]
            - Low: [sources]

            PATTERNS: [brief summary]
            SCORE: [1-10]/10
            RED FLAGS: [issues or "None"]
            VERIFY: [2-3 steps]""",
            agent=agents[2],
            expected_output="Organized data with reliability groups, patterns, score, and verification steps."
        ),

        # Task 3: Report Compilation (assigned to Report Compiler)
        Task(
            description=f"""Compile all findings into a JSON report for: {user_query}

            Instructions:
            1. Take all information from previous tasks
            2. Fill the JSON template with actual data
            3. Use "Unknown" or "N/A" for missing fields, but try to infer Bot Metrics and Coordination Patterns if possible based on source reliability.
            4. Ensure valid JSON output

            Context:
            - Query: {user_query}
            - URLs: {urls or 'None'}
            - Keywords: {keywords or 'None'}
            - Hashtags: {hashtags or 'None'}

            OUTPUT MUST BE VALID JSON matching this schema:
            {json_schema_template}

            CRITICAL: Use ACTUAL findings from previous tasks. Do not invent data.""",
            agent=agents[3],
            expected_output="Complete JSON report following the NewsAnalysisReport schema."
        )
    ]