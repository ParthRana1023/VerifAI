import json
from crewai import Task
from models import NewsAnalysisReport
from typing import List
from streamlit.runtime.caching import cache_data

def model_to_json_template(model_class: type) -> str:
    """
    Generates a JSON schema template from a Pydantic model.
    """
    return json.dumps(model_class.model_json_schema(), indent=4)

def create_news_analysis_tasks(agents: List[str], user_query: str,
                               urls: List[str] = None,
                               hashtags: List[str] = None,
                               keywords: List[str] = None) -> List[Task]:
    if not agents or len(agents) < 6:
        print(f"Expected 6 agents, got {len(agents) if agents else 0}")
        return None

    # Generate JSON template dynamically from Pydantic model
    json_schema_template = model_to_json_template(NewsAnalysisReport)
        
    return [
        Task(
            description=f"""QUICK SEARCH: Find 3-5 recent news articles about: {user_query}
                        
            SIMPLE INSTRUCTIONS:
            1. Provide a 1-sentence summary
            2. Find exactly 3-5 article titles and URLs
            3. Note the source domain for each
            4. Rate each source as High/Medium/Low reliability based on common knowledge
            
            REQUIRED OUTPUT FORMAT:
            ARTICLES FOUND:
            1. Title: [Title] | Source: [domain] | URL: [url] | Reliability: [High/Medium/Low]
            2. Title: [Title] | Source: [domain] | URL: [url] | Reliability: [High/Medium/Low]
            3. Title: [Title] | Source: [domain] | URL: [url] | Reliability: [High/Medium/Low]
            
            SUMMARY: [One sentence about what these articles cover]
            """,
            agent=agents[0],
            expected_output="List of 3-5 articles with titles, sources, URLs, and reliability ratings, plus a one-sentence summary."
        ),
        Task(
            description=f"""QUICK ANALYSIS: Analyze themes from article titles found for: {user_query}
                        
            SIMPLE INSTRUCTIONS:
            1. Review ONLY the article titles from the previous task
            2. Identify 5-7 key themes/topics
            3. List 8-10 important keywords
            4. Note any obvious conflicts in headlines
            5. Give a basic quality assessment
            
            REQUIRED OUTPUT FORMAT:
            THEMES: [theme1, theme2, theme3, theme4, theme5]
            KEYWORDS: [word1, word2, word3, word4, word5, word6, word7, word8]
            CONFLICTS: [Any obvious contradictions in headlines, or "None obvious"]
            QUALITY: [High/Medium/Low with brief reason]""",
            agent=agents[1],
            expected_output="Quick thematic analysis with themes, keywords, conflicts, and quality assessment from headlines only."
        ),
        Task(
            description=f"""QUICK SOCIAL SEARCH: Find hashtags and sentiment for: {user_query}
                        
            SIMPLE INSTRUCTIONS:
            1. Search for 3-5 relevant hashtags about this topic
            2. Assess general engagement as High/Medium/Low
            3. Determine overall sentiment as Positive/Negative/Neutral/Mixed
            4. Note if topic is trending or not
            
            REQUIRED OUTPUT FORMAT:
            HASHTAGS: #hashtag1, #hashtag2, #hashtag3
            ENGAGEMENT: [High/Medium/Low]
            SENTIMENT: [Positive/Negative/Neutral/Mixed]
            TRENDING: [Yes/No]
            """,
            agent=agents[2],
            expected_output="Basic social media metrics with hashtags, engagement level, sentiment, and trending status."
        ),
        Task(
            description=f"""ORGANIZE DATA: Structure all findings for: {user_query}
                        
            SIMPLE INSTRUCTIONS:
            1. Group articles by reliability (High/Medium/Low)
            2. Organize themes into main categories
            3. Create simple data structure
            4. Summarize patterns found
            
            REQUIRED OUTPUT FORMAT:
            HIGH RELIABILITY: [list of high-reliability sources]
            MEDIUM RELIABILITY: [list of medium-reliability sources]
            LOW RELIABILITY: [list of low-reliability sources]
            MAIN CATEGORIES: [grouped themes]
            PATTERNS: [brief summary of what data shows]""",
            agent=agents[3],
            expected_output="Organized data with reliability groupings, theme categories, and pattern summary."
        ),
        Task(
            description=f"""BASIC RELIABILITY CHECK: Assess information quality for: {user_query}
                        
            SIMPLE INSTRUCTIONS:
            1. Rate overall reliability 1-10 based on sources found
            2. Note any obvious red flags (if any)
            3. Suggest 2-3 basic verification steps
            4. Keep assessment simple and fast
            
            REQUIRED OUTPUT FORMAT:
            RELIABILITY SCORE: [1-10]/10
            RED FLAGS: [Any obvious issues, or "None obvious"]
            VERIFICATION STEPS: 
            - Step 1
            - Step 2
            - Step 3
            """,
            agent=agents[4],
            expected_output="Basic reliability assessment with score, red flags, and verification steps."
        ),
        Task(
            description=f"""COMPILE REPORT: Create JSON report for: {user_query}
                        
            INSTRUCTIONS:
            1. Take all information from previous tasks
            2. Fill the JSON template with actual data found
            3. Use "Unknown" or "N/A" for missing information
            4. Keep data realistic based on what was actually found
            5. Ensure valid JSON format
            
            Context:
            - Query: {user_query}
            - URLs: {urls or 'None provided'}
            - Keywords: {keywords or 'None provided'}
            - Hashtags: {hashtags or 'None provided'}
            
            OUTPUT MUST BE VALID JSON following this template:
            {json_schema_template}
            
            CRITICAL: Replace template values with ACTUAL findings from previous tasks.
            Use simple, realistic values. Do not make up complex analysis.
            Focus on speed and accuracy over comprehensiveness.""",
            agent=agents[5],
            expected_output="Complete JSON report following the NewsAnalysisReport schema with actual findings from the analysis."
        )
    ]

@cache_data(ttl=3600, show_spinner=False)
def analyze_sentiment(text):
    # Existing analysis logic
    # Add memory cleanup
    return {
        'sentiment': processed_sentiment,
        'confidence': confidence_score,
        'emotion': dominant_emotion,
        'subjectivity': subjectivity_score,
        'irony_detected': irony_flag,
        'sarcasm_level': sarcasm_level,
        'keywords': list(keywords_set),
        'entity_mentions': list(named_entities),
        'topic_distribution': topic_probs
    }