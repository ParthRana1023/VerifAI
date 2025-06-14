from crewai import Task

def create_news_analysis_tasks(agents, user_query, urls=None, hashtags=None, keywords=None):
    if not agents:
        return None
    
    # Ensure we have all required agents
    if len(agents) < 6:
        print(f"Expected 6 agents, got {len(agents)}")
        return None
    
    # Simplified JSON schema template for faster processing
    json_schema_template = '''
    {
        "query_summary": "Brief summary of the query analysis",
        "key_findings": "Main findings from the analysis",
        "related_articles": [
            {"title": "Article Title", "url": "https://example.com"}
        ],
        "related_words": ["keyword1", "keyword2", "keyword3"],
        "topic_clusters": [
            {"topic": "Main Topic", "size": 5, "related_narratives": ["narrative1"]}
        ],
        "top_sources": [
            {"domain": "example.com", "factual_rating": "High", "articles_count": 3, "engagement": 500}
        ],
        "top_hashtags": [
            {"hashtag": "#example", "engagement_rate": 10.0, "reach": 1000, "sentiment": "Neutral"}
        ],
        "similar_posts_time_series": [
            {"date": "2024-01-01", "count": 5}
        ],
        "fake_news_sites": [
            {"site": "unknown", "shares": 0}
        ],
        "content_analysis": {
            "language_percentage": 90.0,
            "coordination_percentage": 5.0,
            "source_percentage": 80.0,
            "bot_like_activity_percentage": 10.0
        },
        "propaganda_analysis": {
            "overall_reliability_score": 70.0,
            "propaganda_techniques": [
                {"technique_name": "None detected", "frequency": 0, "severity": 0.0, "example": "N/A", "explanation": "No obvious propaganda techniques found"}
            ],
            "misinformation_indicators": [
                {"indicator_type": "None detected", "confidence": 0.9, "correction": "N/A", "source_verification": ["N/A"]}
            ],
            "coordination_patterns": [
                {"pattern_type": "None detected", "strength": 0.1, "entities_involved": ["N/A"], "timeline": "N/A"}
            ],
            "bot_activity_metrics": {
                "bot_likelihood_score": 0.1,
                "account_creation_patterns": "Normal",
                "behavioral_indicators": ["Human-like behavior"],
                "network_analysis": "No suspicious activity"
            },
            "fake_news_sites": [
                {"domain": "none", "shares": 0, "engagement": 0, "known_false_stories": 0, "verification_failures": [], "deceptive_practices": [], "network_connections": []}
            ],
            "manipulation_timeline": [
                {"timestamp": "2024-01-01T00:00:00", "event": "No manipulation detected", "manipulation_type": "None"}
            ],
            "narrative_fingerprint": {"main_narrative": 1.0},
            "cross_verification_results": {"verified_claims": 0, "disputed_claims": 0, "unverified_claims": 0},
            "recommended_verification_steps": ["Check multiple reliable news sources", "Look for fact-checking websites"]
        },
        "platform_facts": ["Basic factual information available"],
        "cross_source_facts": ["Cross-referenced with available sources"]
    }
    '''
        
    return [
        Task(
            description=f"""QUICK SEARCH: Find 3-5 recent news articles about: {user_query}
            
            TIME LIMIT: Complete this in under 2 minutes.
            
            SIMPLE INSTRUCTIONS:
            1. Use ONLY search tools (do NOT scrape full websites)
            2. Find exactly 3-5 article titles and URLs
            3. Note the source domain for each
            4. Rate each source as High/Medium/Low reliability based on common knowledge
            5. Provide a 1-sentence summary
            
            REQUIRED OUTPUT FORMAT:
            ARTICLES FOUND:
            1. Title: [Title] | Source: [domain] | URL: [url] | Reliability: [High/Medium/Low]
            2. Title: [Title] | Source: [domain] | URL: [url] | Reliability: [High/Medium/Low]
            3. Title: [Title] | Source: [domain] | URL: [url] | Reliability: [High/Medium/Low]
            
            SUMMARY: [One sentence about what these articles cover]
            
            DO NOT: Scrape content, analyze deeply, or spend more than 2 minutes.""",
            agent=agents[0],
            expected_output="List of 3-5 articles with titles, sources, URLs, and reliability ratings, plus a one-sentence summary."
        ),
        Task(
            description=f"""QUICK ANALYSIS: Analyze themes from article titles found for: {user_query}
            
            TIME LIMIT: Complete this in under 2 minutes.
            
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
            QUALITY: [High/Medium/Low with brief reason]
            
            DO NOT: Scrape full articles, conduct deep research, or exceed time limit.""",
            agent=agents[1],
            expected_output="Quick thematic analysis with themes, keywords, conflicts, and quality assessment from headlines only."
        ),
        Task(
            description=f"""QUICK SOCIAL SEARCH: Find hashtags and sentiment for: {user_query}
            
            TIME LIMIT: Complete this in under 90 seconds.
            
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
            
            DO NOT: Deep dive into social media analysis or exceed time limit.""",
            agent=agents[2],
            expected_output="Basic social media metrics with hashtags, engagement level, sentiment, and trending status."
        ),
        Task(
            description=f"""ORGANIZE DATA: Structure all findings for: {user_query}
            
            TIME LIMIT: Complete this in under 1 minute.
            
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
            PATTERNS: [brief summary of what data shows]
            
            DO NOT: Conduct new research or spend more than 1 minute.""",
            agent=agents[3],
            expected_output="Organized data with reliability groupings, theme categories, and pattern summary."
        ),
        Task(
            description=f"""BASIC RELIABILITY CHECK: Assess information quality for: {user_query}
            
            TIME LIMIT: Complete this in under 1 minute.
            
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
            
            DO NOT: Conduct deep verification research or exceed time limit.""",
            agent=agents[4],
            expected_output="Basic reliability assessment with score, red flags, and verification steps."
        ),
        Task(
            description=f"""COMPILE REPORT: Create JSON report for: {user_query}
            
            TIME LIMIT: Complete this in under 90 seconds.
            
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
            """ + json_schema_template + """
            
            CRITICAL: Replace template values with ACTUAL findings from previous tasks.
            Use simple, realistic values. Do not make up complex analysis.
            Focus on speed and accuracy over comprehensiveness.""",
            agent=agents[5],
            expected_output="Complete JSON report following the NewsAnalysisReport schema with actual findings from the analysis."
        )
    ]