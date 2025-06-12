from crewai import Task

def create_news_analysis_tasks(agents, user_query, urls=None, hashtags=None, keywords=None):
    if not agents:
        return None
    
    # Ensure we have all required agents
    if len(agents) < 6:
        print(f"Expected 6 agents, got {len(agents)}")
        return None
    
    # Create the JSON schema template without f-string formatting issues
    json_schema_template = '''
    {
        "query_summary": "Brief summary of the query",
        "key_findings": "Main findings from the analysis",
        "related_articles": [
            {"title": "Article Title", "url": "https://example.com"}
        ],
        "related_words": ["keyword1", "keyword2", "keyword3"],
        "topic_clusters": [
            {
                "topic": "Main Topic",
                "size": 10,
                "related_narratives": ["narrative1", "narrative2"]
            }
        ],
        "top_sources": [
            {
                "domain": "example.com",
                "factual_rating": "High",
                "articles_count": 5,
                "engagement": 1000
            }
        ],
        "top_hashtags": [
            {
                "hashtag": "#example",
                "engagement_rate": 15.5,
                "reach": 50000,
                "sentiment": "Positive"
            }
        ],
        "similar_posts_time_series": [
            {
                "date": "2024-01-01",
                "count": 25
            }
        ],
        "fake_news_sites": [
            {
                "site": "fakename.com",
                "shares": 100
            }
        ],
        "content_analysis": {
            "language_percentage": 85.0,
            "coordination_percentage": 15.0,
            "source_percentage": 75.0,
            "bot_like_activity_percentage": 10.0
        },
        "propaganda_analysis": {
            "overall_reliability_score": 75.0,
            "propaganda_techniques": [
                {
                    "technique_name": "Appeal to emotion",
                    "frequency": 3,
                    "severity": 6.0,
                    "example": "Example text",
                    "explanation": "Explanation of technique"
                }
            ],
            "misinformation_indicators": [
                {
                    "indicator_type": "Factual error",
                    "confidence": 0.8,
                    "correction": "Corrected information",
                    "source_verification": ["source1.com", "source2.com"]
                }
            ],
            "coordination_patterns": [
                {
                    "pattern_type": "Synchronized publishing",
                    "strength": 0.7,
                    "entities_involved": ["site1.com", "site2.com"],
                    "timeline": "Within 2 hours"
                }
            ],
            "bot_activity_metrics": {
                "bot_likelihood_score": 0.3,
                "account_creation_patterns": "Normal distribution",
                "behavioral_indicators": ["Regular posting", "Human-like engagement"],
                "network_analysis": "No suspicious clustering"
            },
            "fake_news_sites": [
                {
                    "domain": "fake-news.com",
                    "shares": 500,
                    "engagement": 2000,
                    "known_false_stories": 3,
                    "verification_failures": ["Failed fact-check 1"],
                    "deceptive_practices": ["Misleading headlines"],
                    "network_connections": ["connected-site.com"]
                }
            ],
            "manipulation_timeline": [
                {
                    "timestamp": "2024-01-01T10:00:00",
                    "event": "Initial post",
                    "manipulation_type": "None detected"
                }
            ],
            "narrative_fingerprint": {
                "main_narrative": 0.8,
                "counter_narrative": 0.2
            },
            "cross_verification_results": {
                "verified_claims": 5,
                "disputed_claims": 2,
                "unverified_claims": 1
            },
            "recommended_verification_steps": [
                "Check multiple sources",
                "Verify with fact-checking sites",
                "Look for original sources"
            ]
        },
        "platform_facts": [
            "Fact 1 about the platform",
            "Fact 2 about the platform"
        ],
        "cross_source_facts": [
            "Cross-verified fact 1",
            "Cross-verified fact 2"
        ]
    }
    '''
        
    return [
        Task(
            description=f"""Search and collect news articles about: {user_query}
            
            Instructions:
            1. Use search tools to find 5-7 recent news articles about the topic
            2. Extract basic information: title, URL, source domain, publication date
            3. Assess source reliability (High/Medium/Low)
            4. Note any obvious bias or credibility issues
            5. Provide a simple summary of what you found
            
            Output format:
            - Article 1: [Title] from [Source] - [URL] - Reliability: [High/Medium/Low]
            - Article 2: [Title] from [Source] - [URL] - Reliability: [High/Medium/Low]
            - Summary: Brief overview of the articles found
            
            Keep it simple and factual.""",
            agent=agents[0],
            expected_output="List of 5-7 news articles with titles, sources, URLs, and basic reliability assessment, plus a brief summary."
        ),
        Task(
            description=f"""Analyze the content of the news articles found for: {user_query}
            
            Instructions:
            1. Review the articles found by the Web Crawler
            2. Extract the main themes and key points
            3. Identify 10-15 important keywords related to the topic
            4. Note any conflicting information between sources
            5. Assess the overall factual quality
            
            Output format:
            Key Findings: [Main points from the articles]
            Important Keywords: [List of relevant words]
            Conflicting Information: [Any contradictions found]
            Factual Assessment: [Overall quality assessment]
            
            Be clear and concise.""",
            agent=agents[1],
            expected_output="Content analysis with key findings, important keywords, conflicting information, and factual assessment."
        ),
        Task(
            description=f"""Research social media activity around: {user_query}
            
            Instructions:
            1. Search for mentions of this topic on social platforms
            2. Find 3-5 relevant hashtags being used
            3. Assess general engagement levels (High/Medium/Low)
            4. Note the overall sentiment (Positive/Negative/Neutral/Mixed)
            5. Look for any viral or trending patterns
            
            Output format:
            Top Hashtags: #hashtag1, #hashtag2, #hashtag3
            Engagement Level: [High/Medium/Low]
            Overall Sentiment: [Positive/Negative/Neutral/Mixed]
            Trending Patterns: [Any notable patterns]
            
            Keep observations factual and simple.""",
            agent=agents[2],
            expected_output="Social media analysis with top hashtags, engagement levels, sentiment, and trending patterns."
        ),
        Task(
            description=f"""Organize the collected data for easy understanding of: {user_query}
            
            Instructions:
            1. Group the findings into main topic categories
            2. Create a simple timeline if dates are available
            3. Organize sources by reliability level
            4. Prepare a summary of data patterns found
            
            Output format:
            Topic Categories: [Main themes grouped]
            Timeline: [Key dates and events if available]
            Source Reliability: High: [list], Medium: [list], Low: [list]
            Data Patterns: [Summary of what the data shows]
            
            Focus on clear organization.""",
            agent=agents[3],
            expected_output="Organized data with topic categories, timeline, source reliability groupings, and pattern summary."
        ),
        Task(
            description=f"""Examine the content for propaganda techniques and misinformation about: {user_query}
            
            Instructions:
            1. Look for obvious propaganda techniques (emotional appeals, loaded language, etc.)
            2. Check for clear factual errors or misleading claims
            3. Note any suspicious coordination between sources
            4. Assess overall credibility on a scale of 1-10
            5. Suggest basic verification steps
            
            Output format:
            Propaganda Techniques Found: [List any obvious techniques]
            Factual Issues: [Any clear errors or misleading claims]
            Source Coordination: [Any suspicious patterns]
            Credibility Score: [1-10 with brief explanation]
            Verification Steps: [How readers can verify the information]
            
            Be careful to only flag obvious issues.""",
            agent=agents[4],
            expected_output="Analysis of propaganda techniques, factual issues, coordination patterns, credibility score, and verification recommendations."
        ),
        Task(
            description=f"""Create a comprehensive report combining all analysis of: {user_query}
            
            Instructions:
            1. Compile all findings from the previous tasks
            2. Create a structured JSON report following the NewsAnalysisReport schema
            3. Include the most important findings at the top
            4. Add practical recommendations for readers
            5. Keep the language clear and accessible
            
            Create a complete report in JSON format, strictly following the NewsAnalysisReport Pydantic model schema.
            
            Context:
            - Query: {user_query}
            - URLs: {urls or 'None provided'}
            - Keywords: {keywords or 'None provided'}
            - Hashtags: {hashtags or 'None provided'}
            
            Output MUST be a valid JSON object following this schema:
            """ + json_schema_template + """
            
            Ensure all fields are populated with actual analysis data, not placeholder text.""",
            agent=agents[5],
            expected_output="A comprehensive news analysis report in JSON format, strictly adhering to the NewsAnalysisReport Pydantic model schema."
        )
    ]