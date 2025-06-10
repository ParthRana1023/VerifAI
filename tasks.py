from crewai import Task
from models import NewsAnalysisReport

def create_news_analysis_tasks(agents, user_query, urls=None, hashtags=None, keywords=None):
    if not agents:
        return None
        
    return [
        Task(
            description=f"Crawl news websites for articles related to: {user_query}. Identify reliable and unreliable sources. Extract article URLs, publication dates, and engagement metrics.",
            agent=agents[0],
            expected_output="A comprehensive dataset of news articles with their sources, reliability metrics, and engagement statistics."
        ),
        Task(
            description=f"Analyze the content of collected news articles for: {user_query}. Extract key findings, related topics, narrative patterns, and assess the factual nature of the content.",
            agent=agents[1],
            expected_output="Content analysis including key findings, related words for wordcloud, topic clusters, and fact assessments from multiple sources."
        ),
        Task(
            description=f"Track how the news topic '{user_query}' is spreading on social media. Identify top hashtags, engagement rates, reach, sentiment, and track similar posts over time.",
            agent=agents[2],
            expected_output="Social media analysis report with top hashtags, engagement metrics, sentiment analysis, and temporal spread patterns."
        ),
        Task(
            description="Generate data visualization structures for topic clusters, wordclouds, time series of news spread, and source reliability comparisons.",
            agent=agents[3],
            expected_output="Data structures ready for visualization including topic clusters with size metrics, temporal data for time series, and comparative source reliability data."
        ),
        Task(
            description=f"""Conduct a comprehensive analysis of news content related to '{user_query}' 
            for propaganda, misinformation, and coordinated inauthentic behavior.
        
            1. IDENTIFY PROPAGANDA TECHNIQUES:
              - Detect specific propaganda techniques (name-calling, bandwagon, testimonial, etc.)
              - Rate severity and provide concrete examples from articles
              - Calculate frequency of each technique across sources
        
            2. ASSESS MISINFORMATION INDICATORS:
              - Fact-check key claims against verified information
              - Identify missing context that changes interpretation
              - Document factual errors with correction sources
              - Evaluate manipulated quotes, images, or statistics
        
            3. DETECT COORDINATION PATTERNS:
              - Identify synchronized publishing or messaging
              - Track identical phrasing across seemingly unrelated sources
              - Analyze cross-platform narrative amplification
              - Map connections between sources spreading similar misinformation
        
            4. MEASURE BOT-LIKE ACTIVITY:
              - Calculate bot likelihood scores for sharing patterns
              - Identify suspicious account behaviors and creation patterns
              - Analyze network spread characteristics typical of inauthentic amplification
        
            5. CATALOG FAKE NEWS SITES:
              - Identify highest-impact fake news domains by engagement metrics
              - Document history of verification failures
              - Detail deceptive practices employed
              - Map network connections to other disinformation sources
        
            6. DEVELOP VERIFICATION GUIDANCE:
              - Create step-by-step verification process for readers
              - Suggest credible alternative sources for verification
              - Provide red flags that indicate potential misinformation
        
            Use tools to scrape articles, analyze text patterns, and verify claims against reliable 
            sources. Quantify results where possible with specific metrics and confidence scores.
            """,
            agent=agents[4],
            expected_output="""Comprehensive propaganda and misinformation analysis with:
            1. Overall reliability score with confidence intervals
            2. Cataloged propaganda techniques with examples and frequency metrics
            3. Fact-check results with verification sources
            4. Coordination pattern analysis with network visualization data
            5. Bot activity metrics with detailed behavioral indicators
            6. Ranked list of fake news sites with engagement metrics and verification history
            7. Timeline showing evolution of misinformation spread
            8. Narrative fingerprint showing distinctive patterns across sources
            9. Reader guidance for information verification""",
        ),
        Task(
            description="Generate final comprehensive news analysis report integrating all findings.",
            agent=agents[5],
            expected_output="A structured news analysis report summarizing all findings with clear sections for key insights, source reliability, content analysis, and fact comparisons.",
            output_pydantic=NewsAnalysisReport
        )
    ]