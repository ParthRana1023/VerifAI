from crewai import Task
from models import NewsAnalysisReport

def create_news_analysis_tasks(agents, user_query, urls=None, hashtags=None, keywords=None):
    if not agents:
        return None
    
    # Ensure we have all required agents
    if len(agents) < 6:
        print(f"Expected 6 agents, got {len(agents)}")
        return None
        
    return [
        Task(
            description=f"""Crawl news websites for articles related to: {user_query}. 
            
            Your task:
            1. Search for news articles using the provided query
            2. Identify reliable and unreliable sources
            3. Extract article URLs, publication dates, and basic engagement metrics
            4. Focus on finding 5-10 relevant articles maximum
            5. Provide a simple summary of findings
            
            Keep the analysis focused and avoid complex JSON structures.""",
            agent=agents[0],
            expected_output="A simple list of news articles with their sources, basic reliability assessment, and key metadata. Format as plain text with clear sections.",
            # Remove complex output requirements
            context=f"Query: {user_query}, URLs: {urls or []}, Keywords: {keywords or []}"
        ),
        Task(
            description=f"""Analyze the content of collected news articles for: {user_query}. 
            
            Your task:
            1. Read through the articles found by the Web Crawler
            2. Extract key findings and main themes
            3. Identify related topics and words
            4. Assess the factual nature of the content
            5. Keep analysis simple and focused
            
            Provide clear, structured output without complex formatting.""",
            agent=agents[1],
            expected_output="Content analysis with key findings, related words, main topics, and basic fact assessment. Use simple text format with clear headings.",
            context=f"Analyzing articles for query: {user_query}"
        ),
        Task(
            description=f"""Track how the news topic '{user_query}' is spreading on social media.
            
            Your task:
            1. Search for social media mentions of the topic
            2. Identify relevant hashtags (top 3-5)
            3. Assess general engagement and sentiment
            4. Track basic temporal patterns
            5. Keep analysis simple and factual
            
            Focus on observable patterns rather than complex metrics.""",
            agent=agents[2],
            expected_output="Social media analysis with top hashtags, basic engagement info, general sentiment, and simple timeline. Use plain text format.",
            context=f"Social media tracking for: {user_query}"
        ),
        Task(
            description=f"""Create simple data structures for visualization of the news analysis.
            
            Your task:
            1. Organize findings into clear categories
            2. Prepare data for topic clusters (simple groupings)
            3. Create basic time series information
            4. Structure source reliability data
            5. Keep everything simple and clear
            
            Focus on organizing existing data rather than creating complex visualizations.""",
            agent=agents[3],
            expected_output="Organized data structures for visualization including topic groups, timeline data, and source information. Use simple, clear formatting.",
            context=f"Preparing visualization data for: {user_query}"
        ),
        Task(
            description=f"""Conduct basic analysis of news content for propaganda and misinformation patterns.
            
            Your task:
            1. Look for obvious propaganda techniques in the content
            2. Identify clear misinformation indicators
            3. Check for coordination patterns if apparent
            4. Assess basic credibility signals
            5. Provide simple reliability scoring
            
            Focus on clear, observable patterns rather than deep analysis. Keep findings simple and well-supported.""",
            agent=agents[4],
            expected_output="Basic propaganda and misinformation analysis with clear examples, simple reliability scores, and verification suggestions. Use plain text with clear sections.",
            context=f"Analyzing content reliability for: {user_query}"
        ),
        Task(
            description=f"""Generate a comprehensive but simple news analysis report.
            
            Your task:
            1. Compile all findings from previous tasks
            2. Create a structured report with clear sections
            3. Include key findings, source analysis, and fact comparisons
            4. Keep the report readable and well-organized
            5. Ensure all sections are populated with relevant information
            
            Create a complete report that summarizes all the analysis work.""",
            agent=agents[5],
            expected_output="A complete news analysis report with all required sections filled out clearly and comprehensively.",
            output_pydantic=NewsAnalysisReport,
            context=f"Generating final report for: {user_query}"
        )
    ]