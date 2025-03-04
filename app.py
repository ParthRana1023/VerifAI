import getpass
import os
import warnings
import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from crewai import LLM, Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from dotenv import load_dotenv

# Suppress Pydantic deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Modify the setup_api_keys function
def setup_api_keys(force_reset=False):
    # Check for existing keys
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Only prompt if both are missing
    if not (groq_key or openai_key) or force_reset:
        choice = input("Choose LLM provider (1 for Groq, 2 for OpenAI): ")
        if choice == "1":
            groq_key = getpass.getpass("Enter your Groq API key: ")
            os.environ["GROQ_API_KEY"] = groq_key
        else:
            openai_key = getpass.getpass("Enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = openai_key

    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key or force_reset:
        serper_key = getpass.getpass("Enter your Serper API key: ")
        os.environ["SERPER_API_KEY"] = serper_key

    return True

# Update the LLM creation function
def get_llm():
    if os.getenv("GROQ_API_KEY"):
        return LLM(
            temperature=0.3,
            model="groq/llama-3.2-3b-preview"
        )
    elif os.getenv("OPENAI_API_KEY"):
        return LLM(
            temperature=0.3,
            model="gpt-4-turbo"
        )
    else:
        raise ValueError("No valid API key found for LLM provider")

# def get_groq_llm():
#     return LLM(
#         temperature=0.3,
#         model="groq/llama-3.2-3b-preview"
#         # groq_api_key=os.getenv("GROQ_API_KEY")
#     )

class SourceReliability(BaseModel):
    domain: str
    factual_rating: str  # High, Low, Mixed, Mostly Factual
    articles_count: int
    engagement: int

class SocialMediaMetrics(BaseModel):
    hashtag: str
    engagement_rate: float  # Percentage
    reach: int
    sentiment: str  # Positive, Negative, Neutral

class ContentAnalysisMetrics(BaseModel):
    language_percentage: float
    coordination_percentage: float
    source_percentage: float
    bot_like_activity_percentage: float

class TimeSeriesData(BaseModel):
    date: str
    count: int

class PropagandaTechnique(BaseModel):
    technique_name: str  # e.g., "Appeal to fear", "False equivalence", "Strawman"
    frequency: int  # How many instances detected
    severity: float  # 0-10 scale of severity
    example: str  # A brief example from the article
    explanation: str  # Why this is considered propaganda

class MisinformationIndicator(BaseModel):
    indicator_type: str  # e.g., "Factual error", "Missing context", "Manipulated content"
    confidence: float  # 0-1 scale of confidence in detection
    correction: str  # The factual correction or missing context
    source_verification: List[str]  # Sources that verify/contradict

class CoordinationPattern(BaseModel):
    pattern_type: str  # e.g., "Identical phrasing", "Synchronized publishing", "Cross-platform amplification"
    strength: float  # 0-1 scale of coordination strength
    entities_involved: List[str]  # Websites, accounts, networks involved
    timeline: str  # Brief description of coordination timeline

class BotActivityMetrics(BaseModel):
    bot_likelihood_score: float  # 0-1 scale 
    account_creation_patterns: str  # Description of suspicious patterns
    behavioral_indicators: List[str]  # List of indicators suggesting bot activity
    network_analysis: str  # Brief description of network behavior

class FakeNewsSite(BaseModel):
    domain: str
    shares: int
    engagement: int
    known_false_stories: int
    verification_failures: List[str]  # List of fact-checking failures
    deceptive_practices: List[str]  # Deceptive practices employed
    network_connections: List[str]  # Connected entities in disinformation network

class EnhancedPropagandaAnalysis(BaseModel):
    overall_reliability_score: float  # 0-100 scale
    propaganda_techniques: List[PropagandaTechnique]
    misinformation_indicators: List[MisinformationIndicator]
    coordination_patterns: List[CoordinationPattern]
    bot_activity_metrics: BotActivityMetrics
    fake_news_sites: List[FakeNewsSite]
    manipulation_timeline: List[Dict[str, Any]]  # Timeline of information manipulation
    narrative_fingerprint: Dict[str, float]  # Distinct narrative patterns and their strength
    cross_verification_results: Dict[str, Any]  # Results of cross-verification with reliable sources
    recommended_verification_steps: List[str]  # Recommended steps for readers to verify content


class NewsAnalysisReport(BaseModel):
    query_summary: str
    key_findings: str
    related_articles: List[Dict[str, str]]  # {title: str, url: str}
    related_words: List[str]  # For wordcloud
    topic_clusters: List[Dict[str, Any]]  # {topic: str, size: int, related_narratives: List[str]}
    top_sources: List[SourceReliability]
    top_hashtags: List[SocialMediaMetrics]
    similar_posts_time_series: List[TimeSeriesData]
    fake_news_sites: List[Dict[str, Any]]  # {site: str, shares: int}
    content_analysis: ContentAnalysisMetrics
    propaganda_analysis: EnhancedPropagandaAnalysis
    platform_facts: List[str]
    cross_source_facts: List[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

def create_news_analysis_agents():
    llm = get_llm() 

    return [
        Agent(
            role="Web Crawler",
            goal="Extract news data for the query",
            backstory="An expert web crawler specialized in news sites, capable of identifying reliable sources and extracting relevant articles efficiently.",
            tools=[ScrapeWebsiteTool(), WebsiteSearchTool(), SerperDevTool()],
            tool_calls=True,
            llm=llm,  # Assign Groq LLM
            verbose=True
        ),
        Agent(
            role="News Content Analyst",
            goal="Analyze news content in depth",
            backstory="A seasoned journalist with expertise in fact-checking, source reliability assessment, and content analysis who can identify credible sources, biases, and trends in news articles.",
            tools=[ScrapeWebsiteTool(), WebsiteSearchTool(), SerperDevTool()],
            tool_calls=True,
            llm=llm,  # Assign Groq LLM
            verbose=True
        ),
        Agent(
            role="Social Media Tracking Specialist",
            goal="Track news spread on social media",
            backstory="A social media expert who specializes in tracking how news spreads across platforms, identifying trending hashtags, measuring engagement, and analyzing sentiment related to news topics.",
            tools=[WebsiteSearchTool(), SerperDevTool()],
            tool_calls=True,
            llm=llm,  # Assign Groq LLM
            verbose=True
        ),
        Agent(
            role="News Data Visualization Expert",
            goal="Create data visualizations from news analysis",
            backstory="A data visualization specialist who transforms news analysis data into meaningful visual representations including topic clusters, wordclouds, time series graphs, and reliability charts.",
            tools=[SerperDevTool()],
            tool_calls=True,
            llm=llm,  # Assign Groq LLM
            verbose=True
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
        tool_calls=True,
        llm=llm,  # Assign Groq LLM
        verbose=True,
        allow_delegation=True
        ),
        Agent(
            role="News Report Generator",
            goal="Compile findings into a comprehensive news analysis report",
            backstory="A professional report writer specialized in organizing complex news analysis data into structured, insightful, and actionable reports with clear visualizations and fact comparisons.",
            tool_calls=False,
            llm=llm,  # Assign Groq LLM
            verbose=True
        )
    ]

def create_news_analysis_tasks(agents, user_query, urls=None, hashtags=None, keywords=None):
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

def create_news_analysis_crew(user_query, urls=None, hashtags=None, keywords=None):
    llm = get_llm() 
    agents = create_news_analysis_agents()
    tasks = create_news_analysis_tasks(agents, user_query, urls, hashtags, keywords)
    return Crew(agents=agents, tasks=tasks, process="sequential", llm=llm,  # Assign Groq LLM
            verbose=True)

def run_news_analysis(user_query, urls=None, hashtags=None, keywords=None):
    crew = create_news_analysis_crew(user_query, urls, hashtags, keywords)
    result = crew.kickoff(inputs={'query': user_query, 'urls': urls, 'hashtags': hashtags, 'keywords': keywords})
    return result.pydantic

def save_report_to_file(report, filename="news_analysis_report.md"):
    with open(filename, "w") as f:
        # Key Findings & Summary
        f.write(f"# News Analysis Report: {report.query_summary}\n\n")
        f.write("## Key Findings & Summary\n\n")
        f.write(f"{report.key_findings}\n\n")
        
        # Related Articles
        f.write("## Related Articles\n\n")
        for article in report.related_articles:
            for title, url in article.items():
                f.write(f"- [{title}]({url})\n")
        f.write("\n")
        
        # Related Words (wordcloud)
        f.write("## Related Words\n\n")
        f.write("*Wordcloud visualization would show these terms with size relative to frequency:*\n\n")
        f.write(", ".join(report.related_words))
        f.write("\n\n")
        
        # Topic Clusters
        f.write("## Related Topic Clusters\n\n")
        f.write("*Visualization would show bubbles with sizes relative to prevalence:*\n\n")
        for cluster in report.topic_clusters:
            f.write(f"- **{cluster.get('topic', 'N/A')}** (Size: {cluster.get('size', 'N/A')})\n")
            # Check if 'related_narratives' key exists before accessing it
            related_narratives = cluster.get('related_narratives', [])  # Default to empty list if not found
            if related_narratives:
                f.write("  - Related narratives: " + ", ".join(related_narratives) + "\n")
        f.write("\n")
        
        # Top Sources
        f.write("## List of Top Sources\n\n")
        f.write("| Domain | Factual | Articles | Engagement |\n")
        f.write("|--------|---------|----------|------------|\n")
        for source in report.top_sources:
            f.write(f"| {source.domain} | {source.factual_rating} | {source.articles_count} | {source.engagement} |\n")
        f.write("\n")
        
        # Top Hashtags
        f.write("## Top Hashtags\n\n")
        f.write("| Hashtag | Engagement Rate (%) | Reach | Sentiment |\n")
        f.write("|---------|---------------------|-------|------------|\n")
        for hashtag in report.top_hashtags:
            f.write(f"| {hashtag.hashtag} | {hashtag.engagement_rate} | {hashtag.reach} | {hashtag.sentiment} |\n")
        f.write("\n")
        
        # Time Series Graph
        f.write("## Similar Posts Spread Over Time\n\n")
        f.write("*Time series visualization would show:*\n\n")
        for data_point in report.similar_posts_time_series:
            f.write(f"- {data_point.date}: {data_point.count} posts\n")
        f.write("\n")
        
        # Fake News Sites
        f.write("## Most Shared Fake News Sites\n\n")
        f.write("*Line chart visualization would show:*\n\n")
        for site in report.fake_news_sites:
            # Check if 'site' and 'shares' keys exist before accessing them
            site_name = site.get('site', 'N/A')  # Default to 'N/A' if 'site' key is missing
            shares = site.get('shares', 0)  # Default to 0 if 'shares' key is missing
            f.write(f"- {site_name}: {shares} shares\n")
        f.write("\n")
        
        # Content Analysis Metrics
        f.write("## Content Analysis Metrics\n\n")
        f.write("*Percentage bars visualization would show:*\n\n")
        f.write(f"- Language: {report.content_analysis.language_percentage}%\n")
        f.write(f"- Coordination: {report.content_analysis.coordination_percentage}%\n")
        f.write(f"- Source: {report.content_analysis.source_percentage}%\n")
        f.write(f"- Bot-like activity: {report.content_analysis.bot_like_activity_percentage}%\n")
        f.write("\n")
        
        # Propaganda News
        f.write("## Propaganda and Misinformation Analysis\n\n")
    
        f.write(f"### Overall Reliability Score: {report.propaganda_analysis.overall_reliability_score}/100\n\n")
    
        f.write("### Propaganda Techniques Detected\n\n")
        f.write("| Technique | Frequency | Severity (0-10) | Example |\n")
        f.write("|-----------|-----------|-----------------|--------|\n")
        for technique in report.propaganda_analysis.propaganda_techniques:
            f.write(f"| **{technique.technique_name}** | {technique.frequency} | {technique.severity} | {technique.example} |\n")
        f.write("\n*Explanation of techniques:*\n\n")
        for technique in report.propaganda_analysis.propaganda_techniques:
            f.write(f"- **{technique.technique_name}**: {technique.explanation}\n")
        f.write("\n")
    
        f.write("### Misinformation Indicators\n\n")
        f.write("| Type | Confidence | Correction | Verification Sources |\n")
        f.write("|------|------------|------------|----------------------|\n")
        for indicator in report.propaganda_analysis.misinformation_indicators:
            sources = ", ".join(indicator.source_verification)
            f.write(f"| {indicator.indicator_type} | {indicator.confidence*100:.1f}% | {indicator.correction} | {sources} |\n")
        f.write("\n")
    
        f.write("### Coordination Patterns\n\n")
        for pattern in report.propaganda_analysis.coordination_patterns:
            f.write(f"**{pattern.pattern_type}** (Strength: {pattern.strength*100:.1f}%)\n")
            f.write(f"- Entities involved: {', '.join(pattern.entities_involved)}\n")
            f.write(f"- Timeline: {pattern.timeline}\n\n")
    
        f.write("### Bot Activity Metrics\n\n")
        bot_metrics = report.propaganda_analysis.bot_activity_metrics
        f.write(f"**Bot Likelihood Score: {bot_metrics.bot_likelihood_score*100:.1f}%**\n\n")
        f.write(f"Account Creation Patterns: {bot_metrics.account_creation_patterns}\n\n")
        f.write("Behavioral Indicators:\n")
        for indicator in bot_metrics.behavioral_indicators:
            f.write(f"- {indicator}\n")
        f.write(f"\nNetwork Analysis: {bot_metrics.network_analysis}\n\n")
    
        f.write("### Most Shared Fake News Sites\n\n")
        f.write("| Domain | Shares | Engagement | Known False Stories | Verification Failures |\n")
        f.write("|--------|--------|------------|---------------------|----------------------|\n")
        for site in report.propaganda_analysis.fake_news_sites:
            failures = ", ".join(site.verification_failures[:2]) + (", ..." if len(site.verification_failures) > 2 else "")
            f.write(f"| {site.domain} | {site.shares} | {site.engagement} | {site.known_false_stories} | {failures} |\n")
        f.write("\n")
    
        f.write("### Deceptive Practices by Domain\n\n")
        for site in report.propaganda_analysis.fake_news_sites:
            f.write(f"**{site.domain}**:\n")
            for practice in site.deceptive_practices:
                f.write(f"- {practice}\n")
            f.write("\n")
    
        f.write("### Information Manipulation Timeline\n\n")
        f.write("*Timeline showing how information evolved and spread:*\n\n")
        for entry in report.propaganda_analysis.manipulation_timeline:
            f.write(f"- **{entry.get('date', 'N/A')}**: {entry.get('event', 'N/A')}\n")
        f.write("\n")
    
        f.write("### Narrative Fingerprint\n\n")
        f.write("*Distinctive narrative patterns and their strength:*\n\n")
        for narrative, strength in report.propaganda_analysis.narrative_fingerprint.items():
            f.write(f"- **{narrative}**: {strength*100:.1f}%\n")
        f.write("\n")
    
        f.write("### How to Verify This Information\n\n")
        for i, step in enumerate(report.propaganda_analysis.recommended_verification_steps, 1):
            f.write(f"{i}. {step}\n")
        f.write("\n")
        
        # Facts Comparison
        f.write("## Facts Gathered from Platform\n\n")
        for fact in report.platform_facts:
            f.write(f"- {fact}\n")
        f.write("\n")
        
        f.write("## Facts Gathered from Relevant Sources\n\n")
        for fact in report.cross_source_facts:
            f.write(f"- {fact}\n")
    
    print(f"Report saved to {filename}")

def main():
    if not setup_api_keys():
        print("Invalid API keys. Exiting.")
        return
    
    user_query = input("Enter news topic to analyze: ")
    urls = input("Enter news URLs (comma-separated, optional): ").split(',') if input("Include specific news URLs? (y/n): ").lower() == 'y' else None
    hashtags = input("Enter hashtags to track (comma-separated, optional): ").split(',') if input("Track specific hashtags? (y/n): ").lower() == 'y' else None
    keywords = input("Enter additional keywords (comma-separated, optional): ").split(',') if input("Include additional keywords? (y/n): ").lower() == 'y' else None
    
    report = run_news_analysis(user_query, urls, hashtags, keywords)
    save_report_to_file(report)

if __name__ == "__main__":
    main()