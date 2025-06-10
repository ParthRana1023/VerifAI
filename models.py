from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any

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