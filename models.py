from pydantic import BaseModel, Field, ConfigDict, model_serializer
from typing import List, Dict, Any
from datetime import datetime
import json

class SourceReliability(BaseModel):
    domain: str = ""
    factual_rating: str = "Mixed"  # High, Low, Mixed, Mostly Factual
    articles_count: int = 0
    engagement: int = 0

class SocialMediaMetrics(BaseModel):
    hashtag: str = ""
    engagement_rate: float = 0.0  # Percentage
    reach: int = 0
    sentiment: str = "Neutral"  # Positive, Negative, Neutral

class ContentAnalysisMetrics(BaseModel):
    language_percentage: float = 0.0
    coordination_percentage: float = 0.0
    source_percentage: float = 0.0
    bot_like_activity_percentage: float = 0.0

class TimeSeriesData(BaseModel):
    date: str = ""
    count: int = 0

class PropagandaTechnique(BaseModel):
    technique_name: str = ""  # e.g., "Appeal to fear", "False equivalence", "Strawman"
    frequency: int = 0  # How many instances detected
    severity: float = 0.0  # 0-10 scale of severity
    example: str = ""  # A brief example from the article
    explanation: str = ""  # Why this is considered propaganda

class MisinformationIndicator(BaseModel):
    indicator_type: str = ""  # e.g., "Factual error", "Missing context", "Manipulated content"
    confidence: float = 0.0  # 0-1 scale of confidence in detection
    correction: str = ""  # The factual correction or missing context
    source_verification: List[str] = []  # Sources that verify/contradict

class CoordinationPattern(BaseModel):
    pattern_type: str = ""  # e.g., "Identical phrasing", "Synchronized publishing", "Cross-platform amplification"
    strength: float = 0.0  # 0-1 scale of coordination strength
    entities_involved: List[str] = []  # Websites, accounts, networks involved
    timeline: str = ""  # Brief description of coordination timeline

class BotActivityMetrics(BaseModel):
    bot_likelihood_score: float = 0.0  # 0-1 scale 
    account_creation_patterns: str = ""  # Description of suspicious patterns
    behavioral_indicators: List[str] = []  # List of indicators suggesting bot activity
    network_analysis: str = ""  # Brief description of network behavior

class FakeNewsSite(BaseModel):
    domain: str = ""
    shares: int = 0
    engagement: int = 0
    known_false_stories: int = 0
    verification_failures: List[str] = []  # List of fact-checking failures
    deceptive_practices: List[str] = []  # Deceptive practices employed
    network_connections: List[str] = []  # Connected entities in disinformation network

class EnhancedPropagandaAnalysis(BaseModel):
    overall_reliability_score: float = 0.0  # 0-100 scale
    propaganda_techniques: List[PropagandaTechnique] = []
    misinformation_indicators: List[MisinformationIndicator] = []
    coordination_patterns: List[CoordinationPattern] = []
    bot_activity_metrics: BotActivityMetrics = BotActivityMetrics()
    fake_news_sites: List[FakeNewsSite] = []
    manipulation_timeline: List[Dict[str, Any]] = []  # Timeline of information manipulation
    narrative_fingerprint: Dict[str, float] = {}  # Distinct narrative patterns and their strength
    cross_verification_results: Dict[str, Any] = {}  # Results of cross-verification with reliable sources
    recommended_verification_steps: List[str] = []  # Recommended steps for readers to verify content

class RelatedArticle(BaseModel):
    title: str = Field(..., description="Title of the related article.")
    url: str = Field(..., description="URL of the related article.")
    source: str = Field(..., description="Source of the related article.")
    published_date: str = Field(..., description="Publication date of the related article.")

class TopicCluster(BaseModel):
    cluster_name: str = Field(..., description="Name or theme of the topic cluster.")
    keywords: List[str] = Field(default_factory=list, description="Keywords associated with this cluster.")
    article_count: int = Field(..., description="Number of articles in this cluster.")

class SourceInfo(BaseModel):
    name: str = Field(..., description="Name of the news source.")
    url: str = Field(..., description="URL of the news source.")
    reliability_score: float = Field(..., description="Reliability score of the source (e.g., 0-100).")

class ContentAnalysis(BaseModel):
    sentiment: str = Field(..., description="Overall sentiment of the content (Positive, Negative, Neutral).")
    bias: str = Field(..., description="Identified bias in the content (e.g., Left-leaning, Right-leaning, Neutral).")
    readability_score: float = Field(..., description="Readability score of the content.")
    key_entities: List[str] = Field(default_factory=list, description="Key entities mentioned in the content.")

class PropagandaAnalysis(BaseModel):
    propaganda_techniques_detected: List[str] = Field(default_factory=list, description="List of propaganda techniques detected.")
    misinformation_indicators_detected: List[str] = Field(default_factory=list, description="List of misinformation indicators detected.")
    overall_risk_score: float = Field(..., description="Overall risk score for propaganda/misinformation (0-100).")

class NewsAnalysisReport(BaseModel):
    query_summary: str = Field(..., description="A concise summary of the news analysis query.")
    key_findings: str = Field(..., description="The most important insights and conclusions from the analysis.")
    related_articles: List[RelatedArticle] = Field(default_factory=list, description="A list of related articles found during the analysis.")
    related_words: List[str] = Field(default_factory=list, description="Key words and phrases extracted from the news content.")
    topic_clusters: List[TopicCluster] = Field(default_factory=list, description="Identified clusters of related topics within the news.")
    top_sources: List[SourceInfo] = Field(default_factory=list, description="Information about the most relevant news sources.")
    top_hashtags: List[str] = Field(default_factory=list, description="Prominent hashtags associated with the news.")
    similar_posts_time_series: List[TimeSeriesData] = Field(default_factory=list, description="Time-series data showing the trend of similar news posts.")
    fake_news_sites: List[str] = Field(default_factory=list, description="List of identified fake news or unreliable sources.")
    content_analysis: ContentAnalysis = Field(..., description="Detailed analysis of the news content characteristics.")
    propaganda_analysis: PropagandaAnalysis = Field(..., description="Analysis of propaganda techniques and misinformation indicators.")
    platform_facts: List[str] = Field(default_factory=list, description="Facts and observations related to the platform where the news was found.")
    cross_source_facts: List[str] = Field(default_factory=list, description="Facts cross-verified across multiple sources.")
    analysis_note: str = Field(default="No specific notes.", description="Any additional notes or disclaimers about the analysis.")

    def to_json(self):
        return self.model_dump_json(indent=2)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )