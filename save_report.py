"""
save_report.py — Convert a NewsAnalysisReport (dict or Pydantic model) to Markdown.
"""

from logger_config import get_logger

_logger = get_logger(__name__)


def _get(obj, key, default=None):
    """Get attribute from dict or Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def save_report_to_file(report, filename="news_analysis_report.md"):
    """Write the full analysis report to a markdown file."""
    try:
        md = generate_report_markdown(report)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(md)
        _logger.info("Report saved to %s", filename)
        return True
    except Exception as e:
        _logger.error("Failed to save report: %s", e)
        return False


def generate_report_markdown(report):
    """Return the full report as a markdown string."""
    lines = []

    # ── Title & Key Findings ──
    query_summary = _get(report, "query_summary", "News Analysis")
    lines.append(f"# News Analysis Report: {query_summary}\n")
    lines.append(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    key_findings = _get(report, "key_findings")
    if key_findings:
        lines.append("## Key Findings & Summary\n")
        lines.append(f"{key_findings}\n")

    # ── Related Articles ──
    related_articles = _get(report, "related_articles", [])
    if related_articles:
        lines.append("## Related Articles\n")
        lines.append("| Title | Source | Published | URL |")
        lines.append("|-------|--------|-----------|-----|")
        for article in related_articles:
            title = _get(article, "title", "N/A")
            source = _get(article, "source", "N/A")
            published = _get(article, "published_date", "N/A")
            url = _get(article, "url", "#")
            lines.append(f"| {title} | {source} | {published} | [Link]({url}) |")
        lines.append("")

    # ── Related Keywords ──
    related_words = _get(report, "related_words", [])
    if related_words:
        lines.append("## Related Keywords\n")
        lines.append("*Word cloud visualization — size proportional to frequency:*\n")
        lines.append(", ".join(related_words))
        lines.append("")

    # ── Topic Clusters ──
    topic_clusters = _get(report, "topic_clusters", [])
    if topic_clusters:
        lines.append("## Topic Clusters\n")
        lines.append("*Bubble chart — size proportional to article count:*\n")
        lines.append("| Cluster | Keywords | Articles |")
        lines.append("|---------|----------|----------|")
        for cluster in topic_clusters:
            name = _get(cluster, "cluster_name", "Unknown")
            keywords = _get(cluster, "keywords", [])
            count = _get(cluster, "article_count", 0)
            kw_str = ", ".join(keywords) if keywords else "—"
            lines.append(f"| **{name}** | {kw_str} | {count} |")
        lines.append("")

    # ── Top Sources ──
    top_sources = _get(report, "top_sources", [])
    if top_sources:
        lines.append("## Top Sources\n")
        lines.append("| Domain | Factual Rating | Articles | Engagement |")
        lines.append("|--------|----------------|----------|------------|")
        for source in top_sources:
            domain = _get(source, "domain", "N/A")
            factual = _get(source, "factual_rating", "N/A")
            articles = _get(source, "articles_count", 0)
            engagement = _get(source, "engagement", 0)
            lines.append(f"| {domain} | {factual} | {articles} | {engagement} |")
        lines.append("")

    # ── Top Hashtags ──
    top_hashtags = _get(report, "top_hashtags", [])
    if top_hashtags:
        lines.append("## Top Hashtags\n")
        lines.append("| Hashtag | Engagement Rate (%) | Reach | Sentiment |")
        lines.append("|---------|---------------------|-------|-----------|")
        for ht in top_hashtags:
            hashtag = _get(ht, "hashtag", "N/A")
            eng_rate = _get(ht, "engagement_rate", 0.0)
            reach = _get(ht, "reach", 0)
            sentiment = _get(ht, "sentiment", "Neutral")
            lines.append(f"| {hashtag} | {eng_rate:.1f} | {reach} | {sentiment} |")
        lines.append("")

    # ── Similar Posts Time Series ──
    time_series = _get(report, "similar_posts_time_series", [])
    if time_series:
        lines.append("## Similar Posts Over Time\n")
        lines.append("*Time series line chart:*\n")
        lines.append("| Date | Post Count |")
        lines.append("|------|------------|")
        for entry in time_series:
            date = _get(entry, "date", "N/A")
            count = _get(entry, "count", 0)
            lines.append(f"| {date} | {count} |")
        lines.append("")

    # ── Fake News Sites (simple list) ──
    fake_news_sites = _get(report, "fake_news_sites", [])
    if fake_news_sites:
        lines.append("## Flagged Fake News Sites\n")
        for site in fake_news_sites:
            lines.append(f"- ⚠️ {site}")
        lines.append("")

    # ── Content Analysis Metrics ──
    content_analysis = _get(report, "content_analysis")
    if content_analysis:
        lines.append("## Content Analysis Metrics\n")
        lines.append("*Percentage bar visualization:*\n")
        lang = _get(content_analysis, "language_percentage", 0.0)
        coord = _get(content_analysis, "coordination_percentage", 0.0)
        src = _get(content_analysis, "source_percentage", 0.0)
        bot = _get(content_analysis, "bot_like_activity_percentage", 0.0)
        lines.append(f"- Language: {lang}%")
        lines.append(f"- Coordination: {coord}%")
        lines.append(f"- Source: {src}%")
        lines.append(f"- Bot-like activity: {bot}%")
        lines.append("")

    # ── Propaganda & Misinformation Analysis ──
    propaganda = _get(report, "propaganda_analysis")
    if propaganda:
        lines.append("## Propaganda and Misinformation Analysis\n")

        reliability = _get(propaganda, "overall_reliability_score", 0)
        lines.append(f"### Overall Reliability Score: {reliability}/100\n")

        # Propaganda Techniques
        techniques = _get(propaganda, "propaganda_techniques", [])
        if techniques:
            lines.append("### Propaganda Techniques Detected\n")
            lines.append("| Technique | Frequency | Severity (0-10) | Example |")
            lines.append("|-----------|-----------|-----------------|---------|")
            for tech in techniques:
                name = _get(tech, "technique_name", "Unknown")
                freq = _get(tech, "frequency", 0)
                sev = _get(tech, "severity", 0)
                ex = _get(tech, "example", "")
                lines.append(f"| **{name}** | {freq} | {sev} | {ex} |")
            lines.append("")
            lines.append("*Explanation of techniques:*\n")
            for tech in techniques:
                name = _get(tech, "technique_name", "Unknown")
                explanation = _get(tech, "explanation", "")
                if explanation:
                    lines.append(f"- **{name}**: {explanation}")
            lines.append("")

        # Misinformation Indicators
        indicators = _get(propaganda, "misinformation_indicators", [])
        if indicators:
            lines.append("### Misinformation Indicators\n")
            lines.append("| Type | Confidence | Correction | Verification Sources |")
            lines.append("|------|------------|------------|----------------------|")
            for ind in indicators:
                i_type = _get(ind, "indicator_type", "Unknown")
                conf = _get(ind, "confidence", 0)
                corr = _get(ind, "correction", "")
                sources = _get(ind, "source_verification", [])
                src_str = ", ".join(sources) if sources else "—"
                lines.append(f"| {i_type} | {conf*100:.1f}% | {corr} | {src_str} |")
            lines.append("")

        # Coordination Patterns
        coordination = _get(propaganda, "coordination_patterns", [])
        if coordination:
            lines.append("### Coordination Patterns\n")
            for pattern in coordination:
                p_type = _get(pattern, "pattern_type", "Unknown")
                strength = _get(pattern, "strength", 0)
                entities = _get(pattern, "entities_involved", [])
                timeline = _get(pattern, "timeline", "")
                lines.append(f"**{p_type}** (Strength: {strength*100:.1f}%)")
                if entities:
                    lines.append(f"- Entities involved: {', '.join(entities)}")
                if timeline:
                    lines.append(f"- Timeline: {timeline}")
                lines.append("")

        # Bot Activity Metrics
        bot_metrics = _get(propaganda, "bot_activity_metrics", {})
        if bot_metrics:
            bot_score = _get(bot_metrics, "bot_likelihood_score", 0)
            lines.append("### Bot Activity Metrics\n")
            lines.append(f"**Bot Likelihood Score: {bot_score*100:.1f}%**\n")
            patterns = _get(bot_metrics, "account_creation_patterns", "")
            if patterns:
                lines.append(f"Account Creation Patterns: {patterns}\n")
            indicators_list = _get(bot_metrics, "behavioral_indicators", [])
            if indicators_list:
                lines.append("Behavioral Indicators:")
                for ind in indicators_list:
                    lines.append(f"- {ind}")
            network = _get(bot_metrics, "network_analysis", "")
            if network:
                lines.append(f"\nNetwork Analysis: {network}")
            lines.append("")

        # Fake News Network Sites
        fake_sites = _get(propaganda, "fake_news_sites", [])
        if fake_sites:
            lines.append("### Fake News Network Sites\n")
            lines.append("| Domain | Shares | Engagement | Known False Stories | Verification Failures |")
            lines.append("|--------|--------|------------|---------------------|----------------------|")
            for site in fake_sites:
                domain = _get(site, "domain", "Unknown")
                shares = _get(site, "shares", 0)
                engagement = _get(site, "engagement", 0)
                false_stories = _get(site, "known_false_stories", 0)
                failures = _get(site, "verification_failures", [])
                fail_str = ", ".join(failures[:2]) + (", ..." if len(failures) > 2 else "") if failures else "—"
                lines.append(f"| {domain} | {shares} | {engagement} | {false_stories} | {fail_str} |")
            lines.append("")

            lines.append("### Deceptive Practices by Domain\n")
            for site in fake_sites:
                domain = _get(site, "domain", "Unknown")
                practices = _get(site, "deceptive_practices", [])
                if practices:
                    lines.append(f"**{domain}**:")
                    for p in practices:
                        lines.append(f"- {p}")
                    lines.append("")

        # Manipulation Timeline
        timeline = _get(propaganda, "manipulation_timeline", [])
        if timeline:
            lines.append("### Information Manipulation Timeline\n")
            for entry in timeline:
                date = _get(entry, "date", "N/A") if isinstance(entry, dict) else "N/A"
                event = _get(entry, "event", "N/A") if isinstance(entry, dict) else str(entry)
                lines.append(f"- **{date}**: {event}")
            lines.append("")

        # Narrative Fingerprint
        fingerprint = _get(propaganda, "narrative_fingerprint", {})
        if fingerprint:
            lines.append("### Narrative Fingerprint\n")
            for narrative, strength in fingerprint.items():
                lines.append(f"- **{narrative}**: {strength*100:.1f}%")
            lines.append("")

        # Verification Steps
        steps = _get(propaganda, "recommended_verification_steps", [])
        if steps:
            lines.append("### How to Verify This Information\n")
            for i, step in enumerate(steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

    # ── Platform Facts ──
    platform_facts = _get(report, "platform_facts", [])
    if platform_facts:
        lines.append("## Facts Gathered from Platform\n")
        for fact in platform_facts:
            lines.append(f"- {fact}")
        lines.append("")

    # ── Cross-Source Facts ──
    cross_source_facts = _get(report, "cross_source_facts", [])
    if cross_source_facts:
        lines.append("## Facts Gathered from Relevant Sources\n")
        for fact in cross_source_facts:
            lines.append(f"- {fact}")
        lines.append("")

    # ── Analysis Note ──
    note = _get(report, "analysis_note", "")
    if note and note != "No specific notes.":
        lines.append(f"---\n\n*Note: {note}*\n")

    lines.append("---\n")
    lines.append("*This report was generated using automated AI analysis tools.*")
    lines.append("*Results should be verified with additional sources for critical decisions.*")

    return "\n".join(lines)