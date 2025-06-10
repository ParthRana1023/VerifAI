import streamlit as st

def save_report_to_file(report, filename="news_analysis_report.md"):
    try:
        with open(filename, "w", encoding='utf-8') as f:
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
                related_narratives = cluster.get('related_narratives', [])
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
                site_name = site.get('site', 'N/A')
                shares = site.get('shares', 0)
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
            
            # Propaganda Analysis
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
        return True
    except Exception as e:
        st.error(f"Failed to save report: {e}")
        return False