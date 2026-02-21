import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from setup import setup_crewai_config, setup_api_keys, check_llm_status
from app import run_news_analysis, get_report_as_markdown
from reddit import scrape_reddit_data, extract_keywords, is_reddit_url
import traceback

st.set_page_config(
    page_title="VerifAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_groq_connection():
    """Check if Groq API key is set"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return False, "GROQ_API_KEY not found. Please set it in your environment variables or .env file."
    return True, "Groq API key is set."


# ─── Helper: safely get a value from a dict or Pydantic model ───
def _get(obj, key, default=None):
    """Get attribute from dict or Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ─────────────────────────────────────────────────────────────────
#  DISPLAY REPORT — handles str, dict, and Pydantic model reports
# ─────────────────────────────────────────────────────────────────
def display_report(report):
    """Display the news analysis report using all NewsAnalysisReport models."""
    if not report:
        st.error("No report to display")
        return

    # Plain string — just render as-is
    if isinstance(report, str):
        st.markdown("## Analysis Report")
        st.markdown(report)
        return

    try:
        _display_structured_report(report)
    except Exception as e:
        st.error(f"Error displaying report: {e}")
        st.markdown("## Raw Report")
        st.text(str(report)[:3000])


def _display_structured_report(report):
    """Render every section of NewsAnalysisReport from a dict or Pydantic model."""

    # ── Query Summary ──
    query_summary = _get(report, "query_summary", "News Analysis")
    st.title(f"📊 {query_summary}")

    # ── Key Findings ──
    key_findings = _get(report, "key_findings")
    if key_findings:
        st.header("🔍 Key Findings")
        st.markdown(key_findings)

    # ── Related Articles (RelatedArticle model) ──
    related_articles = _get(report, "related_articles", [])
    if related_articles:
        st.header("📰 Related Articles")
        articles_data = []
        for article in related_articles:
            articles_data.append({
                "Title": _get(article, "title", "N/A"),
                "Source": _get(article, "source", "N/A"),
                "Published": _get(article, "published_date", "N/A"),
                "URL": _get(article, "url", "#"),
            })
        df = pd.DataFrame(articles_data)
        st.dataframe(df, column_config={"URL": st.column_config.LinkColumn("URL")}, hide_index=True)

    # ── Related Words ──
    related_words = _get(report, "related_words", [])
    if related_words:
        st.header("🏷️ Related Keywords")
        st.markdown(" · ".join([f"`{w}`" for w in related_words]))

    # ── Topic Clusters (TopicCluster model) ──
    topic_clusters = _get(report, "topic_clusters", [])
    if topic_clusters:
        st.header("🗂️ Topic Clusters")
        for cluster in topic_clusters:
            name = _get(cluster, "cluster_name", "Unknown Cluster")
            keywords = _get(cluster, "keywords", [])
            count = _get(cluster, "article_count", 0)
            st.markdown(f"**{name}** — {count} articles")
            if keywords:
                st.caption(", ".join(keywords))

    # ── Top Sources (SourceInfo model) ──
    top_sources = _get(report, "top_sources", [])
    if top_sources:
        st.header("📡 Top Sources")
        sources_data = []
        for source in top_sources:
            sources_data.append({
                "Source": _get(source, "name", "N/A"),
                "URL": _get(source, "url", "#"),
                "Reliability": _get(source, "reliability_score", 0),
            })
        df = pd.DataFrame(sources_data)
        st.dataframe(df, column_config={"URL": st.column_config.LinkColumn("URL")}, hide_index=True)
        _plot_sources(sources_data)

    # ── Top Hashtags ──
    top_hashtags = _get(report, "top_hashtags", [])
    if top_hashtags:
        st.header("#️⃣ Top Hashtags")
        st.markdown(" · ".join([f"`{h}`" for h in top_hashtags]))

    # ── Similar Posts Time Series (TimeSeriesData model) ──
    time_series = _get(report, "similar_posts_time_series", [])
    if time_series:
        st.header("📈 Similar Posts Over Time")
        ts_data = []
        for entry in time_series:
            ts_data.append({
                "Date": _get(entry, "date", ""),
                "Count": _get(entry, "count", 0),
            })
        df = pd.DataFrame(ts_data)
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(x="Date", y="Count", data=df, marker="o", ax=ax)
            ax.set_title("Similar Posts Over Time")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.dataframe(df, hide_index=True)

    # ── Fake News Sites ──
    fake_news_sites = _get(report, "fake_news_sites", [])
    if fake_news_sites:
        st.header("🚨 Flagged Fake News Sites")
        for site in fake_news_sites:
            st.markdown(f"- ⚠️ {site}")

    # ── Content Analysis (ContentAnalysis model) ──
    content_analysis = _get(report, "content_analysis")
    if content_analysis:
        st.header("📝 Content Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment = _get(content_analysis, "sentiment", "N/A")
            color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(sentiment, "⚪")
            st.metric("Sentiment", f"{color} {sentiment}")
        with col2:
            bias = _get(content_analysis, "bias", "N/A")
            st.metric("Bias", bias)
        with col3:
            readability = _get(content_analysis, "readability_score", 0)
            st.metric("Readability", f"{readability:.1f}")

        key_entities = _get(content_analysis, "key_entities", [])
        if key_entities:
            st.markdown("**Key Entities:** " + ", ".join([f"`{e}`" for e in key_entities]))

    # ── Enhanced Propaganda Analysis ──
    propaganda = _get(report, "enhanced_propaganda_analysis") or _get(report, "propaganda_analysis")
    if propaganda:
        st.header("🛡️ Propaganda, Bots & Misinformation Analysis")

        risk_score = _get(propaganda, "overall_reliability_score", _get(propaganda, "overall_risk_score", 0))
        # Reversing the logic: if it's reliability (from new model), higher is better.
        # If it's risk (from old fallback), lower is better. We'll standard it to a Reliability Score.
        st.subheader("Overall Reliability Score")
        if risk_score >= 70:
            st.success(f"✅ Reliability Score: **{risk_score}/100** — HIGH CREDIBILITY")
        elif risk_score >= 40:
            st.warning(f"⚡ Reliability Score: **{risk_score}/100** — MIXED CREDIBILITY")
        else:
            st.error(f"⚠️ Reliability Score: **{risk_score}/100** — LOW CREDIBILITY / HIGH RISK")

        techniques = _get(propaganda, "propaganda_techniques", _get(propaganda, "propaganda_techniques_detected", []))
        if techniques:
            st.subheader("Propaganda Techniques Detected")
            for tech in techniques:
                if isinstance(tech, str):
                    st.markdown(f"- 🔸 {tech}")
                else:
                    t_name = _get(tech, "technique_name", "Unknown")
                    t_freq = _get(tech, "frequency", 0)
                    t_sev = _get(tech, "severity", 0)
                    t_ex = _get(tech, "example", "")
                    t_exp = _get(tech, "explanation", "")
                    st.markdown(f"**🔸 {t_name}** (Severity: {t_sev}/10 | Freq: {t_freq})")
                    if t_ex: st.caption(f"Example: *{t_ex}*")
                    if t_exp: st.markdown(f"> {t_exp}")

        misinfo = _get(propaganda, "misinformation_indicators", _get(propaganda, "misinformation_indicators_detected", []))
        if misinfo:
            st.subheader("Misinformation Indicators")
            for ind in misinfo:
                if isinstance(ind, str):
                    st.markdown(f"- 🔻 {ind}")
                else:
                    i_type = _get(ind, "indicator_type", "Unknown")
                    i_conf = _get(ind, "confidence", 0)
                    i_corr = _get(ind, "correction", "")
                    st.markdown(f"**🔻 {i_type}** (Confidence: {i_conf})")
                    if i_corr: st.markdown(f"> *Correction:* {i_corr}")

        coordination = _get(propaganda, "coordination_patterns", [])
        if coordination:
            st.subheader("Network Coordination")
            for coord in coordination:
                c_type = _get(coord, "pattern_type", "")
                c_str = _get(coord, "strength", 0)
                entities = _get(coord, "entities_involved", [])
                st.markdown(f"**🕸️ {c_type}** (Strength: {c_str})")
                if entities: st.caption(f"Entities: {', '.join(entities)}")

        bot_metrics = _get(propaganda, "bot_activity_metrics", {})
        if bot_metrics:
            bot_score = _get(bot_metrics, "bot_likelihood_score", 0)
            if bot_score > 0.5:
                st.subheader("🤖 Bot Activity Detected")
                st.warning(f"Bot Likelihood Score: {bot_score}")
                indicators = _get(bot_metrics, "behavioral_indicators", [])
                for i in indicators: st.markdown(f"- {i}")

        fake_sites = _get(propaganda, "fake_news_sites", [])
        if fake_sites:
            st.subheader("🚨 Fake News Network Sites")
            for site in fake_sites:
                domain = _get(site, "domain", "Unknown")
                false_stories = _get(site, "known_false_stories", 0)
                st.markdown(f"- **{domain}** ({false_stories} known false stories)")

    # ── Platform Facts ──
    platform_facts = _get(report, "platform_facts", [])
    if platform_facts:
        st.header("🌐 Platform Facts")
        for fact in platform_facts:
            st.markdown(f"- {fact}")

    # ── Cross-Source Facts ──
    cross_source_facts = _get(report, "cross_source_facts", [])
    if cross_source_facts:
        st.header("✅ Cross-Source Verified Facts")
        for fact in cross_source_facts:
            st.markdown(f"- {fact}")

    # ── Analysis Note ──
    analysis_note = _get(report, "analysis_note", "")
    if analysis_note and analysis_note != "No specific notes.":
        st.info(f"📝 **Note:** {analysis_note}")


# ─── Chart Helpers ───

def _plot_sources(sources_data):
    """Bar chart of source reliability scores."""
    if not sources_data:
        return
    df = pd.DataFrame(sources_data)
    if df["Reliability"].sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Source", y="Reliability", data=df, ax=ax, palette="viridis")
    ax.set_title("Source Reliability Scores")
    ax.set_ylabel("Reliability (0-100)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─── Reddit + Manual Analysis (unchanged) ───

def analyze_reddit_post(url):
    """Analyze a Reddit post and return the news analysis report"""
    with st.spinner("Scraping Reddit post..."):
        reddit_data = scrape_reddit_data(url)

    if "error" in reddit_data:
        st.error(f"Error: {reddit_data['error']}")
        return None

    with st.spinner("Extracting keywords..."):
        keywords = extract_keywords(reddit_data)

    if not keywords:
        st.error("No keywords extracted from the post.")
        return None

    keyword_list = [kw['text'] for kw in keywords]

    st.subheader("Reddit Post Information")
    st.write(f"**Title:** {reddit_data['title']}")
    st.write(f"**Subreddit:** r/{reddit_data['subreddit']}")
    st.write(f"**Author:** u/{reddit_data['author']}")
    st.write(f"**Score:** {reddit_data['score']} (Upvote ratio: {reddit_data['upvote_ratio']})")
    st.write(f"**Comments:** {reddit_data['num_comments']}")

    if reddit_data['selftext']:
        with st.expander("Post Content"):
            st.write(reddit_data['selftext'])

    st.subheader("Top Keywords")
    keywords_data = []
    for kw in keywords[:20]:
        keywords_data.append({"Keyword": kw['text'], "Frequency": kw['frequency']})
    st.dataframe(pd.DataFrame(keywords_data), hide_index=True)

    user_query = f"News analysis for: {reddit_data['title']}"
    keywords = keyword_list[:5]
    st.info(f"Running analysis for: {user_query}")

    with st.spinner("Running news analysis... This may take several minutes."):
        report = run_news_analysis(user_query=user_query, keywords=keywords)

    return report


def manual_analysis():
    """Manual news analysis without Reddit integration"""
    st.subheader("Manual News Analysis")

    col1, col2 = st.columns(2)

    with col1:
        user_query = st.text_input(
            "News Topic to Analyze:",
            placeholder="Enter the news topic you want to analyze...",
            key="user_query_1"
        )
        keywords_input = st.text_area(
            "Additional Keywords (one per line):",
            placeholder="keyword1\nkeyword2\nkeyword3",
            key="keywords_input_1"
        )

    with col2:
        urls_input = st.text_area(
            "Specific URLs to analyze (one per line):",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            key="urls_input_1"
        )
        hashtags_input = st.text_input(
            "Hashtags to track (comma-separated):",
            placeholder="#news, #breaking, #analysis",
            key="hashtags_input_1"
        )

    if st.button("Run Manual Analysis", type="primary", key="run_manual_analysis_1"):
        if not user_query.strip():
            st.error("Please enter a news topic to analyze.")
            return None

        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()] if keywords_input.strip() else None
        urls = [u.strip() for u in urls_input.split('\n') if u.strip()] if urls_input.strip() else None
        hashtags = [h.strip() for h in hashtags_input.split(',') if h.strip()] if hashtags_input.strip() else None

        with st.spinner("Running news analysis... This may take several minutes."):
            report = run_news_analysis(
                user_query=user_query.strip(),
                urls=urls,
                hashtags=hashtags,
                keywords=keywords
            )

        return report

    return None


def main():
    try:
        st.title("VerifAI: News Analysis Tool")

        # Sidebar
        st.sidebar.header("About")
        st.sidebar.markdown(
            """
            This tool analyzes news content to extract insights, detect propaganda techniques,
            and identify misinformation patterns. You can analyze Reddit posts to generate
            comprehensive reports.
            """
        )
        st.sidebar.expander("Configure API Keys").markdown(
            """
            To use this tool, you need to configure your Groq API key.
            You can get one at [Groq Console](https://console.groq.com).
            """
        )
        # Check Groq status
        groq_status, groq_msg = check_groq_connection()
        if groq_status:
            st.success(f"✅ {groq_msg}")
        else:
            st.error(f"❌ {groq_msg}")
            st.info("Please set your GROQ_API_KEY in your environment variables or .env file.")

        # API key setup in the sidebar
        with st.sidebar.expander("Configure API Keys"):
            current_key = os.environ.get("SERPER_API_KEY", "")
            serper_api_key = st.text_input(
                "Serper API Key",
                value=current_key if current_key != "dummy-key-for-ollama" else "",
                type="password",
                help="Required for web search functionality",
                key="serper_api_key_1"
            )

            if st.button("Save API Keys", key="save_api_keys_1"):
                if serper_api_key.strip():
                    os.environ["SERPER_API_KEY"] = serper_api_key.strip()
                    st.success("API key saved!")
                else:
                    st.error("Please enter a valid Serper API key")

        # Reddit Analysis interface
        st.header("Reddit Post Analysis")
        st.markdown("Analyze a Reddit post to understand news patterns and credibility.")

        url = st.text_input(
            "Enter a Reddit URL:",
            placeholder="https://www.reddit.com/r/news/comments/...",
            help="Paste a link to a Reddit post you want to analyze",
            key="url_1"
        )

        if st.button("Analyze Reddit Post", type="primary", key="analyze_reddit_post_1"):
            if not url:
                st.error("Please enter a Reddit URL.")
            elif not is_reddit_url(url):
                st.error("Invalid Reddit URL. Please enter a valid Reddit URL.")
            elif not setup_api_keys():
                st.error("API keys not set or invalid. Please set valid API keys in the sidebar.")
            else:
                try:
                    report = analyze_reddit_post(url)

                    if report:
                        st.success("Analysis completed!")
                        st.divider()
                        display_report(report)

                        st.divider()
                        markdown_report = get_report_as_markdown(report)
                        st.download_button(
                            label="📥 Download Report as Markdown",
                            data=markdown_report,
                            file_name=f"reddit_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error("Failed to generate report.")

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error("If this error persists, check your API keys and Ollama setup.")
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

if __name__ == "__main__":
    main()
