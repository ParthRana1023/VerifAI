# VerifAI

VerifAI is an AI-powered platform that analyzes news, social media, and online content for accuracy, bias, and credibility. It leverages advanced LLMs (Groq or OpenAI), real-time web scraping, and data visualization to help users detect misinformation, propaganda, and coordinated inauthentic behavior.

## Features

- **Reddit News Analysis**: Analyze Reddit posts and extract news-related insights, keywords, and trends.
- **Misinformation & Propaganda Detection**: Identify propaganda techniques, misinformation indicators, and fake news sites.
- **Social Media Tracking**: Track news spread, top hashtags, engagement, and sentiment across social platforms.
- **Data Visualization**: Visualize topic clusters, word clouds, time series, and reliability metrics.
- **Comprehensive Reporting**: Generate detailed, structured reports in Markdown or view interactively via Streamlit.

## Project Structure

- `app.py` — Core logic for news analysis, agent/task orchestration, and report generation.
- `streamlit.py` — Streamlit web interface for interactive analysis and visualization.
- `reddit.py` — Reddit scraping, keyword extraction, and Reddit-specific utilities.
- `requirements.txt` — Python dependencies.
- `db/` — Local database and cache files (auto-generated).

## Setup

### 1. Clone the Repository
```sh
# Use PowerShell or your preferred terminal
cd <your-folder>
git clone <repo-url>
cd VerifAI
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Configure API Keys
You will need:
- **Groq** or **OpenAI** API key (for LLM)
- **Serper** API key (for web search)
- **Reddit** API credentials (client_id, client_secret, user_agent, redirect_uri)

You can set these as environment variables or enter them at runtime when prompted.

#### Example `.env` file:
```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent
REDDIT_REDIRECT_URI=your_redirect_uri
```

## Usage

### Command-Line Interface
Run the main analysis tool:
```sh
python app.py
```
You will be prompted for:
- News topic to analyze
- (Optional) News URLs, hashtags, or keywords
- API keys (if not set in environment)

A Markdown report will be generated and saved as `news_analysis_report.md`.

### Streamlit Web App
Launch the interactive web interface:
```sh
streamlit run streamlit.py
```
- Enter a Reddit post URL to analyze news-related content.
- View results, download reports, and explore visualizations.
- Configure API keys in the sidebar.

## Output
- **Markdown Report**: Detailed news analysis, key findings, source reliability, propaganda detection, and more.
- **Interactive Visualizations**: Topic clusters, word clouds, time series, and reliability charts (Streamlit UI).

## Advanced
- The system uses modular agents for crawling, content analysis, social tracking, visualization, and misinformation detection.
- Easily extendable for new data sources or analysis tasks.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## License
This project is for research and educational purposes.

---

*For questions or contributions, please open an issue or pull request.*
