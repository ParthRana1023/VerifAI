import os
import praw
import re
from collections import Counter
from logger_config import get_logger
from app import run_news_analysis
from save_report import save_report_to_file
from setup import setup_api_keys

logger = get_logger(__name__)


def compute_reddit_engagement(reddit_data: dict) -> dict:
    """Derive real engagement metrics from PRAW-scraped Reddit data.

    Returns a dict suitable for passing as ``platform_metrics`` to
    ``app.run_news_analysis()``.
    """
    score = reddit_data.get("score", 0)
    num_comments = reddit_data.get("num_comments", 0)
    upvote_ratio = reddit_data.get("upvote_ratio", 0.5)

    # Estimate total votes from score and ratio
    # score = upvotes - downvotes, ratio = upvotes / total_votes
    # So total_votes ≈ score / (2 * ratio - 1) when ratio > 0.5
    if upvote_ratio > 0.5:
        total_votes = int(score / (2 * upvote_ratio - 1))
    else:
        total_votes = max(score, 1)

    # Reddit rule of thumb: ~10x lurkers per voter
    reach_estimate = total_votes * 10

    # Engagement rate = (interactions / reach) * 100
    total_interactions = score + num_comments
    engagement_rate = (total_interactions / max(reach_estimate, 1)) * 100

    # Derive sentiment from upvote ratio
    if upvote_ratio > 0.7:
        sentiment = "Positive"
    elif upvote_ratio > 0.4:
        sentiment = "Mixed"
    else:
        sentiment = "Negative"

    # Derive comment engagement metrics
    comment_scores = []
    for comment in reddit_data.get("top_comments", []):
        comment_scores.append(comment.get("score", 0))
    avg_comment_score = sum(comment_scores) / max(len(comment_scores), 1)

    return {
        "engagement_rate": round(engagement_rate, 2),
        "reach": reach_estimate,
        "total_interactions": total_interactions,
        "total_votes": total_votes,
        "avg_comment_score": round(avg_comment_score, 1),
        "sentiment_from_ratio": sentiment,
    }

def scrape_reddit_data(url: str) -> dict:
    """
    Scrape data from a Reddit URL using PRAW.
    
    Args:
        url (str): The Reddit URL to scrape
        
    Returns:
        dict: A dictionary containing the scraped data (title, content, author, etc.)
    """
    try:
        # Initialize Reddit via PRAW
        reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
            redirect_uri=os.environ["REDDIT_REDIRECT_URI"],
        )
        reddit.read_only = True
        
        # Validate the URL
        if not is_reddit_url(url):
            logger.error("Invalid Reddit URL: %s", url)
            return {"error": "Invalid Reddit URL"}
        
        # Fetch the submission
        submission = reddit.submission(url=url)
        
        # Collect data
        data = {
            "title": submission.title,
            "selftext": submission.selftext,
            "author": str(submission.author),
            "subreddit": submission.subreddit.display_name,
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "created_utc": submission.created_utc,
            "url": submission.url,
            "permalink": submission.permalink,
            "num_comments": submission.num_comments,
            "is_original_content": submission.is_original_content,
        }
        
        # Collect top-level comments
        submission.comments.replace_more(limit=0)  # Only get the comments that are initially loaded
        comments = []
        for comment in submission.comments[:10]:  # Get the top 10 comments
            comments.append({
                "author": str(comment.author),
                "body": comment.body,
                "score": comment.score,
                "created_utc": comment.created_utc,
            })
        data["top_comments"] = comments
        
        logger.info("Successfully scraped data for Reddit post: %s", submission.title)
        return data
        
    except Exception as e:
        logger.error("Error scraping Reddit data: %s", e)
        return {"error": str(e)}

def extract_keywords(data: dict, top_n: int = 25) -> list:
    """
    Extract keywords from the scraped Reddit data.
    
    Args:
        data (dict): The scraped Reddit data
        top_n (int): Number of top keywords to return
        
    Returns:
        list: A list of dictionaries containing the keywords and their frequencies
    """
    try:
        # Check if there's an error in the data
        if "error" in data:
            logger.error("Cannot extract keywords, error in data: %s", data["error"])
            return []
        
        # Combine title, selftext, and comment bodies
        combined_text = data["title"] + " " + data["selftext"]
        
        # Add comment text if available
        if "top_comments" in data:
            for comment in data["top_comments"]:
                combined_text += " " + comment["body"]
        
        # Extract words
        words = re.findall(r'\b\w+\b', combined_text.lower())
        
        # Define stopwords
        stopwords = set([
            "the", "and", "for", "you", "that", "this", "with", "have", "are",
            "but", "not", "was", "from", "they", "will", "all", "your", "can",
            "has", "had", "been", "their", "more", "which", "when", "what",
            "about", "would", "there", "one", "just", "like", "some", "out",
            "also", "how", "its", "i", "a", "an", "in", "on", "of", "to", "is", 
            "https", "www", "com", "reddit", "edit", "post", "comment", "thread"
        ])
        
        # Filter out stopwords and short words
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Count frequencies
        counter = Counter(filtered_words)
        most_common = counter.most_common(top_n)
        
        # Format result
        keywords = [{"text": word, "frequency": freq} for word, freq in most_common]
        
        logger.info("Successfully extracted %d keywords", len(keywords))
        return keywords
        
    except Exception as e:
        logger.error("Error extracting keywords: %s", e)
        return []

def is_reddit_url(text: str) -> bool:
    """Check if a URL is a Reddit URL."""
    return "reddit.com" in text.lower()

def main():
    if "error" in reddit_data:
        logger.error("Error: %s", reddit_data['error'])
        return

    keywords = extract_keywords(reddit_data)
    if not keywords:
        logger.error("No keywords extracted from the post.")
        return

    keyword_list = [kw['text'] for kw in keywords]
    user_query = "News analysis for: " + ", ".join(keyword_list[:5])

    if not os.path.exists(os.path.dirname(os.path.abspath(__file__))):
        logger.error("Invalid directory path")
        return

    if not setup_api_keys():
        logger.error("Invalid API keys. Exiting.")
        return

    try:
        logger.info("Starting analysis...")
        report = run_news_analysis(
            user_query=user_query,
            keywords=keyword_list
        )
        if report:
            save_report_to_file(report)
            logger.info("Analysis complete!")
        else:
            logger.error("Failed to generate report")

    except Exception as e:
        logger.error("Analysis failed: %s", e)


if __name__ == "__main__":
    from logger_config import setup_logging
    setup_logging("INFO")

    url = input("Enter a Reddit URL: ")

    reddit_data = scrape_reddit_data(url)
    keywords = extract_keywords(reddit_data)

    if "error" not in reddit_data:
        logger.info("Title: %s", reddit_data['title'])
        logger.info("Content: %s", reddit_data['selftext'])
        logger.info("Keywords:")
        for kw in keywords:
            logger.info("- %s: %d", kw['text'], kw['frequency'])
    else:
        logger.error("Error: %s", reddit_data['error'])

    main()