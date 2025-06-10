import os
import praw
import re
import logging
from collections import Counter
from app import run_news_analysis
from save_report import save_report_to_file
from setup import setup_api_keys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        print(f"Error: {reddit_data['error']}")
        return

    keywords = extract_keywords(reddit_data)
    if not keywords:
        print("No keywords extracted from the post.")
        return

    # Convert keywords to comma-separated string
    keyword_list = [kw['text'] for kw in keywords]
    user_query = "News analysis for: " + ", ".join(keyword_list[:5])  # Top 5 keywords
    
    # Verify directory exists
    if not os.path.exists(os.path.dirname(os.path.abspath(__file__))):
        print("Invalid directory path")
        return

    # Set up API keys
    if not setup_api_keys():
        print("Invalid API keys. Exiting.")
        return

    # Run analysis and save
    try:
        print("\nStarting analysis...")
        report = run_news_analysis(
            user_query=user_query,
            keywords=keyword_list
        )
        if report:
            save_report_to_file(report)
            print("Analysis complete!")
        else:
            print("Failed to generate report")
            
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
    
# Example usage
if __name__ == "__main__":
    url = input("Enter a Reddit URL: ")
    
    reddit_data = scrape_reddit_data(url)
    keywords = extract_keywords(reddit_data)
    
    if "error" not in reddit_data:
        print(f"Title: {reddit_data['title']}\n")
        print(f"Content:\n{reddit_data['selftext']}\n")
        print("Keywords:")
        for kw in keywords:
            print(f"- {kw['text']}: {kw['frequency']}")
    else:
        print(f"Error: {reddit_data['error']}")

    main()