import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

# Properly configure CrewAI to use Ollama
def setup_crewai_config():
    """Configure CrewAI to use Ollama instead of OpenAI"""
    # Critical: Remove OpenAI key completely from environment
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    # Ensure OpenAI key is not set to any value
    os.environ.pop("OPENAI_API_KEY", None)
    
    # Set CrewAI to use Ollama explicitly
    os.environ["CREWAI_LLM_PROVIDER"] = "ollama"
    
    # Set Ollama configuration
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    
    # Important: Set a dummy OpenAI key to prevent CrewAI from complaining
    # This is a workaround for CrewAI's OpenAI dependency
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-ollama"

def setup_api_keys():
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        st.error("SERPER_API_KEY is required. Please set it in your environment or .env file.")
        return False
    
    return bool(serper_key)

def get_llm():
    """Get properly configured LLM instance for Ollama"""
    try:
        # Method 1: Try the most explicit Ollama configuration
        return LLM(
            model="ollama/mistral:latest",
            base_url="http://localhost:11434"
        )
    except Exception as e:
        st.warning(f"Primary LLM config failed: {e}")
        try:
            # Method 2: Alternative configuration
            return LLM(
                model="ollama/mistral:latest",
                base_url="http://localhost:11434",
                api_key="not-needed-for-ollama"
            )
        except Exception as e2:
            st.error(f"Secondary LLM config failed: {e2}")
            try:
                # Method 3: Minimal configuration
                return LLM(model="ollama/mistral:latest")
            except Exception as e3:
                st.error(f"All LLM configurations failed: {e3}")
                return None
