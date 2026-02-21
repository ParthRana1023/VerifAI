import os
import signal
import _signal
import threading

from dotenv import load_dotenv
load_dotenv()

# Must be set BEFORE importing crewai so telemetry is disabled at import time
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Patch signal at both the Python and C level to silently ignore calls from
# non-main threads.  CrewAI telemetry registers SIGTERM/SIGINT handlers at
# import time, but Streamlit runs user code in a worker thread.
_original_signal_fn = signal.signal
_original_c_signal = _signal.signal

def _safe_signal(signalnum, handler):
    if threading.current_thread() is not threading.main_thread():
        return signal.getsignal(signalnum)
    return _original_signal_fn(signalnum, handler)

def _safe_c_signal(signalnum, handler):
    if threading.current_thread() is not threading.main_thread():
        return _signal.getsignal(signalnum)
    return _original_c_signal(signalnum, handler)

signal.signal = _safe_signal
_signal.signal = _safe_c_signal

from crewai import LLM
import litellm
import litellm.utils
import streamlit as st
import requests
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

# Force CrewAI to use text-based ReAct prompting instead of native function calling.
# Without this, LiteLLM detects Groq model supports function calling and sends tools
# via the native API, but the model generates XML-style calls that Groq rejects.
litellm.supports_function_calling = lambda model: False
litellm.utils.supports_function_calling = lambda model: False

def setup_crewai_config():
    """Configure CrewAI to use Groq with proper settings"""
    # Remove any existing OpenAI configuration so CrewAI doesn't try to use it
    for key in ["OPENAI_API_KEY", "OPENAI_MODEL_NAME", "OPENAI_API_BASE"]:
        os.environ.pop(key, None)
    
    # Set CrewAI to use Groq
    os.environ["CREWAI_LLM_PROVIDER"] = "groq"
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    
    # Disable function calling and telemetry globally
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_FUNCTION_CALLING"] = "true"

def check_llm_status():
    """Check if GROQ_API_KEY is set and valid."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return False, "GROQ_API_KEY is not set. Please set it in your environment or .env file."
    
    # Basic validation
    if len(groq_api_key) < 10:
        return False, "GROQ_API_KEY appears to be invalid. Please check your API key."
        
    return True, "Groq API key is set and appears valid."

# Keep old name as alias for backward compatibility
check_gemini_status = check_llm_status

def get_llm() -> BaseChatModel:
    """Initializes and returns the appropriate LLM based on configuration."""
    try:
        provider = os.getenv("CREWAI_LLM_PROVIDER", "groq")
        
        if provider == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not set. Get one at https://console.groq.com")
            
            # Use CrewAI's LLM class with LiteLLM format: groq/model-name
            # max_rpm limits requests per minute to stay within Groq rate limits
            return LLM(
                model="groq/llama-3.3-70b-versatile",
                api_key=groq_api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def setup_api_keys():
    """Validate API keys"""
    serper_key = os.getenv("SERPER_API_KEY")
    
    # Don't consider dummy key as valid
    if not serper_key or serper_key == "dummy-key-for-ollama":
        if 'st' in globals():
            st.error("SERPER_API_KEY is required. Please set it in your environment, .env file, or via the sidebar.")
        else:
            print("SERPER_API_KEY is required. Please set it in your environment or .env file.")
        return False
    
    # Basic validation - Serper keys are typically alphanumeric
    if len(serper_key) < 20 or not serper_key.replace('-', '').replace('_', '').isalnum():
        if 'st' in globals():
            st.error("SERPER_API_KEY appears to be invalid. Please check your API key.")
        else:
            print("SERPER_API_KEY appears to be invalid. Please check your API key.")
        return False
    
    return True