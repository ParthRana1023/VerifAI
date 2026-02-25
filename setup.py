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
from logger_config import get_logger

_logger = get_logger(__name__)

# Force CrewAI to use text-based ReAct prompting instead of native function calling.
litellm.supports_function_calling = lambda model: False
litellm.utils.supports_function_calling = lambda model: False

def setup_crewai_config(provider="gemini"):
    """Configure CrewAI to use Google Gemini with proper settings."""
    # Remove any existing OpenAI configuration so CrewAI doesn't try to use it
    for key in ["OPENAI_API_KEY", "OPENAI_MODEL_NAME", "OPENAI_API_BASE"]:
        os.environ.pop(key, None)
    
    # Set CrewAI to use Gemini
    os.environ["CREWAI_LLM_PROVIDER"] = "gemini"
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
    # Avoid "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" warning
    os.environ.pop("GOOGLE_API_KEY", None)
    
    # Disable function calling and telemetry globally
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_FUNCTION_CALLING"] = "true"

def check_llm_status(provider="gemini"):
    """Check if the Gemini API key is set and valid."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False, "GEMINI_API_KEY is not set. Please set it via the sidebar or environment."
    if len(api_key) < 10:
        return False, "GEMINI_API_KEY appears to be invalid. Please check your API key."
    return True, "Gemini API key is set and appears valid."

# Keep old name as alias for backward compatibility
check_gemini_status = check_llm_status

# Available Gemini models (display name → LiteLLM model string)
GEMINI_MODELS = {
    "Gemini 2.5 Flash": "gemini/gemini-2.5-flash",
    "Gemini 3 Flash": "gemini/gemini-3-flash-preview",
}
DEFAULT_GEMINI_MODEL = "Gemini 2.5 Flash"

def get_llm(provider=None, model_name=None) -> BaseChatModel:
    """Initializes and returns a Gemini LLM.

    Args:
        provider: Kept for backward compat; ignored (always Gemini).
        model_name: Display name of the model (e.g. "Gemini 2.5 Flash").
                    Falls back to DEFAULT_GEMINI_MODEL.
    """
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/app/apikey")

        model_name = model_name or DEFAULT_GEMINI_MODEL
        litellm_model = GEMINI_MODELS.get(model_name, GEMINI_MODELS[DEFAULT_GEMINI_MODEL])

        return LLM(
            model=litellm_model,
            api_key=gemini_api_key
        )
    except Exception as e:
        _logger.error("Error initializing LLM: %s", e)
        return None

def setup_api_keys():
    """Validate API keys"""
    serper_key = os.getenv("SERPER_API_KEY")
    
    # Don't consider dummy key as valid
    if not serper_key or serper_key == "dummy-key-for-ollama":
        if 'st' in globals():
            st.error("SERPER_API_KEY is required. Please set it in your environment, .env file, or via the sidebar.")
        else:
            _logger.error("SERPER_API_KEY is required. Please set it in your environment or .env file.")
        return False
    
    # Basic validation - Serper keys are typically alphanumeric
    if len(serper_key) < 20 or not serper_key.replace('-', '').replace('_', '').isalnum():
        if 'st' in globals():
            st.error("SERPER_API_KEY appears to be invalid. Please check your API key.")
        else:
            _logger.error("SERPER_API_KEY appears to be invalid. Please check your API key.")
        return False
    
    return True