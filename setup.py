import os
from dotenv import load_dotenv
from crewai import LLM
import streamlit as st
import requests
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

def setup_crewai_config():
    """Configure CrewAI to use Gemini with proper settings"""
    # Remove any existing OpenAI configuration
    for key in ["OPENAI_API_KEY", "OPENAI_MODEL_NAME", "OPENAI_API_BASE"]:
        os.environ.pop(key, None)
    os.environ["OPENAI_API_KEY"] = ""
    
    # Set CrewAI to use Gemini
    os.environ["CREWAI_LLM_PROVIDER"] = "gemini"
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    
    # Disable function calling and telemetry globally
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_FUNCTION_CALLING"] = "true"

def check_gemini_status():
    """Check if GEMINI_API_KEY is set and valid."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return False, "GEMINI_API_KEY is not set. Please set it in your environment or .env file."
    
    # Basic validation: Gemini API keys are typically long alphanumeric strings
    # This is a very basic check and doesn't validate against the API itself
    if len(gemini_api_key) < 20:
        return False, "GEMINI_API_KEY appears to be invalid. Please check your API key."
        
    return True, "Gemini API key is set and appears valid."

def get_llm() -> BaseChatModel:
    """Initializes and returns the appropriate LLM based on configuration."""
    try:
        # For Gemini - use LiteLLM compatible format
        if os.getenv("CREWAI_LLM_PROVIDER") == "gemini":
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set for Gemini LLM provider.")
            
            # Use CrewAI's LLM class with proper provider prefix for LiteLLM
            return LLM(
                model="gemini/gemini-2.5-flash-lite",  # LiteLLM format: provider/model
                api_key=gemini_api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {os.getenv('CREWAI_LLM_PROVIDER')}")
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