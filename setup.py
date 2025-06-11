import os
from dotenv import load_dotenv
from crewai import LLM
import streamlit as st

load_dotenv()

def setup_crewai_config():
    """Configure CrewAI to use Ollama with proper settings"""
    # Remove any existing OpenAI configuration
    for key in ["OPENAI_API_KEY", "OPENAI_MODEL_NAME", "OPENAI_API_BASE"]:
        os.environ.pop(key, None)
    
    # Set CrewAI to use Ollama
    os.environ["CREWAI_LLM_PROVIDER"] = "ollama"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    
    # Set a dummy OpenAI key to prevent CrewAI from complaining
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-ollama"

def setup_api_keys():
    """Validate API keys"""
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        if 'st' in globals():
            st.error("SERPER_API_KEY is required. Please set it in your environment or .env file.")
        else:
            print("SERPER_API_KEY is required. Please set it in your environment or .env file.")
        return False
    
    return bool(serper_key)

def get_llm():
    """Get properly configured LLM instance for Ollama with JSON mode disabled"""
    try:
        # Disable JSON mode and function calling to avoid compatibility issues
        return LLM(
            model="ollama/mistral:latest",
            base_url="http://localhost:11434",
            # Disable JSON mode which causes issues with Ollama
            json_mode=False,
            # Disable function calling
            supports_functions=False,
            # Add temperature for better output
            temperature=0.1,
            # Disable system finger printing
            supports_system_message=True
        )
    except Exception as e:
        if 'st' in globals():
            st.error(f"Failed to configure LLM: {e}")
        else:
            print(f"Failed to configure LLM: {e}")
        
        # Fallback configuration
        try:
            return LLM(
                model="ollama/mistral:latest",
                base_url="http://localhost:11434",
                temperature=0.1
            )
        except Exception as e2:
            if 'st' in globals():
                st.error(f"Fallback LLM configuration also failed: {e2}")
            else:
                print(f"Fallback LLM configuration also failed: {e2}")
            return None

def check_ollama_status():
    """Check if Ollama is running and has the required model"""
    import requests
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code != 200:
            return False, f"Ollama API returned status {response.status_code}"
        
        # Check if mistral model is available
        models = response.json().get('models', [])
        model_names = [model.get('name', '') for model in models]
        
        mistral_available = any('mistral' in name.lower() for name in model_names)
        if not mistral_available:
            return False, "Mistral model not found. Please run: ollama pull mistral"
        
        return True, "Ollama is running with Mistral model"
        
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Please start Ollama service."
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"