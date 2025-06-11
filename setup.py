import os
from dotenv import load_dotenv
from crewai import LLM
import streamlit as st
import requests
import time

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
    
    # Disable function calling and telemetry globally
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_FUNCTION_CALLING"] = "true"

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

def get_llm():
    """Get properly configured LLM instance for Ollama without function calling"""
    try:
        # Check if Ollama is available first
        ollama_ok, ollama_msg = check_ollama_status()
        if not ollama_ok:
            if 'st' in globals():
                st.error(f"Cannot configure LLM: {ollama_msg}")
            else:
                print(f"Cannot configure LLM: {ollama_msg}")
            return None
        
        # Simple configuration without advanced features
        llm = LLM(
            model="ollama/mistral:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            # Add timeout to prevent hanging
            timeout=120
        )
        
        # Test the LLM with a simple completion
        test_ok, test_msg = test_ollama_completion()
        if not test_ok:
            if 'st' in globals():
                st.warning(f"LLM test failed: {test_msg}")
            else:
                print(f"LLM test failed: {test_msg}")
        
        return llm
        
    except Exception as e:
        if 'st' in globals():
            st.error(f"Failed to configure LLM: {e}")
        else:
            print(f"Failed to configure LLM: {e}")
        
        # Even simpler fallback
        try:
            return LLM(model="ollama/mistral:latest")
        except Exception as e2:
            if 'st' in globals():
                st.error(f"Fallback LLM configuration also failed: {e2}")
            else:
                print(f"Fallback LLM configuration also failed: {e2}")
            return None

def check_ollama_status():
    """Check if Ollama is running and has the required model"""
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
    except requests.exceptions.Timeout:
        return False, "Ollama connection timed out. Please check if Ollama is responding."
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"

def test_ollama_completion():
    """Test a simple completion with Ollama to verify it's working"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:latest",
                "prompt": "Hello, respond with just 'OK' if you can see this.",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and data['response'].strip():
                return True, f"Ollama completion test successful: {data['response'][:50]}..."
            else:
                return False, "Ollama returned empty response"
        else:
            return False, f"Ollama completion test failed with status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "Ollama completion test timed out"
    except Exception as e:
        return False, f"Ollama completion test failed: {str(e)}"

def wait_for_ollama(max_wait_time=60):
    """Wait for Ollama to become available"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        ollama_ok, ollama_msg = check_ollama_status()
        if ollama_ok:
            return True, ollama_msg
        
        if 'st' in globals():
            st.info(f"Waiting for Ollama... ({ollama_msg})")
        else:
            print(f"Waiting for Ollama... ({ollama_msg})")
        
        time.sleep(5)
    
    return False, "Timed out waiting for Ollama to become available"

def get_available_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model.get('name', '') for model in models]
        else:
            return []
    except Exception:
        return []

def pull_model(model_name):
    """Pull a model if it's not available"""
    try:
        if 'st' in globals():
            st.info(f"Pulling model {model_name}... This may take a while.")
        else:
            print(f"Pulling model {model_name}... This may take a while.")
        
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            timeout=600  # 10 minutes timeout for pulling
        )
        
        if response.status_code == 200:
            return True, f"Successfully pulled {model_name}"
        else:
            return False, f"Failed to pull {model_name}: status {response.status_code}"
            
    except Exception as e:
        return False, f"Error pulling model {model_name}: {str(e)}"

def ensure_model_available(model_name="mistral:latest"):
    """Ensure the required model is available"""
    available_models = get_available_models()
    
    if not any(model_name in model for model in available_models):
        if 'st' in globals():
            st.warning(f"Model {model_name} not found. Attempting to pull...")
        else:
            print(f"Model {model_name} not found. Attempting to pull...")
        
        success, msg = pull_model(model_name)
        if success:
            if 'st' in globals():
                st.success(msg)
            else:
                print(msg)
            return True
        else:
            if 'st' in globals():
                st.error(msg)
            else:
                print(msg)
            return False
    
    return True