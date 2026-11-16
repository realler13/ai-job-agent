import time
import logging
import random
import threading
from typing import Optional, Callable
import google.generativeai as genai
from .config import GEMINI_RPM_LIMIT, GEMINI_MAX_RETRIES, GEMINI_INITIAL_BACKOFF

# Global lock for thread safety if multiple threads call this
_lock = threading.Lock()

class RateLimiter:
    """
    A Token Bucket / Leaky Bucket style rate limiter.
    Ensures that we don't exceed `requests_per_minute`.
    """
    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        
    def wait_for_token(self):
        """
        Blocks until enough time has passed to allow a new request.
        """
        with _lock:
            current_time = time.time()
            # If last_request_time is 0, we can start immediately
            if self.last_request_time == 0.0:
                 self.last_request_time = current_time
                 return

            elapsed = current_time - self.last_request_time
            
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                # logger.debug(f"RateLimiter: Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
            self.last_request_time = time.time()

# Global instance
# If config has RPM=15, we aim for safe 10 or 12 to account for network overhead/bursts.
_limiter = RateLimiter(requests_per_minute=GEMINI_RPM_LIMIT)

_gemini_configured = False

def ensure_gemini_configured():
    global _gemini_configured
    if _gemini_configured:
        return True
    
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.warning("⚠️ GOOGLE_API_KEY not found. Gemini models will fail.")
        return False
    
    try:
        genai.configure(api_key=api_key)
        _gemini_configured = True
        return True
    except Exception as e:
        logging.error(f"❌ Failed to configure Gemini: {e}")
        return False

def get_gemini_response(response):
    try:
        # 1. Check for Safety Stops
        if response.candidates and response.candidates[0].finish_reason != 1: # 1 == STOP
            logging.warning(f"⚠️ Generation stopped. Reason: {response.candidates[0].finish_reason}")
            return None

        # 2. CRITICAL FIX: Check for Function Calls
        # Gemini 3 will do this often. If we don't catch it, accessing .text crashes.
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if part.function_call:
                logging.warning("⚠️ Model triggered a Function Call. Returning full response object.")
                return response 

        # 3. Safe Text Access
        return response.text

    except ValueError:
        # Handle empty responses gracefully so we don't trigger the retry loop
        return ""
    except Exception as e:
        logging.warning(f"⚠️ Error in get_gemini_response: {e}")
        return None

def call_gemini_with_retry(
        model_name: str, 
        prompt: str, 
        temperature: float = 0.0,
        generation_config: Optional[dict] = None,
        fallback_chain: Optional[list[str]] = None
    ) -> str:
    """
    Calls the Gemini model with proactive rate limiting and robust exponential backoff.
    Supports fallback models if the primary model fails.
    
    Args:
        model_name: "gemini-..."
        prompt: The input prompt.
        temperature: Model temperature.
        generation_config: Optional extra config.
        fallback_chain: List of model names to try if the primary fails.
        
    Returns:
        Generated text or empty string on failure.
    """
    if not ensure_gemini_configured():
        return "Error: Gemini not configured."
    
    # Create the full list of models to try
    models_to_try = [model_name]
    if fallback_chain:
        models_to_try.extend(fallback_chain)
        
    for current_model in models_to_try:
        logging.info(f"🤖 Attempting with model: {current_model}")
        
        # Configure model instance
        try:
            model = genai.GenerativeModel(current_model)
        except Exception as e:
            logging.error(f"❌ Failed to initialize Gemini model {current_model}: {e}")
            continue # Try next model
            
        backoff = GEMINI_INITIAL_BACKOFF
        success = False
        
        # Internal Retry Loop for the current model
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                # 1. Proactive Rate Limit
                _limiter.wait_for_token()
                
                # 2. Make Request
                # Prepare config parameters
                config_params = {"temperature": temperature}
                if generation_config:
                    config_params.update(generation_config)
                
                config = genai.types.GenerationConfig(**config_params)
                    
                response = model.generate_content(prompt, generation_config=config)
                
                # 3. process response
                text = get_gemini_response(response)
                
                if text is not None and text != "":
                    if hasattr(text, 'candidates'): 
                         logging.warning("⚠️ Received Function Call instead of text. Treating as empty for now.")
                         return ""
                    return text.strip() # SUCCESS!
                
                if text == "":
                     # Handled empty response, likely harmless but empty
                     pass
                
                # If None, it means error or safety stop
                logging.warning(f"⚠️ Gemini {current_model} response was None or invalid (Attempt {attempt}).")

            except Exception as e:
                error_str = str(e).lower()
                is_quota_error = "429" in error_str or "quota" in error_str or "resource exhausted" in error_str
                
                # Critical Failures that should trigger immediate fallback (no retry on this model)
                if is_quota_error and "limit: 0" in error_str:
                     logging.error(f"❌ Gemini Quota limit is 0 for {current_model}. Skipping to next model.")
                     break 
                
                if attempt == GEMINI_MAX_RETRIES:
                    logging.error(f"❌ {current_model} Failed after {GEMINI_MAX_RETRIES} attempts. Last error: {e}")
                    # Loop naturally ends, will proceed to next model in outer loop
                else:
                    # Calculate sleep time
                    sleep_time = backoff + random.uniform(0, 1)
                    
                    if is_quota_error:
                         logging.warning(f"⚠️ Gemini Quota (429) - {current_model} Attempt {attempt}. Sleeping {sleep_time:.1f}s...")
                    else:
                         logging.warning(f"⚠️ Gemini Error: {e} - Attempt {attempt}. Retrying...")
                    
                    time.sleep(sleep_time)
                    backoff = min(backoff * 2, 60.0)
    
    logging.error("❌ All models in the chain failed.")
    return ""
