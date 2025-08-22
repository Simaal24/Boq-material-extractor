"""
Gemini API Client Module

This module provides a centralized, robust interface for communicating with the Google Gemini API.
It handles authentication, rate limiting, retries, and error handling so other modules don't have to.
"""

import asyncio
import time
import json
import logging
import threading
from typing import Optional, Dict, Any

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: Please install: pip install google-generativeai")
    raise

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for Gemini API - ULTRA-CONSERVATIVE for low quotas"""
    
    def __init__(self, max_requests_per_minute: int = 900):
        self.max_requests = max_requests_per_minute
        self.request_times = []
        self.lock = asyncio.Lock()  # Use asyncio.Lock for async context
        logger.info(f"🚦 Rate Limiter: {max_requests_per_minute} requests/minute max")
    
    async def acquire(self):
        """Acquire permission to make a request with optimized rate limiting"""
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [req_time for req_time in self.request_times if now - req_time < 60]
            
            # If we're at the limit, wait
            if len(self.request_times) >= self.max_requests:
                sleep_time = 60 - (now - self.request_times[0]) + 1
                logger.info(f"🚦 QUOTA PROTECTION: Waiting {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                
                # Clean up old requests again after waiting
                now = time.time()
                self.request_times = [req_time for req_time in self.request_times if now - req_time < 60]
            
            # Record this request
            self.request_times.append(now)
            
            # Add smaller buffer between requests for high quotas
            if len(self.request_times) > 1:
                time_since_last = now - self.request_times[-2]
                if time_since_last < 2:
                    buffer_wait = 2 - time_since_last
                    logger.info(f"🚦 BUFFER: Adding {buffer_wait:.1f}s buffer")
                    await asyncio.sleep(buffer_wait)

class GeminiClient:
    """Centralized Gemini API client with robust error handling and rate limiting"""
    
    def __init__(self, api_key: str, max_requests_per_minute: int = 900):
        """Initialize the Gemini client
        
        Args:
            api_key: Google Gemini API key
            max_requests_per_minute: Rate limit for API calls
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model with optimal settings for BOQ processing
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent results
                max_output_tokens=16384,  # Increased for longer responses
                response_mime_type="application/json"  # Force JSON responses
            ),
            safety_settings=safety_settings
        )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        
        # Track API usage
        self.api_calls = 0
        
        logger.info(f"🤖 Gemini Client initialized with rate limit: {max_requests_per_minute}/minute")
    
    async def generate_content(self, prompt: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Generate content using Gemini API with robust error handling
        
        Args:
            prompt: The prompt to send to Gemini
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed JSON response from Gemini, or None if failed
        """
        base_delay = 0.5  # Faster retries
        
        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                await self.rate_limiter.acquire()
                
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info(f"🔄 Retry attempt {attempt + 1}/{max_retries + 1} after {delay}s delay")
                    await asyncio.sleep(delay)
                
                # Make the API call
                logger.debug(f"📤 Making Gemini API call #{self.api_calls + 1}")
                response = await self.model.generate_content_async(prompt)
                self.api_calls += 1
                
                # Check response validity
                if not response.candidates:
                    raise ValueError("No candidates in Gemini response")
                
                candidate = response.candidates[0]
                if candidate.finish_reason != 1:  # 1 = STOP (successful completion)
                    finish_reasons = {
                        0: "FINISH_REASON_UNSPECIFIED",
                        1: "STOP", 
                        2: "MAX_TOKENS",
                        3: "SAFETY",
                        4: "RECITATION",
                        5: "OTHER"
                    }
                    reason_name = finish_reasons.get(candidate.finish_reason, f"UNKNOWN({candidate.finish_reason})")
                    raise ValueError(f"Gemini stopped with reason: {reason_name} ({candidate.finish_reason})")
                
                if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                    raise ValueError("No content parts in Gemini response")
                
                response_text = candidate.content.parts[0].text
                if not response_text:
                    raise ValueError("Empty response text from Gemini")
                
                # Parse JSON response
                parsed_response = self._parse_json_response(response_text)
                
                if parsed_response is not None:
                    logger.debug(f"✅ Gemini API call successful")
                    return parsed_response
                else:
                    raise json.JSONDecodeError("Failed to parse JSON", response.text, 0)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"🔄 JSON decode error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    logger.error(f"❌ JSON parsing failed after {max_retries + 1} attempts")
                    return None
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for non-retriable errors
                non_retriable_errors = [
                    "api key not valid", "invalid api key", "unauthorized", 
                    "403", "401", "billing", "quota exceeded permanently"
                ]
                
                is_non_retriable = any(phrase in error_msg for phrase in non_retriable_errors)
                
                if is_non_retriable:
                    logger.error(f"❌ Non-retriable error: {e}")
                    return None
                
                logger.warning(f"🔄 API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    logger.error(f"❌ API failed after {max_retries + 1} attempts: {e}")
                    return None
        
        return None
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from Gemini, handling markdown formatting
        
        Args:
            response_text: Raw response text from Gemini
            
        Returns:
            Parsed JSON data or None if parsing failed
        """
        try:
            # Clean up the response - remove markdown formatting
            cleaned_response = response_text.strip()
            
            # Handle markdown code blocks
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Find JSON array if not at the start
            if not cleaned_response.startswith('[') and not cleaned_response.startswith('{'):
                json_start = cleaned_response.find('[')
                if json_start == -1:
                    json_start = cleaned_response.find('{')
                
                if json_start >= 0:
                    if cleaned_response.startswith('['):
                        json_end = cleaned_response.rfind(']') + 1
                    else:
                        json_end = cleaned_response.rfind('}') + 1
                    
                    if json_end > json_start:
                        cleaned_response = cleaned_response[json_start:json_end]
            
            # Parse the JSON
            parsed_data = json.loads(cleaned_response)
            logger.debug(f"📄 JSON parsed successfully: {type(parsed_data)}")
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing failed: {e}")
            logger.debug(f"Raw response (first 500 chars): {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing response: {e}")
            return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics
        
        Returns:
            Dictionary with usage stats
        """
        return {
            "total_api_calls": self.api_calls,
            "rate_limit": self.rate_limiter.max_requests,
            "requests_in_last_minute": len(self.rate_limiter.request_times)
        }

# Global client instance (will be initialized when needed)
_client_instance: Optional[GeminiClient] = None

def initialize_client(api_key: str, max_requests_per_minute: int = 900) -> GeminiClient:
    """Initialize the global Gemini client instance
    
    Args:
        api_key: Google Gemini API key
        max_requests_per_minute: Rate limit for API calls
        
    Returns:
        Initialized GeminiClient instance
    """
    global _client_instance
    _client_instance = GeminiClient(api_key, max_requests_per_minute)
    logger.info("🤖 Global Gemini client initialized")
    return _client_instance

def get_client() -> Optional[GeminiClient]:
    """Get the global Gemini client instance
    
    Returns:
        GeminiClient instance or None if not initialized
    """
    if _client_instance is None:
        logger.warning("⚠️ Gemini client not initialized. Call initialize_client() first.")
    return _client_instance

async def generate_content(prompt: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Convenience function to generate content using the global client
    
    Args:
        prompt: The prompt to send to Gemini
        max_retries: Maximum number of retry attempts
        
    Returns:
        Parsed JSON response from Gemini, or None if failed
    """
    # Ensure we have a client
    client = get_client()
    if client is None:
        logger.error("❌ Cannot generate content: Gemini client not initialized")
        return None
    
    return await client.generate_content(prompt, max_retries)

def get_usage_stats() -> Dict[str, Any]:
    """Get usage statistics from the global client
    
    Returns:
        Dictionary with usage stats or empty dict if client not initialized
    """
    client = get_client()
    if client is None:
        return {"error": "Client not initialized"}
    
    return client.get_usage_stats()