#!/usr/bin/env python3
"""
Langfuse OpenAI Integration Example

This script demonstrates how to use Langfuse with OpenAI using the decorator approach.
"""

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

# Load environment variables from .env file
load_dotenv()

# Initialize Langfuse client - will use env vars LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
langfuse = Langfuse()

# Initialize OpenAI client - will use OPENAI_API_KEY env var
client = OpenAI()

@observe(name="extract_keywords")
def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a keyword extraction assistant. Extract the 5 most important keywords from the text."},
                {"role": "user", "content": text}
            ]
        )
        
        # Parse keywords from response
        keywords = response.choices[0].message.content.strip().split("\n")
        keywords = [k.strip() for k in keywords]
        
        # Add custom score based on response quality
        if len(keywords) >= 5:
            langfuse_context.score_current_trace(name="keyword_quality", value=0.9)
        else:
            langfuse_context.score_current_trace(name="keyword_quality", value=0.5)
            
        return keywords
    except Exception as e:
        # The error will be automatically captured and linked to this trace
        raise e

@observe(name="generate_summary")
def generate_summary(text: str, max_length: int = 100) -> str:
    """Generate a summary of the given text."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.3,
        messages=[
            {"role": "system", "content": f"You are a summarization assistant. Summarize the following text in {max_length} words or less."},
            {"role": "user", "content": text}
        ]
    )
    
    summary = response.choices[0].message.content.strip()
    
    # Add token usage information as observation metadata
    langfuse_context.observation_current_span(
        name="token_usage",
        metadata={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    )
    
    return summary

@observe(name="analyze_sentiment")
def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze the sentiment of the given text."""
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant. Analyze the sentiment of the text and return a JSON with scores for positive, negative, and neutral sentiments. The scores should sum to 1.0."},
            {"role": "user", "content": text}
        ]
    )
    
    # Parse the JSON response
    try:
        sentiment_data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # If not valid JSON, try a simpler approach
        content = response.choices[0].message.content.strip()
        sentiment_data = {
            "positive": 0.0,
            "neutral": 0.0,
            "negative": 0.0
        }
        
        # Simple parsing from text
        if "positive" in content.lower():
            sentiment_data["positive"] = 0.7
            sentiment_data["neutral"] = 0.3
        elif "negative" in content.lower():
            sentiment_data["negative"] = 0.7
            sentiment_data["neutral"] = 0.3
        else:
            sentiment_data["neutral"] = 1.0
    
    return sentiment_data

@observe(name="content_workflow")
def process_content(text: str, user_id: str = None) -> Dict[str, Any]:
    """Process text content through a complete workflow."""
    # Set user ID for the current trace if provided
    if user_id:
        langfuse_context.update_current_trace(user_id=user_id)
    
    # Add tags to the current trace
    langfuse_context.update_current_trace(tags=["content-processing", "production"])
    
    # Extract keywords
    keywords = extract_keywords(text)
    
    # Generate summary
    summary = generate_summary(text)
    
    # Analyze sentiment
    sentiment = analyze_sentiment(text)
    
    # Add an overall quality score based on the results
    langfuse_context.score_current_trace(
        name="overall_quality",
        value=sentiment.get("positive", 0) * 0.7 + sentiment.get("neutral", 0) * 0.5
    )
    
    return {
        "keywords": keywords,
        "summary": summary,
        "sentiment": sentiment
    }

def main():
    """Main function to demonstrate OpenAI integration with Langfuse."""
    # Example text for processing
    sample_text = """
    Artificial intelligence has transformed industries worldwide, enabling automation 
    and insights previously unattainable. However, its rapid advancement raises important 
    ethical considerations regarding privacy, bias, and the future of work. Responsible AI 
    development requires careful governance and transparent practices to ensure benefits 
    are widely shared while minimizing potential harms.
    """
    
    try:
        # Process the content with user ID for tracking
        result = process_content(sample_text, user_id="demo-user-123")
        
        # Print the results
        print("\n=== Content Processing Results ===")
        print(f"Summary: {result['summary']}")
        print(f"Keywords: {', '.join(result['keywords'])}")
        print(f"Sentiment: {result['sentiment']}")
        
        # Perform individual API calls
        print("\n=== Individual API Calls ===")
        sentiment = analyze_sentiment("I really enjoyed this product. It exceeded my expectations!")
        print(f"Positive sentiment analysis: {sentiment}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Ensure all events are sent to Langfuse
        langfuse.flush()
        print("\nAll events sent to Langfuse.")

if __name__ == "__main__":
    main() 