import os
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import openai  # OpenAI integration with auto-tracing

# Load environment variables from .env file
load_dotenv()

# Environment variables should be set in .env file:
# LANGFUSE_SECRET_KEY="sk-lf-..."
# LANGFUSE_PUBLIC_KEY="pk-lf-..."
# LANGFUSE_HOST="https://cloud.langfuse.com"  # EU region
# OPENAI_API_KEY="sk-..."

@observe()
def generate_story(topic):
    """Generate a short story about the given topic."""
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=150,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": f"Write a short story about {topic}."}
        ]
    ).choices[0].message.content

@observe()
def generate_summary(text):
    """Generate a summary of the given text."""
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=50,
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize this in one sentence: {text}"}
        ]
    ).choices[0].message.content

@observe()
def classify_sentiment(text):
    """Classify the sentiment of the given text."""
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Classify the sentiment as POSITIVE, NEGATIVE, or NEUTRAL."},
            {"role": "user", "content": text}
        ]
    ).choices[0].message.content

@observe(name="creative-writing-flow")
def main():
    """Main function that orchestrates the creative writing flow."""
    # Update user and session information for the current trace
    langfuse_context.update_current_trace(
        user_id="example-user-123",
        session_id="example-session-456"
    )
    
    # Generate a story about space
    story = generate_story("space exploration")
    print("\nStory:", story)
    
    # Summarize the story
    summary = generate_summary(story)
    print("\nSummary:", summary)
    
    # Classify the sentiment of the summary
    sentiment = classify_sentiment(summary)
    print("\nSentiment:", sentiment)
    
    # Score this specific trace (e.g., for evaluation)
    langfuse_context.score_current_trace(
        name="creativity",
        value=0.95,
        comment="This was a highly creative story about space"
    )
    
    return {
        "story": story,
        "summary": summary,
        "sentiment": sentiment
    }

if __name__ == "__main__":
    try:
        result = main()
        print("\nProcess completed successfully!")
    finally:
        # Ensure all events are sent to Langfuse
        # This is important in short-lived environments like AWS Lambda
        langfuse_context.flush() 