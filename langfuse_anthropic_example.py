import os
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Environment variables should be set in .env file:
# LANGFUSE_SECRET_KEY="sk-lf-..."
# LANGFUSE_PUBLIC_KEY="pk-lf-..."
# LANGFUSE_HOST="https://cloud.langfuse.com"  # EU region
# ANTHROPIC_API_KEY="sk-ant-..."

# Initialize Anthropic client
anthropic = Anthropic()

@observe(name="anthropic-completion", 
         capture_input=True, 
         capture_output=True,
         model_parameters={
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "temperature": 0.7
         })
def generate_with_claude(prompt):
    """Generate text using Anthropic's Claude model."""
    response = anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Manually add usage data to the current observation
    if hasattr(response, "usage"):
        langfuse_context.update_current_observation(
            usage={
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)
            }
        )
    
    return response.content[0].text

@observe(name="research-assistant")
def main():
    """Main function demonstrating Claude with Langfuse tracing."""
    # Set user and session information
    langfuse_context.update_current_trace(
        user_id="anthropic-user-456",
        session_id="research-session-123",
        tags=["anthropic", "research"]
    )
    
    # Generate a comprehensive research summary
    prompt = """
    Write a comprehensive research summary on quantum computing, including:
    1. Basic principles
    2. Current state of the technology
    3. Major challenges
    4. Potential applications
    5. Future outlook
    
    Make it accessible to someone with a basic understanding of physics.
    """
    
    research_summary = generate_with_claude(prompt)
    print("\nResearch Summary (excerpt):", research_summary[:300] + "...")
    
    # Ask a follow-up question
    follow_up_prompt = f"""
    Based on the following research summary on quantum computing, 
    what are the 3 most promising near-term applications?
    
    {research_summary}
    """
    
    applications = generate_with_claude(follow_up_prompt)
    print("\nPromising Applications (excerpt):", applications[:300] + "...")
    
    # Score this trace
    langfuse_context.score_current_trace(
        name="comprehensiveness", 
        value=0.9,
        comment="Provided detailed information on quantum computing"
    )
    
    return {
        "research_summary": research_summary,
        "applications": applications
    }

if __name__ == "__main__":
    try:
        result = main()
        print("\nProcess completed successfully!")
    finally:
        # Ensure all events are sent to Langfuse
        langfuse_context.flush() 