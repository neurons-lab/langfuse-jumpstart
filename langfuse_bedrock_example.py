import os
import json
import boto3
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context

# Load environment variables from .env file
load_dotenv()

# Environment variables should be set in .env file:
# LANGFUSE_SECRET_KEY="sk-lf-..."
# LANGFUSE_PUBLIC_KEY="pk-lf-..."
# LANGFUSE_HOST="https://cloud.langfuse.com"  # EU region
# AWS credentials should be configured via environment variables or AWS config

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'  # Change to your preferred region
)

@observe(name="bedrock-claude-completion", 
         capture_input=True, 
         capture_output=True,
         model_parameters={
            "model": "anthropic.claude-3-sonnet-20240229",
            "max_tokens": 500,
            "temperature": 0.5
         })
def generate_with_bedrock_claude(prompt):
    """Generate text using Claude model on AWS Bedrock."""
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "temperature": 0.5,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })
    
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229',
        body=body
    )
    
    # Parse the response
    response_body = json.loads(response['body'].read().decode('utf-8'))
    
    # Add model information to the current observation
    langfuse_context.update_current_observation(
        model="anthropic.claude-3-sonnet-20240229"
    )
    
    return response_body['content'][0]['text']

@observe(name="bedrock-titan-completion", 
         capture_input=True, 
         capture_output=True,
         model_parameters={
            "model": "amazon.titan-text-express-v1",
            "maxTokenCount": 500,
            "temperature": 0.7
         })
def generate_with_bedrock_titan(prompt):
    """Generate text using Amazon Titan model on AWS Bedrock."""
    
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 500,
            "temperature": 0.7,
            "topP": 0.9
        }
    })
    
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-text-express-v1',
        body=body
    )
    
    # Parse the response
    response_body = json.loads(response['body'].read().decode('utf-8'))
    
    # Add model information to the current observation
    langfuse_context.update_current_observation(
        model="amazon.titan-text-express-v1"
    )
    
    return response_body['results'][0]['outputText']

@observe(name="bedrock-model-comparison")
def main():
    """Main function demonstrating AWS Bedrock models with Langfuse tracing."""
    # Set user and session information
    langfuse_context.update_current_trace(
        user_id="bedrock-user-789",
        session_id="model-comparison-123",
        tags=["bedrock", "comparison"]
    )
    
    # Generate responses using both models with the same prompt
    prompt = "Explain the concept of quantum entanglement in simple terms."
    
    try:
        claude_response = generate_with_bedrock_claude(prompt)
        print("\nClaude Response (excerpt):", claude_response[:200] + "...")
        
        # Add metadata to the observation
        langfuse_context.update_current_observation(
            metadata={"model_type": "anthropic"}
        )
    except Exception as e:
        print(f"Error with Claude: {e}")
        claude_response = f"Error: {str(e)}"
    
    try:
        titan_response = generate_with_bedrock_titan(prompt)
        print("\nTitan Response (excerpt):", titan_response[:200] + "...")
        
        # Add metadata to the observation
        langfuse_context.update_current_observation(
            metadata={"model_type": "amazon"}
        )
    except Exception as e:
        print(f"Error with Titan: {e}")
        titan_response = f"Error: {str(e)}"
    
    # Score the responses
    if "Error" not in claude_response:
        langfuse_context.score(
            name="clarity",
            value=0.85,
            comment="Clear explanation from Claude"
        )
    
    if "Error" not in titan_response:
        langfuse_context.score(
            name="clarity",
            value=0.80,
            comment="Good explanation from Titan"
        )
    
    return {
        "claude_response": claude_response,
        "titan_response": titan_response
    }

if __name__ == "__main__":
    try:
        result = main()
        print("\nProcess completed successfully!")
    finally:
        # Ensure all events are sent to Langfuse
        langfuse_context.flush() 