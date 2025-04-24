# Langfuse Examples

This repository contains examples of integrating Langfuse tracing into LLM applications using different models and frameworks.

## Setup

1. Install the required dependencies:
   ```bash
   pip install langfuse openai anthropic boto3 python-dotenv langchain langchain_openai
   ```

2. Copy the `.env.example` file to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file with your Langfuse and model provider API keys.

## Examples

### 1. OpenAI Integration

The `langfuse_openai_example.py` file demonstrates how to use Langfuse with the OpenAI API.

**Features:**
- Automatic tracing of OpenAI API calls
- User and session tracking
- Trace scoring

**Run the example:**
```bash
python langfuse_openai_example.py
```

### 2. Anthropic Integration

The `langfuse_anthropic_example.py` file shows how to integrate Langfuse with Anthropic's Claude models.

**Features:**
- Manual usage tracking for Anthropic API calls
- Nested spans for complex workflows
- User and session context

**Run the example:**
```bash
python langfuse_anthropic_example.py
```

### 3. AWS Bedrock Integration

The `langfuse_bedrock_example.py` file demonstrates how to use Langfuse with AWS Bedrock models.

**Features:**
- Model comparison between Claude and Titan models
- Parallel execution tracking
- Performance scoring

**Run the example:**
```bash
python langfuse_bedrock_example.py
```

### 4. LangChain Integration

The `langfuse_langchain_example.py` file shows how to use Langfuse with LangChain.

**Features:**
- Integration with LangChain callbacks
- Tracking of question-answering chains
- Simulated retrieval system

**Run the example:**
```bash
python langfuse_langchain_example.py
```

## How It Works

All examples use Langfuse's `@observe()` decorator, which automatically captures:

- Function inputs and outputs
- Execution time
- Nested spans for tracing complex workflows
- LLM-specific parameters and usage statistics

The decorator approach makes it easy to add observability to your code with minimal changes.

## Viewing Traces

After running any example, you can view the traces in the Langfuse UI:

1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com) (or your self-hosted instance)
2. Navigate to the Traces section
3. Find your trace by name, user ID, or session ID

In the UI, you can see the full execution flow, model parameters, inputs/outputs, and performance metrics.

## Decorator Features

The `@observe()` decorator offers several features:

- **Automatic nesting**: Traces maintain their hierarchy when decorated functions call each other
- **OpenAI integration**: Special handling for OpenAI calls to capture model, tokens, and costs
- **Context management**: The `langfuse_context` module provides access to the current trace
- **User and session tracking**: Easy association of traces with users and sessions
- **Scoring**: Add subjective scores to evaluate trace quality
- **Proper flushing**: Ensures all events are sent to Langfuse before program exit

## Next Steps

1. Explore more advanced Langfuse features like custom observations and metrics
2. Integrate Langfuse into your own LLM applications
3. Set up automated scoring and feedback collection
4. Use Langfuse for A/B testing different prompts and models 