# Langfuse Tracing - Decorator Quickstart

This example demonstrates how to use Langfuse's decorator approach to trace LLM calls in a Python application.

## Setup

1. Install the required packages:

```bash
pip install langfuse python-dotenv openai
```

2. Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

3. Open `.env` and add your Langfuse and OpenAI API keys:
   - `LANGFUSE_SECRET_KEY` - from your Langfuse dashboard
   - `LANGFUSE_PUBLIC_KEY` - from your Langfuse dashboard
   - `OPENAI_API_KEY` - your OpenAI API key

## Running the Example

Run the example with:

```bash
python langfuse_quickstart.py
```

This will:
1. Generate a short story about space exploration
2. Create a summary of the story
3. Classify the sentiment of the summary
4. Log all of these steps to Langfuse with automatic tracing

## How It Works

The code uses Langfuse's `@observe()` decorator to automatically trace function calls:

```python
from langfuse.decorators import observe

@observe()
def my_function():
    # Function code here
    pass
```

Key features demonstrated:

1. **Nested Tracing**: The main function calls other traced functions, creating a hierarchical trace
2. **OpenAI Integration**: Using `from langfuse.openai import openai` for automatic tracing of OpenAI calls
3. **User & Session Tracking**: Setting user and session IDs for analytics
4. **Scoring**: Adding evaluation scores to traces
5. **Proper Flushing**: Ensuring all events are sent to Langfuse

## Viewing Traces

After running the example, visit your [Langfuse Dashboard](https://cloud.langfuse.com) to see the traces. You'll find:

- A hierarchical view of the main function and its nested calls
- Detailed information about each OpenAI API call
- Input/output pairs for each function
- Timing information
- The evaluation score

## Next Steps

- Add more functions to your trace
- Integrate with other frameworks like LangChain or LlamaIndex
- Add custom metadata to your traces
- Set up evaluations and scoring for quality monitoring 