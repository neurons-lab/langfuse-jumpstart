#!/usr/bin/env python
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.decorators import observe
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# Load environment variables from .env file
load_dotenv()

# Initialize the Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Create a Langfuse callback handler for LangChain
handler = CallbackHandler()

@observe(name="summarize_text")
def summarize_text(text, user_id=None):
    """Summarize text using LangChain with Langfuse tracing.
    
    The @observe decorator automatically:
    - Creates a span in Langfuse
    - Captures the function name as the span name
    - Records the input parameters
    - Records the return value
    - Captures the execution time
    """
    # Set user ID for this trace
    from langfuse.decorators import langfuse_context
    current_trace = langfuse_context.current_trace()
    if current_trace and user_id:
        current_trace.update_metadata(user_id=user_id)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        callbacks=[handler]  # Add the Langfuse callback handler
    )
    
    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Please summarize the following text in 2-3 sentences:\n\n{text}"
    )
    
    # Create a chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain with Langfuse tracing
    return chain.invoke({"text": text})

@observe(name="qa_chain")
def answer_question(question, context, user_id=None):
    """Answer a question based on context using LangChain with Langfuse tracing."""
    # Set user ID for this trace
    from langfuse.decorators import langfuse_context
    current_trace = langfuse_context.current_trace()
    if current_trace and user_id:
        current_trace.update_metadata(user_id=user_id)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        callbacks=[handler]  # Add the Langfuse callback handler
    )
    
    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the following context, please answer the question accurately and concisely.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    # Create a chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain with Langfuse tracing
    result = chain.invoke({
        "context": context,
        "question": question
    })
    
    # Add score to the trace based on confidence
    # This is just a placeholder logic - in real applications, you would have a better way to measure confidence
    if "I don't know" in result["text"] or "I'm not sure" in result["text"]:
        confidence = 0.3
    elif "might be" in result["text"] or "possibly" in result["text"]:
        confidence = 0.7
    else:
        confidence = 0.9
    
    # Record the confidence score in Langfuse
    current_trace = langfuse_context.current_trace()
    if current_trace:
        current_trace.score(name="answer_confidence", value=confidence)
    
    return result

@observe(name="multi_step_workflow")
def process_document(document, user_id=None):
    """Process a document with multiple LangChain steps, all traced in Langfuse."""
    # Set user ID for this trace
    from langfuse.decorators import langfuse_context
    current_trace = langfuse_context.current_trace()
    if current_trace and user_id:
        current_trace.update_metadata(user_id=user_id)
    
    # First step: Summarize the document
    summary_result = summarize_text(document)
    summary = summary_result["text"]
    
    # Second step: Extract key information
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        callbacks=[handler]  # Add the Langfuse callback handler
    )
    
    extraction_prompt = PromptTemplate(
        input_variables=["document"],
        template="Extract the key entities, dates, and numerical values from this text. Format as a JSON.\n\n{document}"
    )
    
    extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)
    extraction_result = extraction_chain.invoke({"document": document})
    
    # Add a custom observation to Langfuse
    current_trace = langfuse_context.current_trace()
    if current_trace:
        current_trace.observation(
            name="document_extraction",
            input={"document": document},
            output=extraction_result["text"],
            level="INFO"
        )
    
    return {
        "summary": summary,
        "extracted_info": extraction_result["text"]
    }

def main():
    """Run examples of Langfuse tracing with LangChain."""
    
    # Example 1: Text summarization
    text_to_summarize = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    """
    
    summary_result = summarize_text(text_to_summarize, user_id="user-456")
    print(f"Summarized text:\n{summary_result['text']}\n")
    
    # Example 2: Question answering
    context = """
    Langfuse is an open-source LLM engineering platform focused on tracing, evaluation, and monitoring.
    It helps developers debug, analyze, and improve their LLM applications in production.
    Langfuse was founded in 2023 and supports integrations with OpenAI, LangChain, LlamaIndex, and other popular LLM frameworks.
    """
    question = "What is Langfuse and what integrations does it support?"
    
    qa_result = answer_question(question, context, user_id="user-789")
    print(f"Answer to question:\n{qa_result['text']}\n")
    
    # Example 3: Multi-step document processing
    document = """
    QUARTERLY REPORT
    Q3 2023
    
    Revenue: $10.2M (up 15% YoY)
    Operating Expenses: $7.5M
    Net Profit: $2.7M
    
    Key Highlights:
    - Launched new product line on September 15th
    - Expanded to European market, opening office in Berlin
    - Hired 25 new employees, bringing total headcount to 120
    - Customer retention rate improved to 92% from 87% in previous quarter
    """
    
    processing_result = process_document(document, user_id="user-101")
    print(f"Document Summary:\n{processing_result['summary']}\n")
    print(f"Extracted Information:\n{processing_result['extracted_info']}\n")
    
    # Flush to ensure all events are sent before the program exits
    langfuse.flush()
    print("All examples completed and traces sent to Langfuse.")

if __name__ == "__main__":
    main() 