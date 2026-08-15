"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
from typing import Any, Callable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> Callable[[str], list[dict[str, str]]]:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt: str) -> list[dict[str, str]]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task 1: Retrieve context and answer questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(
    vector_store: Any,
    llm: Callable[[str], list[dict[str, str]]],
    question: str,
) -> dict[str, str | list[str]]:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    documents = vector_store.similarity_search(question, k=3)
    sources = [document.page_content for document in documents]
    context = "\n\n".join(sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = llm(prompt)

    if not result or "generated_text" not in result[0]:
        raise RuntimeError("The language model did not return generated text.")

    return {
        "answer": result[0]["generated_text"].strip(),
        "sources": sources,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task 2: Interactive and single-query CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _print_result(result: dict[str, str | list[str]]) -> None:
    """Print a Q&A result in a readable CLI format."""
    print("\nSources:")
    for index, source in enumerate(result["sources"], start=1):
        print(f"  {index}. {source}")
    print(f"\nAnswer: {result['answer']}\n")


def main() -> None:
    """Interactive Q&A loop.

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer
    """
    parser = argparse.ArgumentParser(description="Ask questions about the agency.")
    parser.add_argument(
        "--query",
        help="Ask one question and exit instead of starting the interactive prompt.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.isdir(data_dir) or not any(
        filename.endswith(".txt") for filename in os.listdir(data_dir)
    ):
        parser.error(f"No text documents found in data directory: {data_dir}")

    vector_store = build_knowledge_base(data_dir)
    llm = get_llm()

    if args.query is not None:
        try:
            _print_result(ask_question(vector_store, llm, args.query))
        except ValueError as error:
            parser.error(str(error))
        return

    print('Ask a question about the agency, or type "quit" to exit.')
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() == "quit":
            print("Goodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue

        _print_result(ask_question(vector_store, llm, question))


if __name__ == "__main__":
    main()
