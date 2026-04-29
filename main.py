# ============================================================
# MAIN ENTRY POINT
# Run this to index your PDFs and start asking questions
# ============================================================

from ingestion.pdf_loader import load_all_pdfs
from vectorstore.store import index_chunks, get_store_count
from retrieval.retriever import retrieve, format_context
from llm.claude_client import ask_claude
from utils.logger import get_logger

logger = get_logger("main")


def index():
    """Index all PDFs from the data/pdfs folder."""
    logger.info("Starting PDF indexing...")
    chunks = load_all_pdfs()
    if chunks:
        index_chunks(chunks)
        logger.info(f"Done! Total chunks in store: {get_store_count()}")
    else:
        logger.warning("No chunks indexed. Add PDFs to data/pdfs/")


def ask(question: str):
    """Ask a question against indexed PDFs."""
    logger.info(f"Question: {question}")
    chunks = retrieve(question)

    if not chunks:
        print("⚠️ No relevant documents found. Please index PDFs first.")
        return

    context = format_context(chunks)
    result = ask_claude(question, context)

    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print("\n" + "="*60)
        print(result["answer"])
        print("="*60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py index              — index all PDFs")
        print("  python main.py ask 'your question'")
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        index()
    elif command == "ask":
        question = " ".join(sys.argv[2:])
        ask(question)
    else:
        print(f"Unknown command: {command}")
