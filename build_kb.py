"""Build the searchable knowledge index from the FAQ files in knowledge/.

Run from the project root:  python build_kb.py
Run it again whenever you edit anything in knowledge/.
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from app.embeddings import embed_texts, embedding_model_name  # noqa: E402
from app.knowledge import KnowledgeIndex, fingerprint, kb_dir, load_chunks, save_index  # noqa: E402


async def main() -> int:
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set in .env")
        return 2
    chunks = load_chunks()
    if not chunks:
        print(f"No .md files with content found in {kb_dir()}/")
        return 1
    print(f"Embedding {len(chunks)} chunks from {kb_dir()}/ with {embedding_model_name()} ...")
    vectors = await embed_texts([c.embed_text() for c in chunks], "document")
    path = save_index(
        KnowledgeIndex(model=embedding_model_name(), fingerprint=fingerprint(), chunks=chunks, vectors=vectors)
    )
    print(f"Saved {len(chunks)} chunks ({len(vectors[0])} dimensions each) to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))