# in this file we turn text into vectors with gemini 

import asyncio
import logging
import os 
import math
from typing import Awaitable, Callable

from google.genai import types

from .import classifier

log = logging.getLogger("customer_desk.embeddings")


# these constants control how embeddigns are generated 
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
BATCH_SIZE = 20
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 30

# embedder defines a type functions this is a type alias it describes a callable that: accepts a list of strings and a string and returns an awaitable result that contains the floating - point numbers 
Embedder = Callable[[list[str], str], Awaitable[list[float]]]

# standardize vector length, normalized vectors are convenient for comparing semantic similarity using a cosine similarity or dot product
def normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x*x for x in v ))
    return [ x/norm for x in v] if norm else v 


# it is the actual gemini call it accepts the model the batch and kind( query or a document)
async def _embed_batch(model: str, batch: list[str], kind: str) -> list[list[float]]:
    # get the client 
    client = classifier._get_client()
    if model.startswith("gemini-embedding-2"):
        prefix = "task: search result | query: " if kind == "query" else "title: none | text: "
        contents = [types.Content(parts=[types.Part(text=prefix + t)]) for t in batch]
        config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
    else:
        contents = batch
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIM,
        )
    response = await client.aio.models.embed_content(model=model, contents=contents, config=config)
    vectors = [list(e.values) for e in (response.embeddings or [])]
    if len(vectors) != len(batch):
        raise ValueError(f"asked for {len(batch)} embeddings, got {len(vectors)}")
    return [normalize(v) for v in vectors]


# this functions manages the complete embedding process for a list of texts this functions handle long lists by splitting it inot batches 
async def embed_texts(texts: list[str], kind: str) -> list[list[float]]:
    model = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                out.extend(await asyncio.wait_for(_embed_batch(model, batch, kind), TIMEOUT_SECONDS))
                break
            except Exception as e:
                last_error = e
                log.warning("embedding attempt %d/%d failed: %r", attempt, MAX_ATTEMPTS, e)
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(2**attempt)
        else:
            raise RuntimeError(f"embedding failed after {MAX_ATTEMPTS} attempts: {last_error!r}")
    return out


def embedding_model_name() -> str:
    return os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)