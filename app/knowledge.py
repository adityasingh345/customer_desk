
import hashlib
import json
import logging
import os
import re
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger("customer_desk.knowledge")

MAX_CHUNK_CHARS = 1200


class Chunk(BaseModel):
    id: str 
    source: str  
    title: str  
    text: str  

    def embed_text(self) -> str:
        #What we feed to the embedding model: the question AND its answer
        return f"{self.title}\n{self.text}"


class KnowledgeIndex(BaseModel):
    model: str 
    fingerprint: str  
    chunks: list[Chunk]
    vectors: list[list[float]]  


def kb_dir() -> Path:
    return Path(os.getenv("KB_DIR", "knowledge"))


def index_path() -> Path:
    return Path(os.getenv("DATA_DIR", "data")) / "kb_index.json"


def _split_long(body: str) -> list[str]:
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", body):
        if current and len(current) + len(para) + 2 > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts or [body.strip()]


# cuts each file at every ## heading so one question plus its answers become one chunk. chunk size is the main quality lever in RAG 
def chunk_markdown(text: str, source: str) -> list[Chunk]:
    stem = Path(source).stem
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  
    sections: list[tuple[str, str]] = [] 
    title, lines = "Overview", []
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((title, "\n".join(lines)))
            title, lines = line[3:].strip(), []
        elif line.startswith("# ") and not sections and not lines:
            continue  
        else:
            lines.append(line)
    sections.append((title, "\n".join(lines)))

    chunks: list[Chunk] = []
    for sec_title, body in sections:
        body = body.strip()
        if not body:
            continue
        for piece in _split_long(body):
            chunks.append(Chunk(id=f"{stem}#{len(chunks) + 1}", source=source, title=sec_title, text=piece))
    return chunks


def load_chunks(folder: Path | None = None) -> list[Chunk]:
    folder = folder or kb_dir()
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


def fingerprint(folder: Path | None = None) -> str:
    folder = folder or kb_dir()
    h = hashlib.sha256()
    for path in sorted(folder.glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def save_index(index: KnowledgeIndex, path: Path | None = None) -> Path:
    path = path or index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.model_dump_json(), encoding="utf-8")
    return path


def load_index(path: Path | None = None) -> KnowledgeIndex | None:
    path = path or index_path()
    if not path.exists():
        return None
    return KnowledgeIndex.model_validate(json.loads(path.read_text(encoding="utf-8")))