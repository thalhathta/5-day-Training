from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


@dataclass
class Chunk:
    page: int
    text: str


class CoursePolicyAgent:
    """A lightweight document-grounded QA agent for course policy handbooks."""

    def __init__(
        self,
        pdf_path: str | Path,
        embed_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> None:
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        self.chunks: List[Chunk] = self._load_and_chunk_pdf()
        self.embs: np.ndarray = self._embed_all([c.text for c in self.chunks])

        self.system_rules = """You are a course policy assistant for university faculty.

Strict rules:
1) Answer ONLY using the provided CONTEXT from the course policy document.
2) If the answer is not explicitly present in the CONTEXT, respond exactly:
   Not found in the provided document.
3) When you answer, include:
   - Answer: (1-5 sentences)
   - Evidence: 2-4 bullet points with page numbers, each with a short quote (<= 20 words).
4) Do NOT invent policy details. Do NOT guess.
"""

    def _load_and_chunk_pdf(self) -> List[Chunk]:
        reader = PdfReader(str(self.pdf_path))
        chunks: List[Chunk] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            for c in self._chunk_text(text):
                chunks.append(Chunk(page=i, text=c))
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        out = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            out.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self.overlap)
        return out

    def _embed_all(self, texts: List[str]) -> np.ndarray:
        # Batch embeddings for speed
        BATCH = 64
        mats = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            resp = self.client.embeddings.create(model=self.embed_model, input=batch)
            mats.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
        return np.vstack(mats)

    def _embed_one(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.embed_model, input=[text])
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def _retrieve(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        q = self._embed_one(question)
        q = q / (np.linalg.norm(q) + 1e-12)

        m = self.embs
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
        sims = m @ q

        idx = np.argsort(-sims)[:k]
        return [
            {
                "score": float(sims[j]),
                "page": self.chunks[j].page,
                "text": self.chunks[j].text,
            }
            for j in idx
        ]

    def answer_query(self, question: str, k: int = 4) -> str:
        hits = self._retrieve(question, k=k)
        context = "\n\n".join([f"[Page {h['page']}] {h['text']}" for h in hits])

        resp = self.client.responses.create(
            model=self.llm_model,
            input=[
                {"role": "system", "content": self.system_rules},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
                },
            ],
            temperature=0.0,
        )
        return resp.output_text
