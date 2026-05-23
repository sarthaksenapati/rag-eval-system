from enum import Enum
from dataclasses import dataclass
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.embeddings.openai import OpenAIEmbedding
import sys, os
sys.path.append(".")
from backend.config import settings


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


@dataclass
class Chunk:
    text: str
    metadata: dict
    chunk_id: str
    strategy: str


def chunk_documents(docs: list[Document], strategy: ChunkStrategy) -> list[Chunk]:
    print(f"Chunking {len(docs)} documents with strategy: {strategy.value}")

    if strategy == ChunkStrategy.FIXED:
        splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        nodes = splitter.get_nodes_from_documents(docs)

    elif strategy == ChunkStrategy.SEMANTIC:
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"  # downloads once, runs locally, free forever
        )
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model
        )
        nodes = splitter.get_nodes_from_documents(docs)

    elif strategy == ChunkStrategy.HIERARCHICAL:
        # Creates parent chunks (2048 tokens) and child chunks (512 tokens)
        # We index the child chunks but retrieve parent chunks for more context
        splitter = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        all_nodes = splitter.get_nodes_from_documents(docs)
        nodes = get_leaf_nodes(all_nodes)  # only index the smallest chunks

    chunks = [
        Chunk(
            text=node.get_content(),
            metadata=node.metadata,
            chunk_id=node.node_id,
            strategy=strategy.value
        )
        for node in nodes
        if node.get_content().strip()  # skip empty nodes
    ]

    print(f"Produced {len(chunks)} chunks")
    return chunks