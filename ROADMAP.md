# Multi-Source RAG Learning Roadmap

This roadmap is a living document for the project. We will update it as each milestone is completed and as the project becomes more realistic.

## Goal

Build a multi-source Retrieval-Augmented Generation (RAG) system step by step while learning the Python, embeddings, retrieval, vector databases, and LLM concepts behind it.

## Big Mental Model

RAG has two major phases:

```text
Offline / ingestion time:
sources -> chunks -> embeddings -> vector database

Online / query time:
question -> embedding -> retrieve chunks -> prompt LLM -> answer with sources
```

Multi-source RAG adds source-specific loaders before the shared pipeline:

```text
PDF loader
Web loader
CSV loader
Text loader
        -> common document format
        -> embeddings
        -> vector database
        -> retrieval
        -> answer generation
```

The key idea:

```text
Different sources should become the same document structure before retrieval.
```

## Current Document Shape

Each chunk becomes a structured document:

```python
{
    "text": "chunk content here",
    "metadata": {
        "source": "data/sample.txt",
        "source_type": "text",
        "chunk_index": 0
    },
    "embedding": [0.02, -0.14, 0.77]
}
```

## Milestones

### Milestone 1: Text Ingestion Pipeline

Status: Complete

Build the first pipeline using one local text file.

Concepts:

- Read a local text file.
- Split raw text into chunks.
- Add chunk overlap.
- Convert chunks into structured documents.
- Add metadata such as source, source type, and chunk index.
- Organize the code into reusable functions.

Completed notes:

- Built `load_text_file`, `chunk_text`, `create_documents`, and `process_text_file`.
- Standardized the first document shape with `text`, `metadata.source`, `metadata.source_type`, and `metadata.chunk_index`.

Expected pipeline:

```text
text file -> raw text -> chunks -> documents with metadata
```

### Milestone 2: Embeddings

Status: Complete

Understand how text becomes vectors.

Concepts:

- Fake embeddings for learning the data flow.
- Real embeddings using `sentence-transformers`.
- Difference between keyword-count vectors and semantic vectors.
- Why embedding values are not manually interpreted one by one.

Completed notes:

- Started with fake keyword-count embeddings to understand the flow.
- Replaced fake embeddings with `sentence-transformers`.
- Used `all-MiniLM-L6-v2` to create real semantic embeddings.

Expected pipeline:

```text
documents -> embedding vectors
query -> embedding vector
```

### Milestone 3: Similarity Search

Status: Complete

Understand how a query vector is compared with document vectors.

Similarity and distance methods to learn:

- Squared Euclidean distance
- Euclidean distance
- Cosine similarity
- Dot product
- Manhattan distance

Key ideas:

- Lower distance usually means more similar.
- Higher similarity usually means more similar.
- Cosine similarity is very common for text embeddings.

Expected pipeline:

```text
query embedding -> compare with document embeddings -> top K results
```

Completed notes:

- Implemented squared Euclidean distance, cosine similarity, dot product, and Manhattan distance.
- Compared lower-is-better distance methods with higher-is-better similarity methods.
- Refactored retrieval into a generic `search_documents` function that accepts a scoring function and sort direction.

### Milestone 4: In-Memory Retriever

Status: Complete

Build a small retriever using Python lists before adding a vector database.

Concepts:

- Store embedded documents in memory.
- Search over embedded documents.
- Return top K chunks.
- Keep metadata attached to results.

Completed notes:

- Built an `InMemoryVectorStore` class.
- Added documents into the store with embeddings.
- Queried the store using configurable scoring functions.
- Preserved metadata in retrieved results.

Expected pipeline:

```text
embedded documents + query -> ranked search results
```

### Milestone 5: Vector Database

Status: Complete

Move from in-memory search to a real vector database.

Likely first database:

- ChromaDB

Concepts:

- Collections
- Adding documents
- Adding embeddings
- Metadata storage
- Similarity search
- Persisting data locally

Completed notes:

- Installed and used ChromaDB.
- Created a persistent local Chroma database in `chroma_db/`.
- Built a `ChromaDBStore` wrapper.
- Used `upsert` for repeatable ingestion.
- Formatted raw Chroma query output into clean result dictionaries.
- Added metadata filtering with `where`.
- Separated ingestion and querying into separate functions.

Expected pipeline:

```text
documents -> embeddings -> vector DB -> query -> top K chunks
```

### Milestone 6: Prompt Construction

Status: Current

Turn retrieved chunks into useful context for an LLM.

Concepts:

- Build a context block from retrieved chunks.
- Include metadata for citations.
- Ask the model to answer only from provided context.
- Handle unknown answers.

Expected pipeline:

```text
top K chunks -> prompt context -> LLM prompt
```

### Milestone 7: Answer Generation

Status: Not started

Use an LLM to generate answers from retrieved context.

Concepts:

- Local vs API-based LLMs.
- Prompt templates.
- Grounded answers.
- Citations.
- Refusal when context is insufficient.

Expected pipeline:

```text
question + retrieved context -> answer with sources
```

### Milestone 8: Multi-Source Ingestion

Status: Not started

Add more source types while keeping one common document format.

Possible sources:

- Multiple text files
- Markdown files
- PDFs
- CSV files
- Web pages

Concepts:

- Source-specific loaders
- Shared document format
- Metadata by source type
- Filtering retrieval by source

Expected pipeline:

```text
many source loaders -> common documents -> embeddings -> vector DB
```

### Milestone 9: Retrieval Strategy

Status: Not started

Improve retrieval quality.

Concepts:

- Top K tuning
- Metadata filters
- Chunk size tuning
- Chunk overlap tuning
- Similarity method choice
- Keyword search / BM25
- Hybrid search
- Reranking

Key ideas:

- BM25 is lexical search, not vector similarity.
- Hybrid search combines semantic vector search with keyword search.
- These methods are usually used to improve retrieval quality after the basic vector pipeline works.

Expected result:

```text
better retrieved context before answer generation
```

### Milestone 10: Evaluation

Status: Not started

Measure whether the RAG system is working.

Concepts:

- Create test questions.
- Define expected source chunks.
- Check retrieval quality.
- Check answer quality.
- Track failure cases.

Expected result:

```text
repeatable tests for retrieval and answers
```

## GitHub Checkpoints

Push after meaningful working milestones, not after every tiny edit.

Suggested checkpoint examples:

```text
Build text ingestion pipeline
Add fake embeddings to documents
Add top K document retrieval
Clean up retrieval pipeline
Add real sentence-transformer embeddings
Compare similarity methods
Add ChromaDB vector store
Add PDF ingestion
Add answer generation with citations
```

## Current Focus

We are currently learning:

```text
prompt construction
```

Next implementation step:

```text
Convert retrieved chunks into a context block for an LLM prompt.
```
