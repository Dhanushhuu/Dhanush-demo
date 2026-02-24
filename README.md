# 🤖 CVIP RAG System
> A Production-Ready Retrieval-Augmented Generation System for Computer Vision & Image Processing

[![Databricks](https://img.shields.io/badge/Powered%20by-Databricks-red)](https://databricks.com)
[![LLaMA](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-blue)](https://ai.meta.com)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)

---

## 📋 Overview

The CVIP RAG System is a production-grade question-answering system built on Databricks, designed to answer questions about **Computer Vision and Image Processing** by retrieving relevant content from a curated knowledge base of textbooks, research papers, and surveys.

### Key Capabilities
- ✅ **Technical CVIP questions** with cited answers and page numbers
- ✅ **General knowledge questions** answered via LLM
- ✅ **Memory recall** — ask about any previous question naturally
- ✅ **Smart query routing** — CVIP questions with citations, general knowledge answered gracefully using LLM
- ✅ **Session analytics** with detailed performance metrics

---

## 🏗️ System Architecture
```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Classifier  │  ──► Intent Detection (foundational/comparison/advanced)
│   + Memory Detector │  ──► Domain Relevance Check
└─────────────────────┘
    │
    ├──► Memory Query    ──► ProductionMemorySystem (instant, 0ms)
    │
    ├──► CVIP Domain     ──► Vector Search Retrieval
    │                            │
    │                        Tier-Aware Scoring
    │                            │
    │                        LLaMA 3.3 70B Generation
    │                            │
    │                        Grounding Verification
    │
    └──► General Query   ──► LLM Direct (cached, ~400ms)
```

---

## 🗃️ Knowledge Base

| Source Type | Description | Trust Tier |
|-------------|-------------|------------|
| 📘 Textbooks | Gonzalez & Woods DIP 4th Ed, Szeliski CV 2nd Ed | Tier 1 (Highest) |
| 🧪 Research Papers | ViT, Deep CNN, ResNet, YOLO papers | Tier 2 |
| 📄 Surveys | Deep learning surveys, CV application surveys | Tier 3 |

- **Total Chunks**: 10,097
- **Embedding Model**: BGE-Large
- **Vector Index**: Databricks Vector Search
- **Chunk Size**: ~500 tokens with overlap

---

## ⚙️ Core Components

### 1. `SecureConfig`
Manages all system configuration — workspace URLs, endpoint names, scoring weights, and thresholds. Fails fast if required environment variables are missing.

### 2. `SafeMetadataFetcher`
Preloads all 10,097 chunk metadata records into memory at startup. Eliminates per-query Spark calls — metadata lookups are instant Python dict operations.

### 3. `QueryClassifier`
Classifies queries by:
- **Domain relevance** — 11 keyword categories covering core CVIP topics
- **Intent** — foundational, advanced, comparison, implementation

### 4. `MemoryQueryDetector`
Detects memory recall queries using regex patterns. Supports ordinal references (first through tenth), list requests, and conversational memory queries.

### 5. `ProductionMemorySystem`
Sliding window memory with full session log. Supports ordinal recall, session summaries, and salience-based turn scoring.

### 6. `FixedAnswerGenerator`
Calls LLaMA 3.3 70B via Databricks serving endpoint using direct HTTP requests. Generates textbook-quality answers with mathematical notation, worked examples, and source citations.

### 7. `ImprovedGroundingChecker`
Verifies answer grounding using three metrics:
- **Overlap score** — term overlap between answer and retrieved chunks
- **Citation score** — citation density relative to answer length  
- **Alignment score** — sentence-level alignment with source content

### 8. `FinalQueryLogger`
Logs all queries to a Delta table (`cvip_query_logs`) with support level, confidence, latency, and error tracking. Batched writes with smart flush strategy.

### 9. `SmartFlushManager`
Flushes query logs every N queries OR every T seconds — whichever comes first. Uses RLock to prevent deadlocks.

### 10. `FinalProductionRAG`
Main controller orchestrating all components. Handles session management, query routing, error recovery, and system statistics.

---

## 🔄 Query Processing Pipeline
```
1. Query received
2. QueryClassifier → intent + domain relevance
3. MemoryQueryDetector → is memory recall?
   YES → ProductionMemorySystem.recall() → instant response
   NO  →
4. Is domain relevant?
   YES → Vector Search (top-K retrieval)
       → Tier-aware weighted scoring (α×similarity + β×priority)
       → LLaMA 3.3 70B generation with system prompt
       → Grounding verification → confidence score
   NO  → Is CVIP-adjacent? → route to domain handler
       → Is trivial/lifestyle? → reject
       → Otherwise → LLM general knowledge (cached)
5. Session memory updated
6. Query logged to Delta table
7. Response returned with citations + metrics
```

---

## 📊 Retrieval Scoring

Chunks are scored using a weighted combination:
```
weighted_score = α × similarity_score + β × priority_score
```

Weights vary by query intent:

| Intent | α (Similarity) | β (Priority) |
|--------|---------------|--------------|
| Foundational | 0.6 | 0.4 |
| Advanced | 0.8 | 0.2 |
| Comparison | 0.7 | 0.3 |
| Implementation | 0.7 | 0.3 |

---

## 🚀 Getting Started

### Prerequisites
- Databricks workspace (AWS/Azure/GCP)
- Vector Search endpoint configured
- LLaMA 3.3 70B serving endpoint active

### Environment Setup
```python
import os
os.environ["DATABRICKS_HOST"] = "https://your-workspace.cloud.databricks.com"
```

### Initialize System
```python
rag = FinalProductionRAG(
    enable_reranking=False,
    enable_persistence=True,
    flush_every_n=3,
    flush_every_seconds=30
)
```

### Ask Questions
```python
# CVIP question with citations
response = rag.ask("What is edge detection?", session_id="demo")

# Memory recall
response = rag.ask("What was my first question?", session_id="demo")

# General knowledge
response = rag.ask("What is the capital of Andhra Pradesh?", session_id="demo")
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Knowledge Base | 10,097 chunks |
| Avg Domain Query Latency | ~4-6 seconds |
| Avg General Query Latency | ~400ms (cached: 0ms) |
| Memory Recall Latency | <5ms |
| Metadata Preload Time | ~2.5 minutes (one-time) |
| Supported Ordinals | first through tenth |

---

## 🗂️ Repository Structure
```
├── rag_components.py     # Complete RAG system (all components)
├── README.md             # This file
└── app.yaml              # Databricks Apps deployment config
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Platform | Databricks (Serverless) |
| Vector Database | Databricks Vector Search |
| Embedding Model | BGE-Large |
| LLM | LLaMA 3.3 70B Instruct |
| Storage | Delta Lake |
| Language | Python 3.10+ |
| Memory | In-process sliding window |

---

## 👨‍💻 Author

**Dhanush Kumar**  
Final Year Project — Computer Vision & Image Processing  
2026-2027

---

## 📚 Knowledge Sources

- Gonzalez & Woods, *Digital Image Processing*, 4th Edition
- Szeliski, *Computer Vision: Algorithms and Applications*, 2nd Edition  
- Rawat & Wang, *Deep Convolutional Neural Networks for Image Classification*
- Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*
- Various CVIP surveys and research papers
