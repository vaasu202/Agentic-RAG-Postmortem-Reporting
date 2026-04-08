# Agentic RAG for Postmortem Reporting

An intelligent, agent-driven system that analyzes incident postmortems, logs, and runbooks to automatically generate root cause insights and actionable remediation strategies.

## Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** pipeline designed for incident analysis and postmortem reporting.

Instead of a static RAG system, this solution uses an **AI agent that iteratively reasons, retrieves, and validates information** to produce structured outputs such as:
- Root cause hypotheses
- Supporting evidence
- Recommended remediation steps
- Confidence scores

Unlike traditional pipelines, Agentic RAG systems actively decide what to retrieve, when to use tools, and how to refine their reasoning, making them far more robust for complex workflows like incident analysis.

## Problem Statement

Incident postmortems and operational logs are:
- Unstructured
- Distributed across multiple sources
- Difficult to analyze quickly under pressure

Manual analysis is:
- Time-consuming
- Error-prone
- Hard to standardize

This project solves that by building an AI-powered copilot that:
- Searches historical incidents and runbooks
- Extracts meaningful signals from logs
- Suggests root causes and fixes
- Produces structured, explainable reports

## Key Features

- **Agentic RAG Pipeline**
  - Dynamic reasoning loop (retrieve → analyze → refine → respond)
- **Semantic Knowledge Retrieval**
  - Vector search over incident reports and documentation
- **Tool-Augmented Intelligence**
  - Log parsing and signal extraction
  - Root cause synthesis
  - Remediation planning
- **Structured Output (JSON)**
  - Machine-readable insights for downstream systems
- **Self-Correction**
  - Re-queries and refines answers when confidence is low

---

## System Architecture

            ┌──────────────────────┐
            │   User Query / Logs  │
            └─────────┬────────────┘
                      ↓
            ┌──────────────────────┐
            │   Agent Controller   │
            │ (Decision + Reason)  │
            └─────────┬────────────┘
                      ↓
    ┌────────────────────────────────────┐
    │            Tools Layer             │
    │                                    │
    │  • Knowledge Base Search (RAG)      │
    │  • Log Signal Extraction           │
    │  • Root Cause Generator            │
    │  • Remediation Planner             │
    └────────────────────────────────────┘
                      ↓
            ┌──────────────────────┐
            │  Structured Output   │
            │  (JSON Report)       │
            └──────────────────────┘


## Tech Stack

- **Language:** Python  
- **LLM Framework:** LangChain / Custom Agent Orchestration  
- **Vector Database:** ChromaDB / FAISS  
- **Embeddings:** OpenAI / SentenceTransformers  
- **Frontend (optional):** Streamlit / API endpoints  
- **Orchestration:** Tool-based agent workflow  

## Core Components

### 1. Knowledge Base Retrieval
Searches historical:
- Postmortems
- Runbooks
- Incident documentation

### 2. Log Signal Extraction
Parses logs to identify:
- Error codes
- Failure patterns
- Temporal anomalies

### 3. Root Cause Analysis
Generates:
- Hypotheses with evidence
- Probabilistic confidence

### 4. Remediation Generator
Outputs:
- Step-by-step fixes
- Preventative recommendations

## Example Output

```json
{
  "root_causes": [
    {
      "cause": "Database connection pool exhaustion",
      "confidence": 0.87,
      "evidence": ["timeout errors", "high DB latency"]
    }
  ],
  "remediation": [
    "Increase connection pool size",
    "Implement retry logic",
    "Add DB monitoring alerts"
  ]
}
