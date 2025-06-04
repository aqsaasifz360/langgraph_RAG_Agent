
# LangGraph QA System using Vertex AI

This project is a question-answering system built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Vertex AI](https://cloud.google.com/vertex-ai) from Google Cloud. It processes Lilian Weng's blog posts and allows semantic question answering using vector search and Gemini models.

## Features

- Scrapes and splits blog post content using LangChain tools.
- Embeds documents with `text-embedding-004` from Vertex AI.
- Stores vectors in memory for fast retrieval.
- Uses Gemini 2.0 Flash for:
  - Response generation
  - Relevance grading
  - Question rewriting
- Dynamically routes through a LangGraph based on document relevance.

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with the following:

```env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

Ensure that Vertex AI and Gemini are enabled in your Google Cloud project.

## Usage

```bash
python graph.py
```

This script loads documents, splits and embeds them, builds a retriever, and runs a LangGraph that:

1. Generates an initial response or query
2. Retrieves documents if needed
3. Grades their relevance
4. Either answers or rewrites the question

## Document Sources

- [Reward Hacking (Nov 2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [Hallucination (Jul 2024)](https://lilianweng.github.io/posts/2024-07-07-hallucination/)
- [Diffusion Video (Apr 2024)](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
