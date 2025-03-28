# RAG Chunking Strategies Evaluation

This repository contains code for evaluating different chunking strategies in Retrieval-Augmented Generation (RAG) systems using the RAGAS evaluation framework.

## Overview

The project compares four chunking strategies:
- Fixed-Size Chunking
- Semantic Chunking
- Agentic Chunking
- Recursive Chunking

Each strategy is evaluated using the same document corpus and evaluation metrics to determine their impact on RAG system performance.

## Requirements

```
agno
python-dotenv
qdrant-client
anthropic
ollama

# pdf parser
pypdf

# chunking library
chonkie

# configuration
packaging
importlib-metadata

# evals
ragas
deepeval

# llama-index
llama-index
llama-index-llms-openai
```

## Environment Setup

Create a `.env` file with the following variables:
```
collection_name=your collection
qdrant_url=your qdrant url
api_key=your qdrant api key
ANTHROPIC_API_KEY=sk-ant-
OPENAI_API_KEY=sk-proj-
```

## Usage

1. Place your test document in `data/test_data.pdf`
2. Create ground truth question-answer pairs in `data/ground_truth.json` using the format:
   ```json
   [
     {
       "question": "What is...",
       "answer": "The answer is..."
     }
   ]
   ```
3. Run one of the evaluation scripts:
   ```
   python evaluate_fixed_chunking.py
   python evaluate_semantic_chunking.py
   python evaluate_agentic_chunking.py
   python evaluate_recursive_chunking.py
   ```

## Evaluation Dataset Creation

The `create_eval_ds` function processes ground truth data, submits questions to the RAG agent, and creates a dataset with:
- Original questions
- Reference answers
- Retrieved contexts
- Agent responses

This dataset is used by RAGAS to calculate evaluation metrics.

## Metrics

The evaluation uses five RAGAS metrics:
- Faithfulness: Measures if the generated answer is supported by the retrieved context
- Context Relevance: Evaluates if the retrieved context is relevant to the question
- Context Utilization: Assesses how well the model utilizes the retrieved context
- Context Recall: Measures if the context contains the information needed to answer
- Factual Correctness: Evaluates factual accuracy of the answer

## Results

Evaluation results are output as scores for each metric, which can be compared across different chunking strategies to determine which approach works best for specific document types and use cases.

## Conclusion

Our evaluation demonstrates that document segmentation significantly impacts RAG system performance across all RAGAS metrics. While Fixed-Size chunking provides a baseline, Semantic and Agentic approaches better preserve contextual integrity by respecting natural information boundaries. Notably, Agentic chunking with Claude 3.7 Sonnet showed superior Context Relevance and Factual Correctness by leveraging the LLM's understanding of document structure, while Recursive chunking excelled for hierarchical documents. These findings emphasize that chunking isn't merely a technical detail but a fundamental architectural decision that should be tailored to specific document types and use cases to build more accurate and trustworthy RAG systems.