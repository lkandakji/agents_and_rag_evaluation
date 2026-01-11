# Evaluation Techniques for Agent and RAG pipelines

This repo provides a guide to 12 different techniques for evaluating AI Agents and Retrieval-Augmented Generation (RAG) systems. Each technique is implemented in a runnable Jupyter Notebook, with practical application on industry-standard tools like LangChain and LangSmith. Key challenges targeted include:

*   **Unstructured Outputs:** How do you score a free-form text answer that can be phrased in many correct ways?
*   **Multi-Step Reasoning:** How do you evaluate an agent's decision-making process rather than just its final answer?
*   **Dynamic Data:** How do you test a system whose "correct" answers change over time?
*   **Subjective Quality:** How do you measure qualitative aspects like "helpfulness," "conciseness," or "faithfulness" to a source?

## Table of Evaluation Techniques

| #  | Technique                                 | Description                                                                                                                              | Notebook                                                                      |
|----|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| 1  | **Exact Match**                           | The simplest method. Scores a response as correct (1) only if it is an identical string to the reference answer, and incorrect (0) otherwise. Ideal for fact-based Q&A.                                               | [`01_exact_match.ipynb`](./01_exact_match.ipynb)                               |
| 2  | **LLM-as-Judge**                          | Uses a powerful LLM to grade a Q&A system's response for correctness against a reference answer. Provides a more nuanced, semantic evaluation than exact match.                                                 | [`02_LLM_as_judge.ipynb`](./02_LLM_as_judge.ipynb)                             |
| 3  | **Structured Data Validation**            | Evaluates extraction chains that output structured data (like JSON). Uses metrics like `json_edit_distance` to measure structural and content similarity, ignoring key order.                               | [`03_Structured_data.ipynb`](./03_Structured_data.ipynb)                       |
| 4  | **Dynamic Ground Truth**                  | Tests systems on live, changing data sources. Stores executable code as the "ground truth" which is run at evaluation time to get the live correct answer.                                                       | [`04_dynamic_ground_truth.ipynb`](./04_dynamic_ground_truth.ipynb)             |
| 5  | **Trajectory Evaluation**                 | Evaluates an agent's intermediate steps (tool calls) against an expected sequence. Ensures the agent is not just correct, but correct for the right reasons.                                                     | [`05_trajectory.ipynb`](./05_trajectory.ipynb)                                 |
| 6  | **Tool Precision & Improvement**          | Measures an agent's ability to select the correct tools. Demonstrates a full loop of evaluating, identifying failures, and using an LLM to automatically improve tool descriptions.                             | [`06_tool_precision.ipynb`](./06_tool_precision.ipynb)                         |
| 7  | **Component-wise RAG Evaluation**         | Isolates and evaluates the response generator of a RAG system. By providing fixed source documents, it tests for faithfulness and correctness independent of retriever performance.                             | [`07_component_wise_RAG.ipynb`](./07_component_wise_RAG.ipynb)                 |
| 8  | **RAGAS Framework**                       | Uses the RAGAS framework for a holistic RAG pipeline evaluation, scoring faithfulness, context relevance, context recall, and answer correctness.                                                              | [`08_RAGAS.ipynb`](./08_RAGAS.ipynb)                                           |
| 9  | **Real-time Automated Feedback**          | Attaches reference-free evaluators (e.g., for "helpfulness") as callbacks to a chain. This allows for automated, real-time monitoring of applications in production.                                         | [`09_realtime_feedback.ipynb`](./09_realtime_feedback.ipynb)                   |
| 10 | **Pairwise Comparison**                   | Uses an LLM judge to determine which of two system outputs is better. This reveals subtle qualitative differences that absolute grading can miss, especially when aggregate scores are close.                 | [`10_pairwise_comparison.ipynb`](./10_pairwise_comparison.ipynb)               |
| 11 | **Simulation-based Benchmarking**         | Evaluates a chatbot by simulating conversations. An LLM-powered "user" agent interacts with the chatbot to test its multi-turn conversational abilities in a reproducible way.                                | [`11_simulation.ipynb`](./11_simulation.ipynb)                                 |
| 12 | **Algorithmic Feedback Pipeline**         | Programmatically adds quality scores to completed production runs in batches. This is ideal for scheduled jobs that enrich traces for monitoring, analysis, and dataset curation.                                 | [`12_algorithmic_feedback.ipynb`](./12_algorithmic_feedback.ipynb)             |

## 🚀 Getting Started

Follow these steps to run the notebooks locally and experiment with the evaluation techniques.

### 1. Clone the Repository

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required packages with the following command:

```bash
pip install langchain langchain_openai langchain_experimental langsmith langgraph openai anthropic pandas chromadb lxml html2text jsonschema ragas numpy textstat requests
```

### 3. Set Environment Variables

These notebooks require API keys for various services. The most secure way to manage these is with a `.env` file.

Create a file named `.env` in the root of the project and add the following, replacing the placeholder values with your actual keys:

```
# LangSmith Credentials (Required for all notebooks)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY"

# Model Provider Credentials (Required for most notebooks)
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"

# LangChain Hub Credentials (Optional, for specific examples)
LANGCHAIN_HUB_API_URL="https://api.hub.langchain.com"
LANGCHAIN_HUB_API_KEY="YOUR_LANGCHAIN_HUB_API_KEY"
```

### 4. Run the Notebooks

```bash
jupyter lab
```

## 🛠️ Core Technologies Used

*   [**LangSmith**](https://smith.langchain.com/): The central platform for logging traces, creating datasets, running evaluators, and monitoring the performance of LLM applications.
*   [**LangChain**](https://python.langchain.com/): The core framework used to build the AI agents and RAG pipelines that are being evaluated.
*   [**LangGraph**](https://langchain-ai.github.io/langgraph/): Used in the simulation example to create stateful, multi-actor applications.
*   **OpenAI & Anthropic Models**: The primary LLMs used as reasoning engines for the agents and as judges for evaluation.
*   **RAGAS**: A specialized, open-source framework for in-depth RAG evaluation.
