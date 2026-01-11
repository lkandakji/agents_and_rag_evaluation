# Evaluation Techniques for Agent and RAG pipelines

This repo implements a suite of systematic evaluation techniques for agentic systems and RAG pipelines. Each notebook applies an evaluation approach to test models’ intermediate reasoning and failure behavior, not just final output accuracy. Techniques include trajectory evaluation, dynamic ground truth tests, and simulation benchmarks designed to reveal weak decision patterns.

*   **Unstructured Outputs:** How do you score a free-form text answer that can be phrased in many correct ways?
*   **Multi-Step Reasoning:** How do you evaluate an agent's decision-making process rather than just its final answer?
*   **Dynamic Data:** How do you test a system whose "correct" answers change over time?
*   **Subjective Quality:** How do you measure qualitative aspects like "helpfulness," "conciseness," or "faithfulness" to a source?

| #  | Technique                         | What it Evaluates                                                                                                                | Targeted Failure Mode                                                                                                                          | Notebook                                                           |
| -- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 1  | **Exact Match**                   | Scores a response as correct only if it exactly matches the reference answer. Best suited for deterministic, fact-based queries. | **Surface correctness masking semantic errors** — fails to catch paraphrasing, partial correctness, or subtly wrong answers that look similar. | [`01_exact_match.ipynb`](./01_exact_match.ipynb)                   |
| 2  | **LLM-as-Judge**                  | Uses a stronger LLM to semantically evaluate responses against a reference answer.                                               | **Semantic drift** — detects answers that are fluent but meaningfully incorrect or incomplete despite lexical similarity.                      | [`02_LLM_as_judge.ipynb`](./02_LLM_as_judge.ipynb)                 |
| 3  | **Structured Data Validation**    | Evaluates structured outputs (e.g. JSON) using edit distance metrics that ignore key ordering.                                   | **Silent structural corruption** — catches malformed or partially correct outputs that downstream systems may still accept.                    | [`03_Structured_data.ipynb`](./03_Structured_data.ipynb)           |
| 4  | **Dynamic Ground Truth**          | Evaluates against live, executable ground truth rather than static labels.                                                       | **Stale knowledge / temporal inconsistency** — exposes failures caused by outdated information or changing external state.                     | [`04_dynamic_ground_truth.ipynb`](./04_dynamic_ground_truth.ipynb) |
| 5  | **Trajectory Evaluation**         | Compares an agent’s intermediate steps (tool calls) against an expected reasoning path.                                          | **Right answer for the wrong reason** — detects shortcutting, hallucinated reasoning, or policy violations masked by correct outputs.          | [`05_trajectory.ipynb`](./05_trajectory.ipynb)                     |
| 6  | **Tool Precision & Improvement**  | Measures whether agents select appropriate tools and iteratively improves tool descriptions based on observed failures.          | **Misuse or overuse of tools** — identifies agents that rely on incorrect tools or compensate for poor tool selection with fluent outputs.     | [`06_tool_precision.ipynb`](./06_tool_precision.ipynb)             |
| 7  | **Component-wise RAG Evaluation** | Evaluates the generator independently by fixing retrieved documents.                                                             | **Unfaithful generation** — detects hallucination or misinterpretation even when retrieval is correct.                                         | [`07_component_wise_RAG.ipynb`](./07_component_wise_RAG.ipynb)     |
| 8  | **RAGAS Framework**               | Holistic RAG evaluation across faithfulness, context relevance, recall, and correctness.                                         | **Error attribution ambiguity** — helps distinguish retrieval failures from generation failures when end-to-end performance degrades.          | [`08_RAGAS.ipynb`](./08_RAGAS.ipynb)                               |
| 9  | **Real-time Automated Feedback**  | Attaches reference-free evaluators as callbacks during execution.                                                                | **Undetected degradation in production** — surfaces gradual shifts in quality that do not trigger explicit errors.                             | [`09_realtime_feedback.ipynb`](./09_realtime_feedback.ipynb)       |
| 10 | **Pairwise Comparison**           | Uses an LLM judge to compare two system outputs directly.                                                                        | **Metric insensitivity** — reveals qualitative regressions when absolute scores appear similar or saturated.                                   | [`10_pairwise_comparison.ipynb`](./10_pairwise_comparison.ipynb)   |
| 11 | **Simulation-based Benchmarking** | Evaluates systems through multi-turn conversations with simulated users.                                                         | **Compounding errors over time** — exposes failures that only emerge across longer interactions or stateful dialogues.                         | [`11_simulation.ipynb`](./11_simulation.ipynb)                     |
| 12 | **Algorithmic Feedback Pipeline** | Programmatically scores completed runs for monitoring and dataset curation.                                                      | **Feedback sparsity** — addresses lack of structured signals for large-scale analysis and long-term drift detection.                           | [`12_algorithmic_feedback.ipynb`](./12_algorithmic_feedback.ipynb) |

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
