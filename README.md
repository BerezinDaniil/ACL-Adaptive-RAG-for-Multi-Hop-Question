This repository implements an **adaptive retrieval‑augmented generation (RAG)** framework for multi‑hop question answering, as described in the paper *"Controlling Hallucinations in Multi‑Hop Adaptive RAG"*. The key idea is to use **entropy‑based uncertainty estimation** over tool‑call tokens to decide, at each sub‑question, whether to stop retrieval or continue searching. The code supports both **adaptive** and **naive (single‑step)** retrieval strategies, and includes scripts for validation (threshold tuning) and testing on the MuSiQue dataset.

### File Descriptions

| File / Folder | Description |
|---------------|-------------|
| `qwen_3_5_validation.ipynb` | Scripts to compute **entropy scores** on a validation set for open‑source models (Qwen3.5‑4B, Qwen3.5‑9B, Llama 3.1 8B, Gemma 4B). Outputs raw entropy values for each sub‑question. |
| `find_optimal_threshold.ipynb` | Code to **calculate per‑model entropy thresholds** using Youden’s index. Reads the entropy scores from the validation set and determines the optimal $\tau^*$ that maximizes TPR – FPR. |
| `calculation_metrics_comparison.ipynb` | Scripts that compute **evaluation metrics (accuracy)** and directly compare the adaptive RAG against the naive RAG baseline. Produces tables and figures used in the paper. |
| `test_qwen_adaptive.ipynb` | Scripts to run the **adaptive retrieval** strategy on the test set for each open‑source model. Uses the pre‑computed thresholds to decide dynamically whether to retrieve more documents or stop. |
| `Adaptive_RAG.ipynb` | Core implementation of the **adaptive approach** specifically for Qwen3.5‑4B and Qwen3.5‑9B on the test set. Includes the entropy‑based stopping mechanism and evaluation of TPR / FPR. |
| `naive_RAG.ipynb` | Implementation of the **naive (single‑step) baseline** for Qwen3.5‑4B and Qwen3.5‑9B. Retrieves top‑30 documents based on the main question (no decomposition, no iterative search). |
| `tpr_fpr_calculate.ipynb` | Notebook for computing **True Positive Rate (TPR)** and **False Positive Rate (FPR)** for models using the adaptive approach with entropy‑based uncertainty estimation. |

### Main Functionality

- **Adaptive RAG**  
  - Decomposes each multi‑hop question into sub‑questions (using MuSiQue annotations).  
  - For each sub‑question, retrieves $k=5$ paragraphs and asks the LLM whether to stop (select a supporting paragraph) or retrieve more.  
  - The stop/retrieve decision is based on **entropy of the tool‑call tokens** – only when entropy < threshold $\tau$ does the model stop.  
  - Threshold $\tau$ is calibrated per model on the validation set (`find_optimal_threshold`).  
  - Performance is compared against naive RAG using `calculation_metrics_comparison`.

- **Naive RAG**  
  - No sub‑question decomposition.  
  - Single retrieval step (top‑30 documents) based on the full question.  
  - Evaluated for comparison.

### Key Results (reported in the paper)

- Adaptive retrieval **reduces false retriever calls (FPR)** by up to 3–5× compared to greedy decoding.  
- End‑to‑end multi‑hop accuracy improves over naive retrieval (e.g., Qwen 9B: 0.38 → 0.45).  
- Gains hold even with a weak retriever (BM25), showing that **uncertainty‑aware control** matters more than absolute retriever quality.

### Requirements

- Python 3.8+  
- Transformers, PyTorch, Sentence‑Transformers, OpenRouter API (for API‑based models)  
- MuSiQue dataset
