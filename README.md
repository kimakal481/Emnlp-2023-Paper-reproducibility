README: Reproducing Citation Evaluation with Mistral-7B and ALCE Benchmark
This guide provides a step-by-step walkthrough to set up, run, and evaluate the ALCE benchmark using Mistral-7B-v0.1 for citation generation.

1. System Requirements
Ensure you have the following resources available:

Hardware:
GPU: A high-memory GPU (at least 24GB VRAM recommended)
CPU: 8+ core processor
RAM: 32GB+
Storage: 50GB+ free disk space
Software:
OS: Ubuntu 20.04+ / macOS / Windows WSL
Python: 3.9+
CUDA: 11.8+ (if using NVIDIA GPU)
PyTorch: 2.0+ with GPU support
2. Installation and Setup
Step 1: Clone the ALCE Repository
Open a terminal and run:

bash
Copy
Edit
git clone https://github.com/princeton-nlp/ALCE.git
cd ALCE
Step 2: Install Required Dependencies
Install necessary Python libraries using:

bash
Copy
Edit
pip install torch transformers accelerate datasets sentencepiece evaluate rouge-score
Step 3: Download the Dataset
Run the script to fetch the ALCE datasets:

bash
Copy
Edit
bash download_data.sh
3. Model Setup
Step 4: Load Mistral-7B-v0.1 Model
Modify run.py to use Mistral-7B-v0.1 instead of OpenAI’s API-based models.

In Python, initialize the model:

python
Copy
Edit
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
    device_map="auto",  # Automatically distribute across GPUs
    offload_state_dict=True  # Enables offloading to CPU if needed
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
4. Running the Experiment
Step 5: Execute the Model with ASQA Dataset
Modify the run.py configuration to work with Mistral-7B:

bash
Copy
Edit
python run.py --config configs/asqa_turbo_shot2_ndoc5_gtr_default.yaml --model mistralai/Mistral-7B-v0.1
This step runs Mistral-7B-v0.1 on the ASQA dataset.

5. Evaluating Results
Step 6: Evaluate Citation Accuracy, Fluency, and Relevance
Once the model has generated results, evaluate them using:

bash
Copy
Edit
python eval.py --f result/asqa-Mistral-7B-v0.1-gtr-shot2-ndoc5-42.json --citations --qa --mauve
This script will compute citation recall, precision, fluency, and correctness.

6. Key Considerations
Performance Constraints: Mistral-7B-v0.1 requires a powerful GPU; running on a CPU is extremely slow.
Memory Management: If you face out-of-memory (OOM) issues, try gradient checkpointing or reduce batch size.
Longer Execution Time: Running experiments on the free-tier GPU increases runtime significantly.
Error Handling: If the model crashes due to token limits, modify run.py to truncate input size dynamically.
7. Results Overview
Metrics from ASQA Dataset Run
Metric	Score
Fluency (MAUVE)	46.92
Exact Match (QA-EM)	16.56
F1 Score (QA-F1)	21.91
Citation Recall	17.84
Citation Precision	20.38
These results indicate that Mistral-7B-v0.1 produces weaker citations than GPT-4 but still demonstrates improvement when tuned with better parameters.

8. Troubleshooting
Common Errors & Fixes
CUDA Out of Memory Error
Reduce batch size in run.py
Enable offloading to CPU
Long Execution Time
Use a better GPU
Reduce dataset size for testing
Token Truncation Issues
Modify run.py to adjust max token size
9. Conclusion
This guide provides a reproducible pipeline to run Mistral-7B-v0.1 for citation-based text generation using the ALCE benchmark. While results are lower than OpenAI’s models, optimizations in retrieval strategy and token management can significantly enhance performance.

For further improvements:

Experiment with different RAG techniques
Use better re-ranking models
Try fine-tuning Mistral-7B on citation-rich datasets
