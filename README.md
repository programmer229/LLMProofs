# LLMProofs: Reinforcement Learning for Mathematical Proof Generation

**A specialized RL training framework for teaching language models to generate and self-evaluate mathematical proofs**

LLMProofs is a customized version of the [veRL framework](https://github.com/volcengine/verl) specifically designed for training language models on mathematical proof generation tasks. The system implements a sophisticated agent-judge paradigm where an AI agent generates mathematical proofs and specialized judge models assess their correctness.

## 🎯 Overview

This project focuses on advancing mathematical reasoning capabilities in large language models through:

- **Self-Marking Proofs**: An agent produces mathematical proofs and a judge assesses their correctness automatically
- **LLM-based Evaluation**: Advanced LLM judges that can evaluate mathematical rigor and correctness at IMO (International Mathematical Olympiad) level
- **Reinforcement Learning Training**: Uses PPO (Proximal Policy Optimization) and GRPO to improve proof generation through reward feedback
- **Mathematical Proof Focus**: Specializes in formal mathematical proofs across integration, algebra, geometry, and olympiad-level problems

## ✨ Key Features

### 🤖 Intelligent Mathematical Evaluation
- **LLM Judge System**: Sophisticated judges that evaluate mathematical proofs for correctness, rigor, and completeness
- **Multi-Criteria Scoring**: Evaluates proofs based on logical flow, mathematical accuracy, and step-by-step reasoning
- **Rubric-Based Assessment**: Follows structured marking criteria similar to mathematical olympiad grading

### 📊 Mathematical Proof Datasets
- **Integration Problems**: Symbolic integration proofs with SymPy validation
- **Competition Mathematics**: AIME, IMO, and other mathematical olympiad proof problems
- **Custom Proof Sets**: Extensible framework for adding new mathematical proof domains


## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Actor Model   │    │  Judge Model    │    │   Reward Model  │
│                 │    │                 │    │                 │
│ Generates       │───▶│ Evaluate        │-──▶│ Calculates      │
│ Mathematical    │    │ Solution        │    │ Loss            │
│ Proofs          │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                               |
         │                                               │
         │              ┌─────────────────┐              │
         │              │  GRPO Trainer   │              │
         └──────────────│                 │◀─────────────┘
                        │ Optimizes       │
                        │ Generation      │
                        │ Policy          │
                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPUs (recommended: 24GB+ VRAM)
- Ray framework for distributed training

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLMProofs/train
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys** (for external LLM judges):
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export OPENROUTER_API_KEY="your-openrouter-key"
   ```
Run GRPO training on proof datasets:
```bash
bash scripts/proofs_7b_grpo.sh
```

This script will:
- Load a base model (e.g., DeepSeek-R1-Distill-Qwen-7B)
- Train using mathematical proof datasets
- Evaluate using LLM judges
- Save checkpoints and logs

#### 3. Monitor Training

Training metrics are automatically logged to Weights & Biases:
- Mathematical proof generation quality
- Judge scores and distributions for proof correctness
- Training loss and convergence
- Sample proofs and mathematical evaluations


## 🔬 Advanced Configuration

### Training Parameters

Key configuration options in the training scripts:

```bash
# Model and data configuration
BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DATA_DIR=/path/to/mathematical/datasets

# Training hyperparameters
data.train_batch_size=32
data.max_response_length=8192
actor_rollout_ref.actor.optim.lr=5e-6

# Judge configuration
+judge.model=$BASE_MODEL
+judge.location=local
+judge.gpus=4
```
