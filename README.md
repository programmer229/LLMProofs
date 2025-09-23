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

### ⚡ High-Performance Training
- **Distributed Training**: Multi-GPU support with FSDP and Megatron-LM backends
- **Efficient Inference**: Integration with vLLM and TGI for fast rollout generation
- **Flexible Architecture**: Supports various model sizes from 3B to 70B+ parameters
- **Resource Optimization**: Advanced memory management and gradient checkpointing

### 🔧 Advanced Technical Features
- **LaTeX Parsing**: Improved mathematical notation processing and validation for proofs
- **Batch Processing**: Efficient batch evaluation of mathematical proof solutions
- **Multi-Model Support**: Compatible with various transformer architectures for proof generation
- **Experiment Tracking**: Integration with Weights & Biases and MLflow for proof training metrics

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Actor Model   │    │  Judge Model    │    │   Reward Model  │
│                 │    │                 │    │                 │
│ Generates       │───▶│ Evaluate        │◀───│ Calculates      │
│ Mathematical    │    │ Solution        │    │ Loss            │
│ Proofs          │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                               |
         │                                               │
         │              ┌─────────────────┐              │
         │              │  PPO Trainer    │              │
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
   cd LLMProofs
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

### Basic Usage

#### 1. Prepare Mathematical Datasets

Generate training data for integration problems:
```bash
python data/create_ladder_sympy.py
```

Create competition math datasets:
```bash
python data/create_competition_math.py
```

#### 2. Train a Mathematical Proof Model

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

## 📚 Supported Mathematical Proof Domains

### Integration and Calculus Proofs
- Symbolic integration proof problems
- Definite and indefinite integral proofs
- Complex mathematical expression derivations
- SymPy validation for proof correctness

### Competition Mathematics Proofs
- **AIME**: American Invitational Mathematics Examination proof problems
- **IMO**: International Mathematical Olympiad proof challenges
- **Competition Math**: Various mathematical olympiad proof problems

### Custom Proof Domains
The framework is extensible for new mathematical proof domains:
```python
# Add custom proof evaluation function
def compute_score(solutions_batch, ground_truth_batch, **kwargs):
    # Your custom proof evaluation logic
    return reward_tensor
```

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

### Custom Proof Judge Models

Create custom mathematical proof evaluators:

1. **Implement proof judge function**:
   ```python
   # In verl/utils/reward_score/
   def compute_score(solutions_batch, ground_truth_batch, ...):
       # Custom proof evaluation logic
       return reward_tensor
   ```

2. **Register in main trainer**:
   ```python
   # Add to _select_rm_score_fn() in main_ppo.py
   elif "your_proof_domain" in data_source:
       return your_custom_proof_judge.compute_score
   ```

### Multi-GPU Training

For large-scale training:
```bash
# Configure distributed training
trainer.n_gpus_per_node=8
trainer.nnodes=2
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

## 📁 Project Structure

```
LLMProofs/
├── data/                          # Dataset creation and management
│   ├── create_*.py               # Dataset generation scripts
│   ├── integration_3b_llmjudge/  # Integration problem datasets
│   ├── ladder_variants/          # Mathematical problem variants
│   └── TextbookQuestions/        # Educational content
├── verl/                         # Core RL training framework
│   ├── trainer/                  # Training orchestration
│   ├── workers/                  # Distributed workers
│   ├── utils/reward_score/       # Evaluation functions
│   └── models/                   # Model implementations
├── scripts/                      # Training scripts
├── examples/                     # Usage examples
├── docs/                        # Documentation
└── helper_scripts/              # Utility tools
```

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

- **New Mathematical Proof Domains**: Expand support for additional mathematical proof areas
- **Improved Proof Judges**: Develop more sophisticated proof evaluation methods
- **Optimization**: Performance improvements for proof training and inference
- **Documentation**: Examples and tutorials for new users working with mathematical proofs

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the [veRL framework](https://github.com/volcengine/verl) by ByteDance
- Inspired by mathematical proof research and competition mathematics
- Thanks to the open-source ML community for foundational tools

---

*For detailed documentation, advanced configuration options, and API references, please refer to the `/docs` directory.*
