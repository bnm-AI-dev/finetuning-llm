# INKUBALM: Fine-tuning Africa's First Small Language Model

This repository contains a Jupyter notebook demonstrating the fine-tuning of Lelapa AI's InkubaLM model using the LORA (Low-Rank Adaptation) technique. The project is part of the Lelapa AI Buzuzu-Mavi Challenge, which aims to make the SLM (Small Language Model) smaller and smarter.

## Project Overview

The notebook implements fine-tuning for three specific tasks:

1. Sentiment Analysis
2. Natural Language Inference (NLI)
3. Machine Translation (MMT)

The model is trained on Swahili and Hausa languages, focusing on improving performance across these tasks while maintaining model efficiency.

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Accelerate
- Bitsandbytes
- Pandas
- NumPy
- Hugging Face account with access to the InkubaLM model

## Setup

```bash
pip install -U peft bitsandbytes accelerate transformers torch pandas numpy
```

You'll need to set up your Hugging Face token for accessing the model:

```python
from huggingface_hub import login
login("your_token_here")
```

## Model Architecture

The notebook uses LORA adaptation with the following configuration:

- Base Model: lelapa/InkubaLM-0.4B
- LORA rank (r): 32
- LORA alpha: 32
- Target modules: ["q_proj", "v_proj"]
- Dropout: 0.05

## Training Process

The notebook implements:

1. Custom dataset handling through `InstructionDataset` class
2. Task-specific prompt templates
3. Separate LORA adapters for each task
4. Training configuration with:
   - Batch size: 3
   - Gradient accumulation steps: 2
   - Learning rate: 2e-4
   - Number of epochs: 20
   - Mixed precision (FP16)

## Evaluation

The notebook includes functionality for:

- Loading and processing test data
- Generating responses using task-specific prompts
- Mapping responses to standardized labels for:
  - Hausa sentiment (kyakkyawa/tsaka/korau)
  - Swahili sentiment (chanya/wastani/hasi)
  - XNLI (true/neutral/false)

## File Structure

```
.
├── LORAfinetuning.ipynb    # Main notebook with implementation
└── README.md               # This documentation
```

## Usage

1. Open the notebook in a GPU-enabled environment
2. Set up your Hugging Face credentials
3. Run the cells in sequence
4. The notebook will:
   - Load and prepare the data
   - Train separate LORA adapters for each task
   - Save the adapters
   - Evaluate on test data

## Acknowledgments

This project is part of the Lelapa AI Buzuzu-Mavi Challenge and uses the InkubaLM model. The implementation builds upon the PEFT library and Hugging Face's Transformers.

## License

Please refer to Lelapa AI's licensing terms for the base model usage.
