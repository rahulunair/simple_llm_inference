# Simple LLM Inference: Interactive Text Generation with Intel GPUs

LLM Inference example is a Python script designed to demonstrate interactive text generation using pre-trained Language Models (LLMs) from Hugging Face Transformers and Intel dGPUs. Although the code structure resembles a chatbot, it's important to note that the models used were not specifically trained for conversational purposes. This repository provides a Command Line Interface (CLI) to interact with the models in two modes: with context (remembering previous interactions) and without context. The code is optimized to run on Intel GPUs using Intel Extension for PyTorch (IPEX).

# Features

- **Model Selection**: Choose between predefined models or enter a custom model repository from Hugging Face Hub.
- **Context Control**: Interact with the model in two modes, with and without context.
- **Generation Parameters Control**: Customize the response generation by adjusting parameters like temperature, top_p, top_k, num_beams, and repetition_penalty.
- **Repetition Removal**: The code includes logic to remove repetitive sentences from the generated text.

# Prerequisites

- Python 3.6 or higher
- Intel Extension for PyTorch (IPEX)
- Hugging Face Transformers
- Hugging Face Accelerate

# Setup

1. Clone the repository

```bash
git clone https://github.com/rahulunair/simple_llm_inference.git
```

2. Install dependencies:

```bash
python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
python -m pip install transformers accelerate
```

To know more about IPEX on Intel GPUs use the following [guide](https://github.com/intel/intel-extension-for-pytorch#installation).

# Usage

1. Run `bot.py`

```bash
python bot.py
```

2. Follow the on-screen prompts to select a model and the interaction mode.

# Sample output

```bash
Please select a model:
1. Writer/camel-5b-hf
2. openlm-research/open_llama_3b_v2
3. Enter a custom model repo from HuggingFace Hub
Enter 1 to 3: 2
Using max length: 256
Note: This is a demonstration using pretrained models which were not fine-tuned for chat.
You can choose between two modes of interaction:
1. Interact with context
2. Interact without context
Enter 1 or 2: 1
You: Hello, Bot!
Bot: Hello! How can I assist you today?
```

# Contributing

Feel free to submit issues and pull requests. Contributions are welcome!
