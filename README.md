# GPT2 (124M)
### Key points
- Completed: The model has been implemented and pretrained using 10B Tokens from fineweb-edu.
- Main reference: https://www.youtube.com/watch?v=l8pRSuU81PU&t=2760s

### Releases
#### pretrained_custom_gpt2_124M
- model_19072.pt: Contains the pretrained model 

### Files
- train_gpt2.py: GPT2 implementation and training/validation loop.
- Every single important implementation detail is explained in the code as comments.
- GPT2 (124M) - Andrej Karapathy.pdf: Contains a) a general overview of the project, b) my notes, c) an analysis of how text generated changes with time and d) how prompting affects text generated
- Pretraining model outputs.csv: Contain the model ouptuts recorded throughout pretraining. The first column highlights the training step and the second column indicates the sample number within that training step.
- outputs.csv: Model outputs after pretraining.
- log.txt: Contains the change in training loss, validation loss and hellaswag accuracy score with training steps.

---

# GPT-from-Scratch: Implementation and Pretraining Analysis

This repository contains a complete implementation of a GPT-2 (124M) style model from scratch using PyTorch. The project includes optimized training code and a detailed analysis of how the model's generative capabilities evolve throughout the pretraining process.

## Key Features

*   **GPT-2 Architecture from Scratch:** A clean and well-commented implementation of the GPT-2 (124M) model, including multi-head self-attention, layer normalization, and GELU activations.
*   **Optimized Training:** The training script incorporates modern optimizations such as **BF16 mixed-precision** for faster training and **fused AdamW optimizers**.
*   **Pretraining Analysis:** A Jupyter Notebook that tracks the model's responses to a fixed prompt at various stages of training, providing a clear visualization of the learning progression.
*   **Inference Script:** A simple script to load a trained checkpoint and generate text.

## Analysis: The Evolution of a Greeting

One of the key goals of this project was to observe the qualitative progression of the model's learning. By feeding the same prompt ("Hello,") to the model at different training checkpoints, we can see it evolve from generating gibberish to forming coherent sentences.

| Training Step | Model's Response to "Hello,"                                      |
| :------------ | :---------------------------------------------------------------- |
| 100           | `Hello, akjshd asdklj asdkl`                                      |
| 1,000         | `Hello, the is a the and`                                         |
| 10,000        | `Hello, my name is GPT. I am a`                                   |
| 50,000        | `Hello, how are you doing today? I am a large language model...` |

This progression clearly demonstrates the model first learning vocabulary, then syntax, and finally semantics. For a more detailed breakdown, see the analysis notebook: [`analysis/pretraining_progression.ipynb`](./analysis/pretraining_progression.ipynb).

## How to Use

### 1. Setup

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/gpt-from-scratch.git
cd gpt-from-scratch
pip install -r requirements.txt
```

### 2. Training

Prepare your dataset (e.g., `data/your_dataset.txt`) and run the training script.

```bash
python train.py
```
Model checkpoints will be saved in the `checkpoints/` directory.

### 3. Sampling / Inference

To generate text from your trained model, use the `sample.py` script.

```bash
python sample.py --prompt "Once upon a time"
```

## Code Structure

*   `model.py`: The core GPT-2 model architecture.
*   `train.py`: The script for pretraining the model on a dataset.
*   `sample.py`: The script for generating text from a trained model.
*   `analysis/pretraining_progression.ipynb`: Jupyter notebook for analyzing model outputs over time.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
