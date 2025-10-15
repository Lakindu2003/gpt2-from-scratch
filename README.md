# GPT-from-Scratch: Implementation and Pretraining Analysis

This repository contains a complete implementation of a GPT-2 (124M) style model from scratch using PyTorch. The project includes optimized training code and a detailed analysis of how the model's generative capabilities evolve throughout the pretraining process.

## Key Features

*   **GPT-2 Architecture from Scratch:** A clean and well-commented implementation of the GPT-2 (124M) model, including multi-head self-attention, layer normalization, and GELU activations.
*   **Optimized Training:** The training script incorporates modern optimizations such as **BF16 mixed-precision** for faster training and **fused AdamW optimizers**.
*   **Pretraining Analysis:** A Jupyter Notebook that tracks the model's responses to a fixed prompt at various stages of training, providing a clear visualization of the learning progression.
*   **Inference Script:** A simple script to load a trained checkpoint and generate text.

## Analysis: The Evolution of a Greeting

One of the key goals of this project was to observe the qualitative progression of the model's learning. By feeding the same prompt ("Hello,") to the model at different training checkpoints, we can see it evolve from generating gibberish to forming coherent sentences.

]

## Important files
**[View the Final Report (PDF)](https://lakindu2003.github.io/hk-gmb-fare-equity/capstone_final_report.pdf)**
*   [`train_gpt2.py`](https://github.com/Lakindu2003/gpt2-from-scratch/blob/main/train_gpt2.py): Implementation and training code of GPT2 from scratch with detailed documentation and personal comments.
*   [`project_report_unedited.pdf`](https://lakindu2003.github.io/gpt2-from-scratch/project_report_unedited.pdf): Unedited full project report.
*   [`pretraining_model_outputs.csv`](https://github.com/Lakindu2003/gpt2-from-scratch/blob/main/pretraining_model_outputs.csv): Model outputs recorded across pretraining.
*   [`outputs.csv`](https://github.com/Lakindu2003/gpt2-from-scratch/blob/main/outputs.csv): Model outputs.
*   [`log.txt`](https://github.com/Lakindu2003/gpt2-from-scratch/blob/main/log.txt): Pretraining evaluation log with train and validation losses.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References
1. https://www.youtube.com/watch?v=l8pRSuU81PU&t=2760s
2. https://github.com/karpathy/build-nanogpt/tree/master
3. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
4. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
