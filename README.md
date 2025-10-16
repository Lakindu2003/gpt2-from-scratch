# GPT-from-Scratch: Implementation and Pretraining Analysis

This repository contains a complete implementation of a GPT-2 (124M) style model from scratch using PyTorch. The report project includes optimized training code and a detailed analysis of how the model's generative capabilities evolve throughout the pretraining process.

## Datasets
This project uses the FineWeb-edu 10B Tokens dataset, a higher-quality subset of the FineWeb dataset which focuses on educational content.

## Training Optimisations
Several training optimisations were adopted from the GPT3 paper. This includes:
1. BFloat16: Increases the training speed by reducing the number of mantissa bits involved in the computation. Additionally, the reduction in VRAM usage allows for the batch size to be increased, which also increases the training speed. This assumes that BF16 support is available.
2. Kernel fusion: Kernel fusion works by reducing the number of times data travels between the SRAM and VRAM. This includes `adamw` with fused kernel, `torch.compile()`, and flash attention (softmax trick) optimizations.
3. Powers of two/divisible by two for tensor shapes: Friendly numbers for CUDA kernels.
4. Gradient normalisation and clipping: Gradients capped to 1.0. Prevents exploding gradients.
5. Learning rate scheduler: Uses Cosine decay with warmup.
6. Weight decay: Regularisation.
7. Increase batch size with time: Initial gradients are correlated, so they do contribute relatively little.
8. Gradient accumulation: Emulate the optimal hyperparameters from the GPT3 paper.
9. Token embedding tying: Reduces parameter count by approximately 30%. Regularisation technique.
10. High-quality dataset: 10B tokens from FineWeb-edu.
    
Additionally, GPT's GELU activation function was replaced with a SiLU activation function, as it appears in newer models such as LLaMA 3.1 and is considered more efficient empirically.

## Model evaluation
1. Validation loss logging
2. HellaSwag (multiple choice sentence completion evaluation dataset)

## The Evolution of how the model's outputs changed across pretraining for a greeting prompt
**Prompt: "Hello, I'm a language model,"**
1. At step 20, the model learned to predict frequently occurring  words such as “the”, “that”, and “is”.
2. At step 100, the model learned to memorize the training data. This is evident by the intelligible text generated, which has no relation to the prompt.
3. At 200-300 steps, the model learned to associate the prompt with itself and uses words such as “I” and “my”.
4. At steps 400-500, the model learned a simplified semantic understanding of the word “language” and other related words. This is evident by the decoding of words such as “word”, “argument”, and “text file” that are related to “language”.
5. At step 2800, the generated text starts to show a little relationship to the semantic meaning of the prompt and is slightly intelligible.
6. At step 6000, the generated text is much more coherent and shows some relationship to the semantic meaning of the prompt.
7. At 8000 steps, the text generated is more related to “language”, despite not being related to “language model”. The model has also learnt that there are “programming languages". This is shown by words such as “python” and “syntax”.
8. Over the next steps, the text becomes more and more coherent and related to the prompt.

A more detailed documentation with images can be found in section 8 (pp. 34-37) of the project report.

## Important files
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
