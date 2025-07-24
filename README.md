# GujjuGPT

## Getting Started

GujjuGPT is a Gujrati language based LLM Model
It can take Gujrati as the input, process it and gives it's output back in gujrati

It is still under-development

### Steps

1. Data collection - Done
2. Data extraction - Done
3. Data cleaning - Done
4. Creating a high quality Instruction-style dataset - In Progress
5. i. Fine-tuned llama 1b on 500 Q/A - Done
   ii. Generating more questions - In progress
6. Quantization of the model - Done
7. Fine-tuning - In progress (v0 done)
8. Inference - Not completed

### Tech Specifications

Base model: Llama2-7b (Changed the base model from Mistral to llama)

Finetuning technique: LORA + PEFT

Dataset: Sanghara(Bhasini)

## Versions

### GujjuGPT-v0

- Can generate some semi-contexted gujrati text on a given prompt
- It was trained on a small dataset and is great for gujrati/gujrat-context information
- Does not have a GUI yet, it is only a console based GPT for now
- No inference
- Hasn't been evaluated on metrics/norms

### GujjuGPT-v1

- It has it's own tokenizer now allowing it to embedd input and understand context better
- Made a ChatGPT-like UI Interface for Inference
- Current stats:
- - Training loss: 0.352
- - Validation loss: 0.346

### Limitations

- Limited by Compute to train it on a larger dataset
- Currently using small chunks of data with Lora and Peft to fine-tune the model
