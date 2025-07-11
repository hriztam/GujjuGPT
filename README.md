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
7. Fine-tuning - In progress (v1 done)

### Tech Specifications

Base model: Mistral 7b

Finetuning technique: LORA + PEFT

Dataset: Sanghara(Bhasini)

## Versions

### GujjuGPT-v0

- Can generate some semi-contexted gujrati text on a given prompt
- It was trained on a small dataset and is great for gujrati/gujrat-context information
- Does not have a GUI yet, it is only a console based GPT for now
- No inference
- Hasn't been evaluated on metrics/norms
