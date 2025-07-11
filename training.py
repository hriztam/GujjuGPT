from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import os
from huggingface_hub import login
from datasets import load_dataset

dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
if os.path.exists(dotenv_path):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path)


device = "mps" if torch.backends.mps.is_available() else "cpu"

login(os.getenv("HUGGINGFACE_API_KEY"))


# Load the model and tokenizer
model_path = "./gujjugpt-lora"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)


# Load the dataset
final_dataset = "./dataset/processed_data/qna_generation.json"
dataset = load_dataset("json", data_files=final_dataset, split="train")

#  Format the dataset for instruction-style prompts
def format_instruction(example):
    return {
        "text": f"""### સૂચના:
        {example['instruction']}

        ### ઇનપુટ:
        {example['input']}

        ### પ્રતિસાદ:
        {example['output']}"""
    }

dataset = dataset.map(format_instruction)

# Tokenize the dataset
def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./gujjugpt-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset, 
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./gujjugpt-finetuned")
tokenizer.save_pretrained("./gujjugpt-finetuned")