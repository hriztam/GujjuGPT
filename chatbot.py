from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_path = "./gujjugpt-lora"  # path to your saved merged model

from huggingface_hub import login

dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
if os.path.exists(dotenv_path):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path)

# Replace with your actual token from: https://huggingface.co/settings/tokens
login(os.getenv("HUGGINGFACE_API_KEY"))


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("mps" if torch.backends.mps.is_available() else "cpu")
model.eval()

# Define instruction-style prompt
instruction = "આપેલ ગુજરાતી લખાણના આધારે, નીચેના પ્રશ્નનો જવાબ આપો."
input_text = "પ્રધાનમંત્રી નરેન્દ્ર મોદીના મતે રાષ્ટ્રીય શિક્ષણ નીતિનો મુખ્ય ઉદ્દેશ્ય શું છે?"
prompt = f"{instruction}\n\n{input_text}\n\n"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
