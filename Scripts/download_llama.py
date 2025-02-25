"""
Script for downloading models for local inference from Huggingface

Make sure to run this:
    huggingface-cli login

"""

# MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' # Debug Math solving
# MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
# MODEL_ID = 'meta-llama/Llama-2-13b-chat-hf'
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SAVE_PATH = "."



from transformers import AutoTokenizer, AutoModelForCausalLM


# Save model and tokenizer to disk
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
