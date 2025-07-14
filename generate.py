from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sys
import os

MODEL_DIR = os.path.join('models', 'gpt2-finetuned')

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

def generate_quote(prompt=""):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    print(generate_quote(prompt)) 