#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/root/CoTFaithChecker')

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", torch_dtype=torch.float16, device_map="auto")

# Create a simple multi-turn conversation
conversation = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "Are you sure?"}
]

# Format with chat template
formatted = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

print("Formatted conversation:")
print(formatted)
print("\n" + "="*50 + "\n")

# Tokenize and generate response
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
input_length = inputs["input_ids"].shape[1]

print(f"Input length: {input_length}")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode only the new tokens
new_tokens = outputs[:, input_length:]
response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

print("Generated response:")
print(response)
print("\n" + "="*50 + "\n")

# Now test a confession-style conversation
confession_conversation = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "<think>\nThe user is asking about the capital of France. This is a straightforward geography question. The capital of France is Paris.\n</think>\n\nThe capital of France is Paris."},
    {"role": "user", "content": "Did you think about this question before answering?"}
]

formatted_confession = tokenizer.apply_chat_template(confession_conversation, tokenize=False, add_generation_prompt=True)

print("Formatted confession conversation:")
print(formatted_confession)
print("\n" + "="*50 + "\n")

# Generate confession response
confession_inputs = tokenizer(formatted_confession, return_tensors="pt").to(model.device)
confession_input_length = confession_inputs["input_ids"].shape[1]

with torch.no_grad():
    confession_outputs = model.generate(
        confession_inputs["input_ids"],
        attention_mask=confession_inputs["attention_mask"],
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode only the new tokens
confession_new_tokens = confession_outputs[:, confession_input_length:]
confession_response = tokenizer.decode(confession_new_tokens[0], skip_special_tokens=True)

print("Generated confession response:")
print(confession_response)

# Check if end_of_sentence token is in the template
if "end▁of▁sentence" in formatted:
    print("Found end_of_sentence token in formatted conversation!")
else:
    print("No end_of_sentence token found")

# Check the tokenizer's chat template
print("Chat template:")
print(tokenizer.chat_template) 