from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

class ChatRequest(BaseModel):
    text: str

app = FastAPI()

env_path = os.getenv('MODEL_PATH', 'models/korean-counsel')
model = AutoModelForCausalLM.from_pretrained(env_path, torch_dtype=torch.float16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(env_path)

@app.post('/chat')
def chat(req: ChatRequest):
    prompt = f"<|EMPATHY|> {req.text} <|ADVICE|>"
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        temperature=0.8
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {'reply': response} 