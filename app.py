"""
app.py — Gradio chat UI for the "Senior AI Engineer" agent (Qwen3-4B-Instruct-2507).

No training involved — the persona is a system prompt. Run on any machine with a GPU:

    python app.py

Optional env vars:
    BASE_MODEL  = base model id (default Qwen3-4B-Instruct-2507, NOT gated -> token optional)
    HF_TOKEN    = your HF read token (optional; set only if you hit rate limits)
    LOAD_IN_4BIT= 1 to load in 4-bit (needs a GPU + bitsandbytes); default off (bf16)
"""
import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import SYSTEM_PROMPT

if os.getenv("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "0") == "1"

print(f"Loading {BASE_MODEL} (4-bit: {LOAD_IN_4BIT}) ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
if LOAD_IN_4BIT:
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
model.eval()


def chat(message, history):
    history = history or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=1024,
            do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(reply, skip_special_tokens=True)


demo = gr.ChatInterface(fn=chat, title="Senior AI Engineer (Qwen3-4B)", type="tuples")

if __name__ == "__main__":
    demo.launch()
