"""
prepare.py — light local validation for the "Senior AI Engineer" agent (CPU-only,
no GPU on this machine). There is NO training in this project — the persona is a
system prompt, applied at inference time.

What it does (fast, ~1-2 min):
  1. Loads HF_TOKEN from .env (optional; the model is NOT gated) and logs in if valid.
  2. Verifies the model repo is reachable (anonymous works).
  3. Loads the Qwen3-4B tokenizer.
  4. Builds the agent's system prompt + a sample user message and shows how the
     Qwen3 chat template renders it (this is exactly what the model sees).
  5. Optionally downloads + loads the full model for a CPU smoke generation
     (set LOAD_FULL_MODEL = True). Off by default to keep it fast.

Run:  python prepare.py
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LOAD_FULL_MODEL = False  # True = download ~8GB and run a CPU generation test

from prompt import SYSTEM_PROMPT


def ok(msg):
    print(f"  [OK] {msg}")


def fail(step, err):
    print(f"\n  [FAIL] {step}: {err}", file=sys.stderr)
    print("\nValidation stopped. Fix the issue above and re-run.", file=sys.stderr)
    sys.exit(1)


def main():
    print("=" * 64)
    print(" Senior AI Engineer agent — local validation (CPU, no training)")
    print("=" * 64)

    # 1. token (optional) -----------------------------------------------------
    print("\n[1/3] Hugging Face token (optional, model is not gated) ...")
    token = os.getenv("HF_TOKEN", "").strip()
    if token and not token.startswith("hf_xxxxxxxx"):
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            ok("Logged in with HF_TOKEN from .env.")
        except Exception as e:
            print(f"  [WARN] token login skipped ({e}). Continuing anonymously.")
    else:
        print("  No usable token -> continuing anonymously (model is public).")

    # 2. repo access + tokenizer ---------------------------------------------
    print(f"\n[2/3] Verify model access + load tokenizer ...")
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(MODEL_ID)
        ok(f"{MODEL_ID} reachable ({info.gated or 'not gated'}).")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        ok(f"Tokenizer loaded (vocab={tok.vocab_size}).")
    except Exception as e:
        fail("model/tokenizer", e)

    # 3. system prompt + chat template preview -------------------------------
    print(f"\n[3/3] Build the Senior AI Engineer prompt + chat template ...")
    try:
        sample_q = ("I have 50k labeled support tickets. Should I fine-tune a model "
                    "or build RAG? Walk me through the decision.")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample_q},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        print("  --- system prompt (first 300 chars) ---")
        print("  " + SYSTEM_PROMPT[:300].replace("\n", "\n  ") + " ...")
        print("\n  --- rendered chat-template tail (what the model sees) ---")
        print("  ..." + rendered[-300:].replace("\n", "\n  "))
        ok("System prompt wired into the Qwen3 chat template.")
    except Exception as e:
        fail("chat template", e)

    # optional full model load on CPU ----------------------------------------
    if LOAD_FULL_MODEL:
        print("\n[+] Loading full model on CPU for a smoke generation ...")
        try:
            import torch
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu")
            ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                          return_tensors="pt", tokenize=True)
            out = model.generate(ids, max_new_tokens=60)
            print("  Model reply:", tok.decode(out[0][ids.shape[-1]:],
                                              skip_special_tokens=True))
            ok("Full model loaded + generated on CPU.")
        except Exception as e:
            print(f"  [WARN] full-model smoke test skipped: {e}")

    print("\n" + "=" * 64)
    print(" [DONE] Agent prompt + template validated locally (no training).")
    print(" Next: upload qwen3-chat-agent.ipynb to Kaggle (GPU T4 x2),")
    print("       Run All -> chat with the Senior AI Engineer agent.")
    print("=" * 64)


if __name__ == "__main__":
    main()
