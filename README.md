# Senior AI Engineer Agent (Qwen3-4B)

A fast, lightweight chat agent built on **`Qwen/Qwen3-4B-Instruct-2507`** (4B params, Apache-2.0, **not gated**). The "senior AI engineer" behavior comes entirely from a **system prompt** — **there is no training / no dataset** in this project.

This folder is the **local setup**: it validates that the token, model access, and chat-template/prompt wiring work on your CPU machine. The chat itself runs on a GPU — easiest on a free **Kaggle T4**.

## Project layout
```
qwen3-chat-agent/
├── .env                 # optional HF token (gitignored). Model is public -> token optional.
├── .env.example
├── .gitignore
├── requirements.txt     # HF stack for local validation
├── prompt.py            # the Senior AI Engineer system prompt (shared)
├── prepare.py           # validates token + model access + chat-template wiring (CPU)
├── app.py               # standalone Gradio chat UI (run on a GPU)
├── qwen3-chat-agent.ipynb  # the KAGGLE notebook: load 4-bit -> chat -> Gradio
└── README.md
```

## 1. Local validation (this machine, CPU)
```bash
cd qwen3-chat-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare.py
```
`prepare.py` checks model access, loads the tokenizer, and renders the system prompt through the Qwen3 chat template so you can see exactly what the model receives. It does **not** load the full model by default (fast).

## 2. Run the agent on Kaggle (free GPU T4 x2)
1. **kaggle.com → Create → New Notebook**.
2. **Settings → Accelerator → GPU T4 x2**.
3. *(Optional)* **Add-ons → Secrets → `HF_TOKEN`** = a valid read token. The model is public, so this is optional; if the token is invalid the notebook still runs (login is best-effort and never crashes).
4. Upload `qwen3-chat-agent.ipynb` → **Run All**.
   - Loads Qwen3-4B in **4-bit** (Unsloth)
   - Applies the **Senior AI Engineer** system prompt
   - Runs an inline chat test
   - Launches a **Gradio** chat UI (`share=True`, link lasts 72h)

## 3. Run it locally on your own GPU
```bash
python app.py                       # bf16
LOAD_IN_4BIT=1 python app.py        # 4-bit (less VRAM)
```

## Swapping the persona
Edit the `SYSTEM_PROMPT` in `prompt.py` (and the matching cell in the notebook) to change the agent's role, tone, and expertise.

## Token / 401 note
If you see `401 Unauthorized` from `whoami`, the token is invalid/expired — it's **not required** here. Either remove the secret or paste a fresh fine-grained **read** token from https://huggingface.co/settings/tokens. The notebook continues anonymously either way.

## Security
- `.env` is gitignored — never commit it.
- Rotate any token that was shared/pasted (including in chat).
