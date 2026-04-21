# ============================================================
# inference.py — Test the General LLM
# ============================================================

import torch
from config import DEVICE, MODEL_PATH, BLOCK_SIZE
from model import GeneralLLM
from tokenizer import GeneralTokenizer

def load_model():
    print("Loading General LLM...")
    tok = GeneralTokenizer()
    tok.load()

    model = GeneralLLM().to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    return model, tok

def generate(model, tok, prompt, max_tokens=150, temperature=0.8, top_k=40):
    ids = tok.encode(prompt)
    x   = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    out = model.generate(x, max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())

if __name__ == "__main__":
    model, tok = load_model()

    prompts = [
        "artificial intelligence is",
        "the history of the internet",
        "how does a computer work",
        "python programming language",
        "machine learning algorithms",
    ]

    print("\n=== General LLM Generation Test ===")
    for prompt in prompts:
        response = generate(model, tok, prompt)
        print(f"\nPrompt   : {prompt}")
        print(f"Response : {response}")
        print("-" * 60)
