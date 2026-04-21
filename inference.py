# ============================================================
# inference.py —  LLM Inference
# Load trained model and generate responses
# Use this to plug into your existing agent
# ============================================================

import torch
import os
from model import KONEModel
from tokenizer import KONETokenizer
from config import DEVICE, MODEL_PATH, TOKENIZER_PATH, BLOCK_SIZE


class KONEInference:

    def __init__(self):
        self.device    = DEVICE
        self.model     = None
        self.tokenizer = None
        self.loaded    = False

    def load(self):
        print("Loading KONE LLM...")

        # load tokenizer
        self.tokenizer = KONETokenizer()
        self.tokenizer.load(TOKENIZER_PATH)

        # load model
        self.model = KONEModel().to(self.device)

        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"WARNING: No trained model found at {MODEL_PATH}")
            print("Using random weights — train first!")

        self.model.eval()
        self.loaded = True
        print("KONE LLM ready!")

    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.7, top_k: int = 40) -> str:
        if not self.loaded:
            self.load()

        # encode prompt
        ids = self.tokenizer.encode(prompt)
        if not ids:
            return ""

        # crop to block size
        ids   = ids[-BLOCK_SIZE:]
        idx   = torch.tensor([ids], dtype=torch.long, device=self.device)

        # generate
        with torch.no_grad():
            out_ids = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # decode only new tokens
        new_ids = out_ids[0][len(ids):].tolist()
        return self.tokenizer.decode(new_ids)

    def confidence_score(self, prompt: str) -> float:
        """
        Returns a confidence score 0-1 for how well
        the model handles this prompt.
        Use this to decide whether to fall back to LLaMA.
        """
        if not self.loaded:
            self.load()

        ids = self.tokenizer.encode(prompt)
        if len(ids) < 2:
            return 0.0

        ids    = ids[-BLOCK_SIZE:]
        x      = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        target = torch.tensor([ids[1:]],  dtype=torch.long, device=self.device)

        with torch.no_grad():
            _, loss = self.model(x, target)

        # lower loss = higher confidence
        import math
        perplexity = math.exp(min(loss.item(), 10))
        confidence = max(0.0, min(1.0, 1.0 - (perplexity / 100)))
        return round(confidence, 3)


# ── Plug into your existing agent ────────────────────────
# Replace LLaMA with this:
#
# from inference import KONEInference
# kone_llm = KONEInference()
# kone_llm.load()
#
# response   = kone_llm.generate(query)
# confidence = kone_llm.confidence_score(query)
#
# if confidence < 0.5:
#     # fall back to LLaMA
#     response = ollama.chat(model="llama3.2:1b", ...)


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    llm = KONEInference()
    llm.load()

    test_prompts = [
        "vpn not working",
        "password reset request",
        "laptop screen black",
        "outlook not syncing",
    ]

    print("\n=== KONE LLM Generation Test ===\n")
    for prompt in test_prompts:
        response   = llm.generate(prompt, max_new_tokens=30)
        confidence = llm.confidence_score(prompt)
        print(f"Prompt     : {prompt}")
        print(f"Response   : {response}")
        print(f"Confidence : {confidence}")
        print("-" * 40)
