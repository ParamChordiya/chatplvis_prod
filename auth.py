# auth.py
import os
from huggingface_hub import login
from dotenv import load_dotenv

def setup_hf_auth():
    load_dotenv(override=True)

    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            print("[HF AUTH] Login successful")
        except Exception as e:
            # Don't crash app startup; model loads might still work if cached
            print(f"[HF AUTH] Warning: login failed: {e}")
    else:
        print("[HF AUTH] No HUGGINGFACE_HUB_TOKEN found. "
              "If downloads fail, set it in .env or environment.")
