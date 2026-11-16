"""List Anthropic API models available to our key (bake-off prerequisite).
Never prints the key. Run: .venv/bin/python scripts/check_models.py"""
from dotenv import load_dotenv

load_dotenv()

import anthropic  # noqa: E402

client = anthropic.Anthropic()
print("Models available to this key:")
for m in client.models.list(limit=100).data:
    print(f"  {m.id}")

# Probe the specific candidates we care about with a 1-token ping
CANDIDATES = ["claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
print("\nPing test (1-token) per candidate:")
for mid in CANDIDATES:
    try:
        r = client.messages.create(model=mid, max_tokens=1,
                                   messages=[{"role": "user", "content": "hi"}])
        print(f"  {mid}: OK ({r.model})")
    except Exception as e:
        print(f"  {mid}: UNAVAILABLE — {type(e).__name__}: {str(e)[:100]}")
