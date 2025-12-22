import json

# Load both
with open("models/qwennf4/tokenizer.json") as f:
    local = json.load(f)

# Find remote path
from transformers import AutoTokenizer
tokenizer_remote = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
remote_path = tokenizer_remote.name_or_path
print(f"Remote cached at: {remote_path}")

# Or manually:
import glob
remote_file = glob.glob(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/tokenizer.json"
)[0]

with open(remote_file) as f:
    remote = json.load(f)

# Compare specific fields that matter
print("Comparing key fields:")
print(f"Vocab size: local={len(local.get('model', {}).get('vocab', []))}, "
      f"remote={len(remote.get('model', {}).get('vocab', []))}")

# Check if regex patterns differ
local_pre = local.get('pre_tokenizer', {}).get('pretokenizers', [])
remote_pre = remote.get('pre_tokenizer', {}).get('pretokenizers', [])

if local_pre != remote_pre:
    print("PRE-TOKENIZERS DIFFER!")
    print(f"Local:  {local_pre}")
    print(f"Remote: {remote_pre}")
else:
    print("Pre-tokenizers match")

# Check normalizer
if local.get('normalizer') != remote.get('normalizer'):
    print("NORMALIZERS DIFFER!")
else:
    print("Normalizers match")