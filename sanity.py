import os
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

MODEL_DIR = "/home/apokol/Breast_Restore/bart_large_mnli"

print("\n=== Checking model directory ===")
if not os.path.exists(MODEL_DIR):
    raise RuntimeError("Model directory not found: " + MODEL_DIR)

print("Model directory:", MODEL_DIR)
print("Files in model directory:")
for f in sorted(os.listdir(MODEL_DIR)):
    print(" -", f)

print("\n=== Loading config ===")
config = AutoConfig.from_pretrained(MODEL_DIR)
print("Model type:", getattr(config, "model_type", None))
print("Number of labels:", getattr(config, "num_labels", None))

print("\n=== Loading tokenizer (FORCE SLOW TOKENIZER) ===")
# KEY FIX:
# use_fast=False forces the "slow" (python) tokenizer that uses vocab.json + merges.txt
# and avoids parsing tokenizer.json via Rust tokenizers (which is failing on this server).
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
print("Tokenizer class:", tokenizer.__class__.__name__)

print("\n=== Loading model ===")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
print("Model class:", model.__class__.__name__)

print("\n=== Creating zero-shot pipeline ===")
clf = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

print("\n=== Running test ===")
text = "The tissue expander was removed and replaced with a silicone implant."
labels = ["stage2 reconstruction", "not stage2 reconstruction"]

result = clf(text, candidate_labels=labels)
print("\n=== RESULT ===")
print(result)
