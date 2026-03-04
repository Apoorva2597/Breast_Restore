import os
import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------------------------------------------
# Absolute path to your model directory
# ---------------------------------------------------
MODEL_DIR = "/home/apokol/Breast_Restore/bart_large_mnli"

print("\n=== Checking model directory ===")

if not os.path.exists(MODEL_DIR):
    raise RuntimeError("Model directory not found: " + MODEL_DIR)

print("Model directory:", MODEL_DIR)
print("Files in model directory:")

for f in os.listdir(MODEL_DIR):
    print(" -", f)

# ---------------------------------------------------
# Load config to confirm model metadata
# ---------------------------------------------------
print("\n=== Loading config ===")

config = AutoConfig.from_pretrained(MODEL_DIR)

print("Model type:", config.model_type)
print("Number of labels:", config.num_labels)

# ---------------------------------------------------
# Load tokenizer
# ---------------------------------------------------
print("\n=== Loading tokenizer ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print("Tokenizer loaded")

# ---------------------------------------------------
# Load model weights
# ---------------------------------------------------
print("\n=== Loading model ===")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

print("Model loaded successfully")

# ---------------------------------------------------
# Build zero-shot pipeline
# ---------------------------------------------------
print("\n=== Creating zero-shot pipeline ===")

classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer
)

print("Pipeline created")

# ---------------------------------------------------
# Run test inference
# ---------------------------------------------------
print("\n=== Running test ===")

text = "The tissue expander was removed and replaced with a silicone implant."

labels = [
    "stage2 reconstruction",
    "not stage2 reconstruction"
]

result = classifier(text, labels)

print("\n=== RESULT ===")
print(result)
