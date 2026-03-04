from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_DIR="/home/apokol/Breast_Restore/bart_large_mnli"

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

clf = pipeline(
    "zero-shot-classification",
    model=mdl,
    tokenizer=tok,
    device=-1
)

print(clf(
    "The tissue expander was removed and replaced with a silicone implant.",
    candidate_labels=["stage2 reconstruction","not stage2 reconstruction"]
))
