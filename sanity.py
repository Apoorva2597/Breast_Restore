from transformers import pipeline

clf = pipeline(
    "zero-shot-classification",
    model="bart_large_mnli",
    tokenizer="bart_large_mnli"
)

result = clf(
    "The tissue expander was removed and replaced with a silicone implant.",
    candidate_labels=["stage2 reconstruction", "not stage2"]
)

print(result)
