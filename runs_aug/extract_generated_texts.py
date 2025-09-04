import json
import os

# paths
input_file = "run_0_0.01/results.json"   # your big file
output_file = "run_0_0.01/generated_texts.json"

# load results
with open(input_file, "r") as f:
    data = json.load(f)

# extract just the generated_text fields
generated_texts = [entry["generated_text"] for entry in data]

# save to new file
with open(output_file, "w") as f:
    json.dump(generated_texts, f, indent=2)

print(f"Saved {len(generated_texts)} texts to {output_file}")
