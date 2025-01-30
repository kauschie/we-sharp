import json
import os

codebook = {str(number): idx for idx, number in enumerate(range(1024))}

# save codebook as JSON
output_folder = "CodeBooks"
os.makedirs(output_folder, exist_ok=True)

with open(os.path.join(output_folder, "NBcodebook.json"), "w") as f:
    json.dump(codebook, f, indent=2)
print(f"Codebook saved to {output_folder}/NBcodebook.json")
