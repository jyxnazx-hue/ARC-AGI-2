import json
from pathlib import Path
# pyrefly: ignore [missing-import]
import kagglehub

# Replace with your exact Kaggle account handle (case-sensitive)
KAGGLE_USERNAME = "jyxnazx"
DATASET_SLUG = "arc-agi-2-split"
handle = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

# 1. Point directly to your local data folder
data_dir = Path("data").resolve()

# 2. Write required dataset metadata into data/
meta_file = data_dir / "dataset-metadata.json"
meta = {
    "title": "ARC-AGI-2 Split",
    "id": handle,
    "licenses": [{"name": "CC0-1.0"}]
}
with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Uploading {data_dir} to Kaggle as {handle}...")

# 3. Upload dataset via kagglehub
kagglehub.dataset_upload(
    handle=handle,
    local_dataset_dir=str(data_dir)
)

print("\nUpload complete! Dataset is ready on Kaggle.")