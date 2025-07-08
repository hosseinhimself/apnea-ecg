import os
import requests
from tqdm import tqdm

# Output directory
DEST_DIR = "./apnea_apn_files"
os.makedirs(DEST_DIR, exist_ok=True)

# Base URL
BASE_URL = "https://physionet.org/files/apnea-ecg/1.0.0/"

# Generate filenames: x01.apn to x35.apn
apn_files = [f"x{str(i).zfill(2)}.apn" for i in range(1, 36)]

# Download loop
for fname in tqdm(apn_files, desc="Downloading .apn files"):
    url = f"{BASE_URL}{fname}?download"
    dest_path = os.path.join(DEST_DIR, fname)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"❌ Failed to download {fname}: {e}")

print(f"✅ All downloads complete. Saved in: {DEST_DIR}")
