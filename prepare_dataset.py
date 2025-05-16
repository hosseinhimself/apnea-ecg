import os
import shutil

# 1. Adjust these paths to your setup:
SRC_ROOT = "/media/hossein/f4dbfedf-df3c-4514-a2e1-7cfab19e706b/Projects/OSA/apnea-ecg-database-1.0.0"
DST_ROOT = os.path.join(os.getcwd(), "apnea-ecg")

RECORDS_DIR = os.path.join(DST_ROOT, "records")
ANNOT_DIR  = os.path.join(DST_ROOT, "annotations")

os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

for fname in os.listdir(SRC_ROOT):
    src = os.path.join(SRC_ROOT, fname)
    if not os.path.isfile(src):
        continue

    name, ext = os.path.splitext(fname)
    ext = ext.lower()

    # 2. Move all signal‐related files into records/
    if ext in {".dat", ".hea", ".qrs", ".xws"}:
        shutil.copy2(src, os.path.join(RECORDS_DIR, fname))

    # 3. Rename .apn → .apnea (so wfdb.rdann(..., extension="apnea") will find it)
    #    and copy into records/, and also keep a copy in annotations/
    elif ext == ".apn":
        new_name = f"{name}.apnea"
        shutil.copy2(src, os.path.join(RECORDS_DIR, new_name))
        shutil.copy2(src, os.path.join(ANNOT_DIR, fname))

print("Dataset reorganized under:", DST_ROOT)
print("  records/   ← contains .dat, .hea, .qrs, .xws, and .apnea files")
print("  annotations/ ← contains original .apn files")
