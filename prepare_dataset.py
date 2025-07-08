import os
import shutil

# 1. Adjust these paths to your setup:
# SRC_ROOT: The directory where the raw, extracted PhysioNet Apnea-ECG database files are located.
#           e.g., if you unzipped 'apnea-ecg-database-1.0.0.zip', this should be the path to that unzipped folder.
SRC_ROOT = "apnea-ecg-database-1.0.0"

# DST_ROOT: The desired destination directory for the reorganized dataset.
#           By default, it creates an 'apnea-ecg' folder in the current working directory.
DST_ROOT = os.path.join(os.getcwd(), "apnea-ecg")

# RECORDS_DIR: Subdirectory within DST_ROOT to store signal-related files (.dat, .hea, etc.)
#              and the renamed annotation files (.apnea). This is where wfdb will look for records.
RECORDS_DIR = os.path.join(DST_ROOT, "records")

# ANNOT_DIR: Subdirectory within DST_ROOT to store copies of the original .apn annotation files.
#            This keeps the original annotation format separate.
ANNOT_DIR  = os.path.join(DST_ROOT, "annotations")

# Create the destination directories if they don't already exist.
# exist_ok=True prevents an error if the directories are already present.
os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

# Iterate through all files in the source root directory.
for fname in os.listdir(SRC_ROOT):
    src = os.path.join(SRC_ROOT, fname) # Construct the full source path for the current file.

    # Skip directories, process only files.
    if not os.path.isfile(src):
        continue

    name, ext = os.path.splitext(fname) # Separate filename from its extension.
    ext = ext.lower() # Convert extension to lowercase for consistent checking.

    # 2. Move all signal-related files into records/
    # Files with these extensions are typically physiological signals, headers, or related data.
    if ext in {".dat", ".hea", ".qrs", ".xws"}:
        # Copy the file to the RECORDS_DIR. shutil.copy2 preserves metadata like creation/modification times.
        shutil.copy2(src, os.path.join(RECORDS_DIR, fname))

    # 3. Rename .apn → .apnea (so wfdb.rdann(..., extension="apnea") will find it)
    #    and copy into records/, and also keep a copy in annotations/
    # The 'wfdb' library (used in dataset_loader.py) expects annotation files to have a '.apnea' extension.
    elif ext == ".apn":
        new_name = f"{name}.apnea" # Create the new filename with the '.apnea' extension.
        # Copy the renamed file to the RECORDS_DIR, making it accessible to wfdb.
        shutil.copy2(src, os.path.join(RECORDS_DIR, new_name))
        # Copy the original .apn file to the ANNOT_DIR for archival/reference.
        shutil.copy2(src, os.path.join(ANNOT_DIR, fname))

# Print confirmation messages about the reorganization.
print("Dataset reorganized under:", DST_ROOT)
print("  records/   ← contains .dat, .hea, .qrs, .xws, and .apnea files")
print("  annotations/ ← contains original .apn files")