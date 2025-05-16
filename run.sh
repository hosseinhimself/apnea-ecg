#!/bin/bash

echo "📁 Creating expected folder structure..."
mkdir -p apnea-ecg/records
mkdir -p apnea-ecg/annotations

echo "🚚 Copying only matching .dat/.hea files with annotation files..."

cd apnea-ecg-database-1.0.0

# Loop through all .dat files
for dat_file in *.dat; do
    record_id="${dat_file%.dat}"
    annotation_file="${record_id}-apnea.atr"

    # ✅ Corrected: check for the file in the current directory
    if [[ -f "$annotation_file" ]]; then
        echo "✅ Including $record_id"
        cp "${record_id}.dat" ../apnea-ecg/records/
        cp "${record_id}.hea" ../apnea-ecg/records/
        cp "${annotation_file}" ../apnea-ecg/annotations/
    else
        echo "⏭️ Skipping $record_id (no annotation: $annotation_file)"
    fi
done

cd ..

echo "✅ Files copied successfully."

# Optional: install Python dependencies
echo "📦 Installing Python dependencies..."
#pip install wfdb torch numpy pandas scikit-learn pyyaml > /dev/null

# Run your main script
echo "🚀 Running main.py with config.yaml..."
python3 main.py --config config.yaml

