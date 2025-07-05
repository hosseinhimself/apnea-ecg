## main.py

import os
import sys
import yaml
import argparse
from typing import Dict, Any

import torch

from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


def load_config(config_path: str) -> Dict[str, Any]:
    """
    فایل کانفیگ YAML را بارگذاری می‌کند.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main() -> None:
    """
    تابع اصلی برای اجرای کامل پایپ‌لاین تشخیص آپنه خواب.
    """
    parser = argparse.ArgumentParser(description="Run OSA detection experiments based on the paper.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config file '{args.config}': {e}")
        sys.exit(1)

    print("======================================================")
    print("OSA Detection - Exact Replication of the Paper")
    print(f"Using config file: {args.config}")
    print("======================================================")

    try:
        # --- مرحله ۱: آماده‌سازی دیتاست ---
        # تمام منطق بارگذاری، پیش‌پردازش و تقسیم‌بندی اکنون درون get_dataloaders قرار دارد.
        print("[Data] Initializing DatasetLoader and creating dataloaders...")
        dataset_loader = DatasetLoader(config)
        train_loader, val_loader, test_loader = dataset_loader.get_dataloaders()
        print("[Data] Dataloaders created successfully.")

        # --- مرحله ۲: ساخت مدل ---
        print("[Model] Building CNN-Transformer model as per the paper's architecture...")
        model = Model(params=config)

        # --- مرحله ۳: آموزش مدل ---
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
        trainer.train()
        print("[Training] Training finished.")

        # --- مرحله ۴: ارزیابی مدل ---
        # پس از آموزش، بهترین مدل ذخیره شده با نام 'best_model.pth' را بارگذاری می‌کنیم.
        print("[Evaluation] Loading best model and evaluating on the test set...")
        model.load_state_dict(torch.load("best_model.pth"))
        
        evaluator = Evaluation(model=model, test_loader=test_loader)
        segment_metrics = evaluator.evaluate()
        
        # این بخش را می‌توان در صورت نیاز فعال کرد
        # recording_metrics = evaluator.aggregate_recording_results()

        # --- مرحله ۵: گزارش نتایج ---
        print("\n[Results] Segment-wise metrics on the test set:")
        def fmt_float(v: float) -> str:
            if v is None or (isinstance(v,float) and (v != v)):
                return "N/A"
            return f"{v*100:.2f}%"

        print(f"  Accuracy:           {fmt_float(segment_metrics.get('accuracy'))}")
        print(f"  Sensitivity:        {fmt_float(segment_metrics.get('sensitivity'))}")
        print(f"  Specificity:        {fmt_float(segment_metrics.get('specificity'))}")
        print(f"  F1 Score:           {fmt_float(segment_metrics.get('f1_score'))}")
        auc_val = segment_metrics.get("auc")
        print(f"  AUC:                {auc_val:.4f}" if isinstance(auc_val, float) else "N/A")
        
    except Exception as e:
        print(f"\nFATAL ERROR during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()