import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from dataset import Gesture_Dataset
from model_MobileViT import MicroDopplerMobileViT


def evaluate(model_path='./saved_models_gesture/MicroDopplerMobileViT_best.pth',
             csv_file='./gesture_metadata.csv',
             batch_size=128,
             num_workers=8,
             device=None,
             save_dir='./saved_models_gesture'):
    """
    Load a trained model and evaluate it on the test set.
    Saves metrics as JSON and confusion matrix as .npy in save_dir.
    """
    # --- Device setup ---
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load metadata CSV ---
    df = pd.read_csv(csv_file)

    # Remove duplicate rows & reset index
    df = df.drop_duplicates().reset_index(drop=True)

    # Clean gesture names (remove bracketed content, strip whitespace)
    df['gesture_name'] = df['gesture_name'].str.replace(r'\(.*?\)', '', regex=True)
    df['gesture_name'] = df['gesture_name'].str.strip()

    # Filter target gestures
    target_keywords = ['Push&Pull', 'Sweep', 'Clap', 'Slide', 'Draw-O', 'Draw-Zigzag']
    pattern = '|'.join(target_keywords)
    df = df[df['gesture_name'].str.contains(pattern, regex=True, na=False)]

    # Train-test split with stratification
    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df['gesture_name'], random_state=42
    )
    print(f"📊 Loaded {len(df)} samples from metadata.")
    print(f"📊 [Data] Train: {len(train_df)} | Test: {len(test_df)}")

    # --- Dataset and DataLoader ---
    # Train dataset is still instantiated to keep consistent label mapping
    train_dataset = Gesture_Dataset(train_df, augment=False)
    test_dataset = Gesture_Dataset(test_df, augment=False)
    num_classes = len(train_dataset.gesture_classes)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Build model & Load weights ---
    model = MicroDopplerMobileViT(num_classes=num_classes).to(device)
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Loaded model from: {model_path}")

    # --- Evaluation loop ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (doppler, label) in enumerate(test_loader):
            doppler = doppler.to(device)
            label = label.to(device).long()

            outputs = model(doppler)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            # Print all predictions in this batch
            print(f"\n🟦 [Eval] Batch {batch_idx + 1}/{len(test_loader)}:")
            for i in range(len(outputs)):
                print(f"  Predicted: {predicted[i].item()} | Label: {label[i].item()}")

    # --- Compute metrics ---
    test_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    report = classification_report(
        all_labels, all_preds, target_names=train_dataset.gesture_classes, output_dict=True
    )
    macro_f1 = float(report['macro avg']['f1-score'])
    macro_precision = float(report['macro avg']['precision'])
    macro_recall = float(report['macro avg']['recall'])

    print(f"📏 Test Accuracy: {test_acc:.4f} | 🧠 Macro-F1: {macro_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.gesture_classes))

    # --- Save metrics and confusion matrix ---
    os.makedirs(save_dir, exist_ok=True)
    model_class_name = model.__class__.__name__
    meta_filename = f'{model_class_name}_eval_metrics.json'
    cm_filename = f'{model_class_name}_eval_confusion.npy'

    metrics = {
        'test_accuracy': test_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'classification_report': report,
        'num_classes': num_classes,
        'classes': train_dataset.gesture_classes,
        'num_test_samples': len(test_dataset)
    }
    with open(os.path.join(save_dir, meta_filename), 'w') as f_json:
        json.dump(metrics, f_json, indent=2)

    cm = confusion_matrix(all_labels, all_preds)
    np.save(os.path.join(save_dir, cm_filename), cm)

    print(f"📄 Metrics saved: {meta_filename}")
    print(f"🔢 Confusion matrix saved: {cm_filename}")

    return metrics, cm


if __name__ == '__main__':
    # Change this to your trained model path
    evaluate(model_path='./saved_models_gesture/MicroDopplerMobileViT_best.pth')
