import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import Gesture_Dataset
import json
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from model_MobileViT import MicroDopplerMobileViT


def train():
    # === Hyperparameters ===
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 384
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load metadata CSV ===
    csv_file = './gesture_metadata.csv'
    df = pd.read_csv(csv_file)

    # ✅ Remove duplicate rows
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # ✅ Clean gesture names (remove bracketed content and strip whitespace)
    df['gesture_name'] = df['gesture_name'].str.replace(r'\(.*?\)', '', regex=True)
    df['gesture_name'] = df['gesture_name'].str.strip()

    # ✅ Filter target gestures
    target_keywords = ['Push&Pull', 'Sweep', 'Clap', 'Slide', 'Draw-O', 'Draw-Zigzag']
    pattern = '|'.join(target_keywords)
    df = df[df['gesture_name'].str.contains(pattern, regex=True, na=False)]

    # === Train-test split with stratification ===
    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['gesture_name'], random_state=42)
    print(f"📊 Loaded {len(df)} samples from metadata.")
    print(f"📊 [Data] Train: {len(train_df)} samples | Test: {len(test_df)} samples")

    # === Directory to save models ===
    model_save_path = './saved_models_gesture'
    os.makedirs(model_save_path, exist_ok=True)

    # === Prepare dataset and dataloaders ===
    train_dataset = Gesture_Dataset(train_df, augment=True)
    test_dataset = Gesture_Dataset(test_df, augment=False)
    num_classes = len(train_dataset.gesture_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # ✅ Compute class weights to handle class imbalance
    label_counts = train_df['gesture_name'].value_counts()
    total_count = len(train_df)
    class_counts = [label_counts.get(g, 1) for g in train_dataset.gesture_classes]
    class_weights = [total_count / count for count in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("🧮 Class Distribution & Weights:")
    for g, count, weight in zip(train_dataset.gesture_classes, class_counts, class_weights):
        print(f"  Gesture: {str(g):15s} | Count: {count:5d} | Weight: {weight.item():.6f}")

    # === Initialize model, loss, optimizer, scheduler ===
    model = MicroDopplerMobileViT(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=196, gamma=0.5)

    best_acc = 0.0
    best_f1 = 0.0
    print("🚀 Training started...")

    # === Training loop ===
    for epoch in range(num_epochs):
        model.train()
        correct = total = 0
        epoch_loss = 0.0

        # --- Training batches ---
        for batch_idx, (doppler, label) in enumerate(train_loader):
            doppler, label = doppler.to(device), label.to(device).long()
            outputs = model(doppler)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * doppler.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if batch_idx % 100 == 0:
                print(f"🟢 Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        train_acc = correct / total
        avg_loss = epoch_loss / total
        print(f"📘 Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        # === Evaluation on test set ===
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (doppler, label) in enumerate(test_loader):
                doppler, label = doppler.to(device), label.to(device).long()
                outputs = model(doppler)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                if batch_idx == 0:
                    print("🟦 [Eval] Sample predictions vs labels:")
                    for i in range(min(20, len(outputs))):
                        print(f"Predicted: {predicted[i].item()}, Label: {label[i].item()}")

        test_acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_preds)
        report = classification_report(all_labels, all_preds,
                                       target_names=train_dataset.gesture_classes,
                                       output_dict=True)

        macro_f1 = report['macro avg']['f1-score']
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']

        print(f"📏 Test Accuracy: {test_acc:.4f} | 🧠 Macro-F1: {macro_f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=train_dataset.gesture_classes))

        # === Save the best model based on Macro-F1 ===
        if macro_f1 > best_f1 and epoch >= 96:
            best_f1 = macro_f1
            best_acc = test_acc

            model_class_name = model.__class__.__name__
            model_filename = f'{model_class_name}_best.pth'
            meta_filename = f'{model_class_name}_best_metrics.json'
            cm_filename = f'{model_class_name}_best_confusion.npy'

            torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))

            metrics = {
                'epoch': epoch + 1,
                'test_accuracy': test_acc,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'classification_report': report
            }
            with open(os.path.join(model_save_path, meta_filename), 'w') as f_json:
                json.dump(metrics, f_json, indent=2)

            cm = confusion_matrix(all_labels, all_preds)
            np.save(os.path.join(model_save_path, cm_filename), cm)

            print(f"✅ Saved new best model: {model_filename}")
            print(f"📄 Metrics saved: {meta_filename}")
            print(f"🔢 Confusion matrix saved: {cm_filename}")

        scheduler.step()
        print("=" * 40)

    print(f"🏁 Training complete. Best Test Accuracy: {best_acc:.4f} | Best Macro-F1: {best_f1:.4f}")
    print(f"📊 Total samples: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")


if __name__ == '__main__':
    train()
