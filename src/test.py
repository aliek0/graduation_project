import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import os
from model import STGCN_v2
from train import KTHDataset, DEVICE, DATA_DIR, BATCH_SIZE, TARGET_FRAME_COUNT, ACTION_NAMES, ACTION_MAP

# --- TEST SETTINGS (UPDATED FOR 5 CLASSES) ---
MODEL_PATH = 'kth_best_model_5_class.pth'
NUM_CLASSES = 5

def plot_confusion_matrix(cm, class_names):
    """Function to plot and save the confusion matrix."""
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names)
    plt.ylim(len(class_names), 0)
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (5 Classes)')
    plt.savefig('confusion_matrix_5_class.png')
    print("\nConfusion matrix saved to 'confusion_matrix_5_class.png'")

def evaluate_model():
    print(f"Using device: {DEVICE}")
    test_dataset = KTHDataset(data_dir=os.path.join(DATA_DIR, 'test'), action_map=ACTION_MAP, target_frames=TARGET_FRAME_COUNT)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples loaded: {len(test_dataset)}")
    
    model = STGCN_v2(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=ACTION_NAMES, digits=4)
    
    print("\n--- TEST RESULTS ---")
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    with open('classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print("\nClassification report saved to 'classification_report.txt'")
    
    plot_confusion_matrix(cm, ACTION_NAMES)

if __name__ == '__main__':
    try:
        import seaborn
        import matplotlib
        import sklearn
    except ImportError:
        print("\nWarning: A required library is not found. Please install with: 'pip install scikit-learn seaborn matplotlib'")
    else:
        evaluate_model()
