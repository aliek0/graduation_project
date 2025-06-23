import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import re
import json

from model import STGCN_v2

# --- TRAINING SETTINGS (UPDATED FOR 5 CLASSES) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'KTH_yolo_npy/'
NUM_CLASSES = 5
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
TARGET_FRAME_COUNT = 150

ACTION_NAMES = ["boxing", "handclapping", "handwaving", "running", "walking"]
ACTION_MAP = {name: i for i, name in enumerate(ACTION_NAMES)}

class KTHDataset(Dataset):
    """Custom Dataset for loading KTH .npy files."""
    def __init__(self, data_dir, action_map, target_frames):
        self.data_dir = data_dir
        self.action_map = action_map
        self.target_frames = target_frames
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".npy"):
                action_name_match = re.search(r'_([a-z]+)_', filename)
                if action_name_match:
                    action_name = action_name_match.group(1)
                    if action_name in self.action_map:
                        label = self.action_map[action_name]
                        samples.append((os.path.join(self.data_dir, filename), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        data = np.load(npy_path)
        
        # 1. Padding / Truncating
        current_frames = data.shape[0]
        if current_frames > self.target_frames:
            start = (current_frames - self.target_frames) // 2
            data = data[start : start + self.target_frames, :, :, :]
        elif current_frames < self.target_frames:
            padding = np.zeros((self.target_frames - current_frames, 2, 25, 3))
            data = np.concatenate((data, padding), axis=0)
            
        # 2. Normalization (relative to the center joint)
        center_joint = data[:, :, 1, :][:, :, np.newaxis, :]
        data = data - center_joint
        
        # 3. Transpose for PyTorch (T, M, V, C) -> (C, T, V, M)
        data = np.transpose(data, (3, 0, 2, 1)).astype(np.float32)
        
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)

def main():
    print(f"Using device: {DEVICE}")

    # DataLoaders
    train_dataset = KTHDataset(data_dir=os.path.join(DATA_DIR, 'train'), action_map=ACTION_MAP, target_frames=TARGET_FRAME_COUNT)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = KTHDataset(data_dir=os.path.join(DATA_DIR, 'val'), action_map=ACTION_MAP, target_frames=TARGET_FRAME_COUNT)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Model, Loss Function, Optimizer
    model = STGCN_v2(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_accuracy = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=running_loss/len(train_loader), acc=100.*correct_train/total_train)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)

        # --- Validation Phase ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=val_loss/len(val_loader), acc=100.*correct_val/total_val)
        
        val_loss_epoch = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        history['val_loss'].append(val_loss_epoch)
        history['val_acc'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'kth_best_model_5_class.pth')
            print(f"✨ New best model saved! Accuracy: {best_val_accuracy:.2f}%")

    print(f"\nTraining finished! Best validation accuracy: {best_val_accuracy:.2f}%")
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    print("Training history saved to 'training_history.json'.")

if __name__ == '__main__':
    main()