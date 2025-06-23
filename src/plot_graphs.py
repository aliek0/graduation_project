import json
import matplotlib.pyplot as plt

try:
    with open('training_history.json', 'r') as f:
        history = json.load(f)
except FileNotFoundError:
    print("Error: 'training_history.json' not found. Please run train.py first.")
    exit()

epochs = range(1, len(history['train_acc']) + 1)

plt.figure(figsize=(14, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_graphs.png')
print("Graphs saved to 'training_graphs.png'.")
