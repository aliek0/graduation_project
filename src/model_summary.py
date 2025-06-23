from torchsummary import summary
from model import STGCN_v2

# Create a model instance for 5 classes
model = STGCN_v2(num_classes=5)

print("ST-GCN Model Architecture Summary")

# --- CHANGE ---
# The input_size now includes the "M" (bodies) dimension, which is 2 in our dataset.
# This creates a 5D tensor (Batch, Channels, Frames, Joints, Bodies) that the model expects.
# Shape: (3, 150, 25, 2) -> (C, T, V, M)
summary(model, input_size=(3, 150, 25, 2), device="cpu")