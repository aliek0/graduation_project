import torch
import torch.nn as nn

class STGCN_v2(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network (ST-GCN) architecture
    for skeleton-based action recognition.
    """
    def __init__(self, num_classes=5, drop_prob=0.5):
        super(STGCN_v2, self).__init__()
        
        # Spatial Layers
        self.spatial_conv1 = nn.Conv2d(3, 64, kernel_size=1)
        self.spatial_conv2 = nn.Conv2d(64, 16, kernel_size=1)
        
        # Temporal Layers
        self.temporal_conv1 = nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 25, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # Input shape: (N, C, T, V, M) -> (Batch, Channels, Frames, Joints, Bodies)
        # Reshape for model: (N*M, C, T, V)
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, C, T, V)
        
        # Spatial convolution
        x = self.relu(self.spatial_conv1(x))
        x = self.relu(self.spatial_conv2(x))
        
        # Temporal convolution
        x = self.relu(self.temporal_conv1(x))
        x = self.relu(self.temporal_conv2(x))
        
        # Average pooling over the temporal dimension
        x = x.mean(dim=2)
        
        # Flatten
        x = x.view(N * M, -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Reshape output to original batch size (average over bodies)
        x = x.view(N, M, -1)
        x = x.mean(dim=1)
        
        return x