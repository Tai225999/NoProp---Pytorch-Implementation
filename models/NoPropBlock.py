import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class NoPropBlock(nn.Module):
    """
    Implements the û_θ_t block architecture described in the NoProp paper (Fig. 6, Left).
    
    The block consists of three main pathways:
    1. Image Processing Pathway: Processes input images through CNN layers
    2. Latent State Pathway: Processes the latent state z_t through MLP layers
    3. Combined Pathway: Processes concatenated features from both pathways
    
    Architecture:
    - Image Pathway: Conv2d -> ReLU -> MaxPool -> Dropout -> Conv2d -> ReLU -> Dropout -> Conv2d -> ReLU -> MaxPool -> Dropout -> Flatten -> Linear -> BatchNorm
    - Latent Pathway: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm
    - Combined Pathway: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear
    """
    def __init__(self, num_classes, embedding_dim, img_channels=3, img_size=32):
        """
        Initialize the NoProp block.
        
        Args:
            num_classes (int): Number of classes in the dataset
            embedding_dim (int): Dimension of the embedding space
            img_channels (int, optional): Number of input image channels. Defaults to 3.
            img_size (int, optional): Size of input images. Defaults to 32.
        """
        super(NoPropBlock, self).__init__()
        self.embedding_dim = embedding_dim

        # --- Image Processing Pathway ---
        self.conv_image_embed = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 -> 16
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16 -> 8
            nn.Dropout2d(0.3),
            nn.Flatten(),
            nn.Linear(128 * (img_size // 4) * (img_size // 4), 256),
            nn.BatchNorm1d(256)
        )
        img_feature_dim = 256

        # --- Latent State (z_t) Processing Pathway ---
        self.fc_zt_embed_1 = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc_zt_embed_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        zt_feature_dim = 256

        # --- Combined Processing Pathway ---
        combined_dim = img_feature_dim + zt_feature_dim
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # --- Block Output Layer ---
        self.logit_layer = nn.Linear(128, num_classes)

    def forward(self, z_t_1, x, W_embed):
        """
        Forward pass of the NoProp block.
        
        Args:
            z_t_1 (torch.Tensor): Latent state from previous step (Batch, embedding_dim)
            x (torch.Tensor): Input image (Batch, Channels, Height, Width)
            W_embed (torch.Tensor): Embedding matrix weights (num_classes, embedding_dim)

        Returns:
            torch.Tensor: Predicted clean embedding û_θ_t(z_{t-1}, x) (Batch, embedding_dim)
        """
        img_feat = self.conv_image_embed(x)
        zt_feat_1 = self.fc_zt_embed_1(z_t_1)
        zt_feat = self.fc_zt_embed_2(zt_feat_1) + zt_feat_1
        combined = torch.cat((img_feat, zt_feat), dim=1)
        processed_combined = self.fc_combined(combined)
        # This prediction is used in the NoProp loss (Eq. 8)
        logits = self.logit_layer(processed_combined)
        probs = F.softmax(logits, dim=1)
        predicted_u_y = torch.mm(probs, W_embed)
        return predicted_u_y


def sanity_check():
    """
    Performs a sanity check on the NoPropBlock by:
    1. Creating a model instance
    2. Generating dummy inputs
    3. Printing model summary using torchinfo
    
    This function is useful for verifying the model architecture and dimensions.
    """
    model = NoPropBlock(embedding_dim=64, img_channels=3, img_size=32)

    dummy_z = torch.randn(8, 64)                  # (Batch, embedding_dim)
    dummy_x = torch.randn(8, 3, 32, 32)           # (Batch, Channels, H, W)

    summary(model, input_data=(dummy_z, dummy_x), device='cpu')
