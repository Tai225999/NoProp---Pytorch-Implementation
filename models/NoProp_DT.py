import torch
import torch.nn as nn
import torch.nn.functional as F
from models.NoPropBlock import NoPropBlock

class NoPropNetDT(nn.Module):
    """
    Implements the NoProp-DT model structure.
    
    The model consists of:
    1. T instances of NoPropBlock for denoising
    2. A final classification layer
    3. An embedding matrix for class representations
    
    The core training logic (loss, updates) happens outside this definition, typically in train.py.
    This class primarily defines the components and the inference pass.
    """
    
    def __init__(self, num_blocks, num_classes, embedding_dim, img_channels=3, img_size=32):
        """
        Initialize the NoProp-DT model.
        
        Args:
            num_blocks (int): Number of denoising blocks (T in the paper)
            num_classes (int): Number of classes in the dataset
            embedding_dim (int): Dimension of the embedding space
            img_channels (int, optional): Number of input image channels. Defaults to 3.
            img_size (int, optional): Size of input images. Defaults to 32.
        """
        super(NoPropNetDT, self).__init__()
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Create T instances of the shared block
        # Each block θ_t is trained independently in NoProp
        self.blocks = nn.ModuleList([
            NoPropBlock(num_classes, embedding_dim, img_channels, img_size)
            for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(embedding_dim, num_classes)

        # Class Embedding Matrix
        # If learned, it's updated during training along with blocks and output_layer
        self.embedding_matrix = nn.Embedding(num_classes, embedding_dim)
        # Orthogonal initialization if possible
        if embedding_dim >= num_classes:
             torch.nn.init.orthogonal_(self.embedding_matrix.weight)

    def set_noise_schedule(self, alpha_bar):
        """
        Stores the precomputed alpha_bar values for the diffusion process.
        
        Args:
            alpha_bar (torch.Tensor): Precomputed alpha_bar values for each timestep
        """
        self.register_buffer('alpha_bar', alpha_bar)

    def get_target_embedding(self, y):
        """
        Gets the clean embedding u_y for labels y.
        
        Args:
            y (torch.Tensor): Class labels (Batch,)
            
        Returns:
            torch.Tensor: Embeddings for the labels (Batch, embedding_dim)
        """
        return self.embedding_matrix(y)

    def generate_noisy_sample(self, u_y, t_idx):
        """
        Generates z_t ~ q(z_t | y) = N(z_t | sqrt(alpha_bar_t)*u_y, (1-alpha_bar_t)*I)
        
        Args:
            u_y (torch.Tensor): Clean embeddings (Batch, embedding_dim)
            t_idx (torch.Tensor): Time indices (Batch,)
            
        Returns:
            torch.Tensor: Noisy samples z_t (Batch, embedding_dim)
            
        Raises:
            ValueError: If noise schedule (alpha_bar) is not set
        """
        if self.alpha_bar is None:
            raise ValueError("Noise schedule (alpha_bar) not set.")
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t_idx]).view(-1, 1)
        one_minus_alpha_bar_t = (1.0 - self.alpha_bar[t_idx]).view(-1, 1)

        epsilon = torch.randn_like(u_y)
        z_t = sqrt_alpha_bar_t * u_y + torch.sqrt(one_minus_alpha_bar_t) * epsilon
        return z_t

    def forward_inference(self, x, noise_schedule_params):
        """
        Implements the NoProp inference pass (denoising from z_0 to z_T).
        
        The inference process:
        1. Starts with random noise z_0
        2. Iteratively denoises through T steps
        3. Uses the final state z_T for classification
        
        Args:
            x (torch.Tensor): Input image batch (Batch, Channels, Height, Width)
            noise_schedule_params (dict): Dictionary containing:
                - a_t (torch.Tensor): Scaling factor for predicted embedding
                - b_t (torch.Tensor): Scaling factor for current state
                - c_t_sqrt (torch.Tensor): Scaling factor for noise

        Returns:
            torch.Tensor: Logits output from the final classification layer (Batch, num_classes)
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize z_0 with Gaussian noise
        z_t = torch.randn(batch_size, self.embedding_dim, device=device)

        # Retrieve inference parameters from schedule
        a_t_vals = noise_schedule_params['a_t']
        b_t_vals = noise_schedule_params['b_t']
        c_t_sqrt_vals = noise_schedule_params['c_t_sqrt']
        W_embed = self.embedding_matrix.weight.to(device=device, dtype=torch.float32)

        # Apply T denoising steps sequentially
        for t in range(self.num_blocks): 
            # Get the prediction of the clean embedding from the t-th block
            predicted_u_y = self.blocks[t](z_t, x, W_embed) # û_θ_{t+1}(z_t, x)

            # Retrieve schedule parameters for step t+1
            a_t = a_t_vals[t+1]
            b_t = b_t_vals[t+1]
            c_t_sqrt = c_t_sqrt_vals[t+1]

            # Apply the NoProp inference step (Eq. 3)
            epsilon_t = torch.randn_like(z_t) if c_t_sqrt > 0 else torch.zeros_like(z_t)
            z_t = a_t * predicted_u_y + b_t * z_t + c_t_sqrt * epsilon_t
            # z_t now holds the state for the *next* step (z_{t+1} in paper)

        # Final prediction using the state after T steps (z_T)
        y_hat_logits = self.output_layer(z_t)
        return y_hat_logits
