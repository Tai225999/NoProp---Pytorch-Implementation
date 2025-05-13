import torch
import torch.nn.functional as F
import math
import json
from tqdm import tqdm

# --- Noise Schedule Helpers (for NoProp) ---

def get_cosine_schedule(T: int, s: float = 0.008):
    """
    Generate NoProp-DT cosine noise schedule (in reverse order) and compute:
    - alpha_bar: from t=0 (pure noise) to t=T (full signal)
    - snr_diff: SNR(t) - SNR(t-1)
    - inference_params: a_t, b_t, sqrt(c_t) for sampling
    
    Args:
        T (int): Total diffusion steps
        s (float): Small smoothing factor (default: 0.008)
    
    Returns:
        Tuple containing:
            - alpha_bar (torch.Tensor): Signal proportion at timestep t (length T+1)
            - snr_diff (torch.Tensor): SNR differences (length T)
            - inference_params (dict): Dictionary containing:
                - a_t (torch.Tensor): Scaling factor for predicted embedding
                - b_t (torch.Tensor): Scaling factor for current state
                - c_t_sqrt (torch.Tensor): Scaling factor for noise
    """
    # 1. Generate cosine alpha_bar from 1 → 0 and flip to get 0 → 1
    steps = torch.linspace(0, T, T + 1)
    f = lambda t: math.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = torch.tensor([f(t) / f(0) for t in steps], dtype=torch.float32)
    alpha_bar = 1.0 - alpha_bar  # Flip so t=0 is noise (0) and t=T is signal (1)
    alpha_bar = torch.clamp(alpha_bar, min=1e-5, max=1 - 1e-5)

    # 2. Compute alpha_t = alpha_bar[t - 1] / alpha_bar[t]
    alphas = torch.zeros(T + 1)
    alphas[1:] = alpha_bar[:-1] / torch.clamp(alpha_bar[1:], min=1e-8)

    # 3. SNR differences
    MAX_SNR = 100.0
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    snr = torch.clamp(snr, max=MAX_SNR)
    snr_diff = snr[1:] - snr[:-1]

    # 4. Inference parameters
    a_t = torch.zeros(T + 1)
    b_t = torch.zeros(T + 1)
    sqrt_c_t = torch.zeros(T + 1)

    for t in range(1, T + 1):
        ab_prev = alpha_bar[t - 1]
        ab_curr = alpha_bar[t]
        alpha_prev = alphas[t - 1]

        a = math.sqrt(ab_curr) * (1 - alpha_prev) / (1 - ab_prev + 1e-8)
        b = math.sqrt(alpha_prev) * (1 - ab_curr) / (1 - ab_prev + 1e-8)
        c = ((1 - ab_curr) * (1 - alpha_prev)) / (1 - ab_prev + 1e-8)

        a_t[t] = a
        b_t[t] = b
        sqrt_c_t[t] = math.sqrt(c)

    inference_params = {
        'a_t': a_t,
        'b_t': b_t,
        'c_t_sqrt': sqrt_c_t
    }
    print("Computed cosine noise schedule.")

    return alpha_bar, snr_diff, inference_params

# --- Training Functions ---

def train_noprop_dt_epoch(model, dataloader, optimizer, T, eta, device, noise_schedule):
    """
    Implements Algorithm 1 from the NoProp paper: trains NoProp-DT one epoch by iterating over t = 1..T.
    For each t, only the corresponding block is trained using the full dataset.

    Args:
        model (nn.Module): NoProp model with .blocks, .output_layer, .get_target_embedding, .generate_noisy_sample
        dataloader (torch.utils.data.DataLoader): PyTorch DataLoader containing training data
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam)
        T (int): Number of diffusion steps (i.e., number of blocks)
        eta (float): Hyperparameter scaling the denoising loss
        device (torch.device): Torch device to train on
        noise_schedule (Tuple): Tuple containing (alpha_bar, snr_diff, inference_params)

    Returns:
        float: Average total loss over all steps and batches
    """
    model.to(device)
    model.train()

    alpha_bar, snr_diff, _ = noise_schedule
    alpha_bar = alpha_bar.to(device, dtype=torch.float32)
    snr_diff = snr_diff.to(device, dtype=torch.float32)

    total_loss_accum = 0.0
    num_batches = 0

    # Loop over each diffusion step t = 1..T
    for t in range(1, T + 1):
        block_idx = t - 1
        snr_weight = snr_diff[block_idx]

        # Loop over dataset
        progress_bar = tqdm(dataloader, desc=f"Training block {block_idx}", leave=False)
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            optimizer.zero_grad()

            # Step 1: Get class embeddings u_y
            u_y = model.get_target_embedding(y)  # shape: [B, d]
            W_embed = model.embedding_matrix.weight.to(device=device, dtype=torch.float32)

            # Step 2: Sample z_{t-1} from q(z_{t-1} | y)
            t_index_tensor = torch.full((batch_size,), block_idx, device=device, dtype=torch.long)
            z_t_minus_1 = model.generate_noisy_sample(u_y, t_index_tensor)  # [B, d]

            # Step 3: Predict embedding from (z_{t-1}, x)
            predicted_u_y = model.blocks[block_idx](z_t_minus_1, x, W_embed)

            # Step 4: Compute denoising loss
            mse = F.mse_loss(predicted_u_y, u_y, reduction='none').view(batch_size, -1).mean(dim=1)
            denoising_loss = 0.5 * T * eta * (snr_weight) * mse  # shape: [B]

            # Step 5: Compute prediction loss from z_T
            t_T_tensor = torch.full((batch_size,), T, device=device, dtype=torch.long)
            z_T = model.generate_noisy_sample(u_y, t_T_tensor)
            logits = model.output_layer(z_T)
            prediction_loss = F.cross_entropy(logits, y, reduction='none')

            # Step 6: Combine and optimize
            u_y_norm_sq = u_y.pow(2).sum(dim=1)
            log_var_term = torch.log((1.0 - alpha_bar[0]) + 1e-8)
            embedding_dim = u_y.size(1)
            kl_per_item = 0.5 * (alpha_bar[0] * u_y_norm_sq
                                - embedding_dim * alpha_bar[0]
                                - embedding_dim * log_var_term
                                )
            loss = (denoising_loss + prediction_loss + kl_per_item).mean()
            loss.backward()
            optimizer.step()

            total_loss_accum += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())

    avg_total_loss = total_loss_accum / num_batches
    return avg_total_loss

@torch.no_grad()
def evaluate(model, dataloader, device, desc="Evaluating"):
    """
    Evaluates the model on the given dataloader.
    For NoProp-DT, uses the inference parameters to sample from the model.
    
    Args:
        model (nn.Module): The model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader containing evaluation data
        device (torch.device): Device to evaluate on
        desc (str, optional): Description for progress bar. Defaults to "Evaluating"
    
    Returns:
        float: Classification accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        # Get class embeddings
        u_y = model.get_target_embedding(y)
        # Generate z_T
        t_T_tensor = torch.full((batch_size,), model.num_blocks, device=device, dtype=torch.long)
        z_T = model.generate_noisy_sample(u_y, t_T_tensor)
        # Get predictions
        logits = model.output_layer(z_T)

        _, predicted = torch.max(logits.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        progress_bar.set_postfix(acc=100. * correct / total)

    accuracy = 100. * correct / total
    return accuracy

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename="checkpoint.pth.tar"):
    """
    Saves a checkpoint of the model and optimizer state.
    
    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        epoch (int): Current epoch number
        loss (float): Current loss value
        accuracy (float): Current accuracy value
        filename (str, optional): Path to save checkpoint. Defaults to "checkpoint.pth.tar"
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(state, filename)

def save_history(history_data, filename="training_history.json"):
    """
    Saves training history to a JSON file.
    
    Args:
        history_data (dict): Dictionary containing training history data
        filename (str, optional): Path to save history. Defaults to "training_history.json"
    """
    with open(filename, 'w') as f:
        json.dump(history_data, f)

def load_history(filename="training_history.json"):
    """
    Loads training history from a JSON file.
    
    Args:
        filename (str, optional): Path to history file. Defaults to "training_history.json"
    
    Returns:
        dict or None: Loaded history data if successful, None if file not found or invalid
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
