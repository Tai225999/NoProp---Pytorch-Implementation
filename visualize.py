import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from typing import Dict, List, Optional, Tuple
import os

def plot_training_history(history_file: str, save_dir: Optional[str] = None):
    """
    Plot training history from a JSON file.
    
    Args:
        history_file (str): Path to the training history JSON file
        save_dir (str, optional): Directory to save plots. If None, plots are displayed.
    
    The function creates three subplots:
    1. Training and validation accuracy over time
    2. Training loss over time
    3. Learning rate over time
    """
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot accuracy
    epochs = range(1, len(history['train_accuracy_history']) + 1)
    ax1.plot(epochs, history['train_accuracy_history'], label='Train')
    ax1.plot(epochs, history['val_accuracy_history'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(epochs, history['train_loss_history'], label='Train')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate
    ax3.plot(epochs, history['lr_history'], label='Learning Rate')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
    else:
        plt.show()
    plt.close()

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str], save_path: Optional[str] = None):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        class_names (List[str]): Names of classes
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def visualize_embeddings(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                        device: torch.device, save_path: Optional[str] = None):
    """
    Visualize class embeddings using t-SNE.
    
    Args:
        model (nn.Module): NoProp-DT model
        dataloader (DataLoader): DataLoader containing evaluation data
        device (torch.device): Device to run inference on
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    from sklearn.manifold import TSNE
    
    # Get embeddings for all samples
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            u_y = model.get_target_embedding(y)
            embeddings.append(u_y.cpu().numpy())
            labels.append(y.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Class Embeddings')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_noise_schedule(alpha_bar: torch.Tensor, snr_diff: torch.Tensor,
                       save_path: Optional[str] = None):
    """
    Plot the noise schedule used in NoProp-DT.
    
    Args:
        alpha_bar (torch.Tensor): Signal proportion at each timestep
        snr_diff (torch.Tensor): SNR differences between timesteps
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot alpha_bar
    timesteps = range(len(alpha_bar))
    ax1.plot(timesteps, alpha_bar.cpu().numpy())
    ax1.set_title('Noise Schedule (alpha_bar)')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Signal Proportion')
    ax1.grid(True)
    
    # Plot SNR differences
    ax2.plot(timesteps[:-1], snr_diff.cpu().numpy())
    ax2.set_title('SNR Differences')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('SNR Difference')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def visualize_predictions(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                         device: torch.device, num_samples: int = 10,
                         save_path: Optional[str] = None):
    """
    Visualize model predictions on sample images.
    
    Args:
        model (nn.Module): NoProp-DT model
        dataloader (DataLoader): DataLoader containing evaluation data
        device (torch.device): Device to run inference on
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        x, y = next(iter(dataloader))
        x, y = x[:num_samples].to(device), y[:num_samples]
        
        # Get predictions
        logits = model.forward_inference(x, model.noise_schedule_params)
        _, preds = torch.max(logits, 1)
        
        # Denormalize images
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        x_denorm = x.cpu() * std + mean
        
        # Plot
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        for i in range(num_samples):
            # Plot image
            axes[0, i].imshow(x_denorm[i].permute(1, 2, 0).numpy())
            axes[0, i].set_title(f'True: {y[i].item()}\nPred: {preds[i].item()}')
            axes[0, i].axis('off')
            
            # Plot prediction probabilities
            probs = torch.softmax(logits[i], dim=0)
            axes[1, i].bar(range(len(probs)), probs.cpu().numpy())
            axes[1, i].set_title('Class Probabilities')
            axes[1, i].set_xticks(range(len(probs)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

if __name__ == '__main__':
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description='Visualize NoProp-DT results')
    parser.add_argument('--history', type=str, help='Path to training history JSON file')
    parser.add_argument('--save-dir', type=str, help='Directory to save plots')
    args = parser.parse_args()
    
    if args.history:
        plot_training_history(args.history, args.save_dir)

