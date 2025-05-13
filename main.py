import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import os
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

# Import necessary components from other files
from data import load_cifar10
from models.NoProp_DT import NoPropNetDT
from train import (
    train_noprop_dt_epoch,
    evaluate,
    save_checkpoint,
    save_history,
    load_history,
    get_cosine_schedule
)

def set_lr(optimizer, lr):
    """
    Sets the learning rate for all parameter groups in the optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated
        lr (float): The new learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Optimizer LR set to {lr:.2e}")


def main(args):
    """
    Main training function for NoProp-DT model.
    
    Args:
        args (argparse.Namespace): Command line arguments containing:
            - no_cuda (bool): Whether to disable CUDA
            - checkpoint_dir (str): Directory to save checkpoints
            - embedding_type (str): Type of embedding ('one-hot', 'learned', or 'prototype')
            - embedding_dim (int): Dimension of embeddings for learned embeddings
            - num_blocks (int): Number of denoising blocks
            - batch_size (int): Batch size for training
            - lr (float): Learning rate
            - weight_decay (float): Weight decay for optimizer
            - lr_scheduler_patience (int): Patience for learning rate scheduler
            - early_stopping_patience (int): Patience for early stopping
            - epochs (int): Total number of epochs to train
            - warmup_epochs (int): Number of epochs for learning rate warmup
            - save_every (int): Save checkpoint every N epochs
            - history (str): Path to history file for resuming training
            - checkpoint (str): Path to checkpoint file for resuming training
            - eta (float): Learning rate scaling factor for NoProp
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Create Checkpoint and Log Directory ---
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    log_dir = os.path.join(args.checkpoint_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    history_filepath = os.path.join(log_dir, "training_history.json")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Log directory: {log_dir}")
    print(f"History file: {history_filepath}")

    # Load Data
    train_loader, val_loader, test_loader = load_cifar10(batch_size=args.batch_size)

    # --- Handle NoProp Embedding Strategy ---
    if args.embedding_type == 'one-hot':
        print("Using fixed one-hot embeddings.")
        print("Building NoPropNetDT model...")
        model = NoPropNetDT(
            num_blocks=args.num_blocks,
            num_classes=10,
            embedding_dim=10
        ).to(device)
        if hasattr(model, 'embedding_matrix'): model.embedding_matrix.weight.requires_grad = False
    elif args.embedding_type == 'learned':
        print(f"Using learned embeddings (dim={args.embedding_dim}).")
        print("Building NoPropNetDT model...")
        model = NoPropNetDT(
            num_blocks=args.num_blocks,
            num_classes=10,
            embedding_dim=args.embedding_dim
        ).to(device)
        if hasattr(model, 'embedding_matrix'): model.embedding_matrix.weight.requires_grad = True
    elif args.embedding_type == 'prototype':
        print("Using prototype embeddings.")
        print("Building NoPropNetDT model...")
        model = NoPropNetDT(
            num_blocks=args.num_blocks,
            num_classes=10,
            embedding_dim=10
        ).to(device)
        if hasattr(model, 'embedding_matrix'): model.embedding_matrix.weight.requires_grad = True
    else:
        raise ValueError(f"Unknown embedding type: {args.embedding_type}")

    # Build Model
    
    train_epoch_fn = train_noprop_dt_epoch

    # --- Setup Noise Schedule for NoProp ---
    print(f"Setting up cosine noise schedule with T={args.num_blocks}")
    alpha_bar, snr_diff, inference_params = get_cosine_schedule(T=args.num_blocks)
    model.set_noise_schedule(alpha_bar.to(device))
    noise_schedule_train = (alpha_bar, snr_diff, inference_params)

    print(f"Model built: NoPropNetDT")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # --- Optimizer ---
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Learning Rate Scheduler ---
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=args.lr_scheduler_patience, verbose=True, min_lr=1e-6
    )

    # --- Load Checkpoint and History (if resuming) ---
    start_epoch = 0
    best_accuracy = 0.0
    skip_warmup = False
    train_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    lr_history = []

    # Try loading history first
    if args.history:
        print(f"Attempting to load history from: {args.history}")
        loaded_hist_data = load_history(args.history)
        if loaded_hist_data:
            train_loss_history = loaded_hist_data['train_loss_history']
            train_accuracy_history = loaded_hist_data['train_accuracy_history']
            val_accuracy_history = loaded_hist_data['val_accuracy_history']
            lr_history = loaded_hist_data['lr_history']
            start_epoch = len(train_loss_history)
            best_accuracy = max(val_accuracy_history) if val_accuracy_history else 0.0
            last_lr = lr_history[-1]
            set_lr(optimizer, last_lr)
            skip_warmup = True
            print(f"Resuming from history: Start epoch {start_epoch}, Best Val Acc {best_accuracy:.2f}%, Last LR {last_lr:.2e}")
            scheduler.best = best_accuracy
            print(f"Scheduler 'best' value reset to {scheduler.best:.4f} based on loaded history.")
        else:
            print("Could not load valid history file, starting fresh.")

    # Load checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            try:
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                ckpt_epoch = checkpoint.get('epoch', 0)
                if ckpt_epoch > start_epoch:
                    print(f"Checkpoint epoch ({ckpt_epoch}) is later than history epoch ({start_epoch}). Using checkpoint epoch.")
                    start_epoch = ckpt_epoch

                ckpt_best_acc = checkpoint.get('accuracy', 0.0)
                if ckpt_best_acc > best_accuracy:
                     print(f"Updating best accuracy from checkpoint value ({ckpt_best_acc:.2f}%)")
                     best_accuracy = ckpt_best_acc
                     scheduler.best = best_accuracy
                     print(f"Scheduler 'best' value updated to {scheduler.best:.4f} from checkpoint.")

                if not args.history or not loaded_hist_data:
                    skip_warmup = True
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Resuming from checkpoint (no valid history loaded): Start epoch {start_epoch}, Best Val Acc {best_accuracy:.2f}%, Current LR {current_lr:.2e}")
                    print("Warmup skipped.")
                else:
                    set_lr(optimizer, lr_history[-1])
                    print(f"Resuming from checkpoint AND history: Start epoch {start_epoch}, Best Val Acc {best_accuracy:.2f}%, LR set to {lr_history[-1]:.2e}")

            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting training from scratch.")
                start_epoch = 0
                best_accuracy = 0.0
                skip_warmup = False
                train_loss_history, train_accuracy_history, val_accuracy_history, lr_history = [], [], [], []

        else:
            print(f"Checkpoint file not found: {args.checkpoint}. Starting training from scratch.")
            train_loss_history, train_accuracy_history, val_accuracy_history, lr_history = [], [], [], []

    # --- Early Stopping Initialization ---
    epochs_without_improvement = 0

    # --- Training Loop ---
    print(f"\nStarting/Resuming training from epoch {start_epoch} for {args.epochs} total epochs...")
    if skip_warmup: print("Warmup phase is skipped.")
    else: print(f"Warmup active for the first {args.warmup_epochs} epochs.")
    print(f"LR Scheduler: ReduceLROnPlateau (Patience: {args.lr_scheduler_patience})")
    print(f"Early Stopping: Active (Patience: {args.early_stopping_patience})")

    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        print(f"\n--- Epoch {epoch_num}/{args.epochs} ---")
        start_time = time.time()

        # --- Learning Rate Warmup Logic ---
        current_lr = optimizer.param_groups[0]['lr']
        perform_warmup_step = epoch < args.warmup_epochs and not skip_warmup
        if perform_warmup_step:
            initial_lr = 1e-5
            target_lr = args.lr
            warmup_lr = initial_lr + (target_lr - initial_lr) * (epoch / args.warmup_epochs)
            set_lr(optimizer, warmup_lr)
            current_lr = warmup_lr
            print(f"Warmup Epoch {epoch_num}/{args.warmup_epochs}: Set LR to {current_lr:.2e}")
        elif epoch == args.warmup_epochs and not skip_warmup:
            set_lr(optimizer, args.lr)
            current_lr = args.lr
            print(f"Warmup finished. Set LR to {current_lr:.2e}")

        # --- Training Step ---
        model.train()
        avg_loss = train_epoch_fn(
            model, train_loader, optimizer, args.num_blocks, args.eta, device, noise_schedule_train
        )

        # --- Validation Step ---
        model.eval()
        val_accuracy = evaluate(
            model, val_loader, device, desc="Validation"
        )

        # --- Learning Rate Scheduling ---
        scheduler.step(val_accuracy)

        # --- Early Stopping Check ---
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            # Save best model checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model.pth.tar")
            save_checkpoint(
                model, optimizer, epoch_num, avg_loss, val_accuracy,
                filename=checkpoint_path
            )
            print(f"New best model saved with accuracy: {val_accuracy:.2f}%")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
                break

        # --- Save Regular Checkpoint ---
        if (epoch_num + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch_num}.pth.tar")
            save_checkpoint(
                model, optimizer, epoch_num, avg_loss, val_accuracy,
                filename=checkpoint_path
            )

        # --- Update History ---
        train_loss_history.append(avg_loss)
        val_accuracy_history.append(val_accuracy)
        lr_history.append(current_lr)

        # --- Save History ---
        history_data = {
            'train_loss_history': train_loss_history,
            'train_accuracy_history': train_accuracy_history,
            'val_accuracy_history': val_accuracy_history,
            'lr_history': lr_history
        }
        save_history(history_data, history_filepath)

        # --- Print Epoch Summary ---
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch_num} Summary:")
        print(f"  Training Loss: {avg_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.2f}s")

    # --- Final Evaluation ---
    print("\n--- Final Evaluation ---")
    model.eval()
    test_accuracy = evaluate(
        model, test_loader, device, desc="Test"
    )
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoProp Training")
    parser.add_argument("--num-blocks", type=int, default=10, help="Number of blocks")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--embedding-type", type=str, default="learned", choices=["one-hot", "learned", "prototype"], help="Embedding type")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--eta", type=float, default=1.0, help="NoProp eta parameter")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--lr-scheduler-patience", type=int, default=5, help="LR scheduler patience")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from")
    parser.add_argument("--history", type=str, help="Path to history file to resume from")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    args = parser.parse_args()
    main(args)
