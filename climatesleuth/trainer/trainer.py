import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import time
import os


class ContrastiveTrainer:
    """
    Trainer class for distributed contrastive learning using NT-Xent loss.

    Args:
        model: PyTorch model (wrapped in DDP).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function (e.g., NT-Xent loss).
        optimizer: Optimizer (e.g., Adam).
        scaler: GradScaler for mixed precision training.
        device: Device to run on (e.g., torch.device("cuda")).
        rank: Process rank in DDP.
        world_size: Total number of processes.
        seed: Random seed for reproducibility.
        accumulation_steps: Gradient accumulation steps.
        patience: Early stopping patience.
        epochs: Total number of training epochs.
        save_prefix: Prefix for model checkpoint filenames.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scaler,
                 device, rank, world_size, seed, accumulation_steps=100, patience=10, epochs=100,
                 save_prefix="contrastive_model_vgg"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        self.epochs = epochs

        self.save_path = f"{save_prefix}_seed_{seed}_rank_{rank}_{accumulation_steps}_accumulation_steps.pth"
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def run(self):
        """
        Main training loop that runs for multiple epochs and handles early stopping,
        validation, logging, and saving the best model.
        """
        for epoch in range(self.epochs):
            print(f"[Rank {self.rank}] 🔁 Epoch {epoch + 1}/{self.epochs}")
            
            # Reset dataloader epoch for distributed shuffling
            self.train_loader.dataset.set_epoch(epoch)
            self.val_loader.dataset.set_epoch(epoch)
            print(f"[Rank {self.rank}] Dataset epoch reset")

            # Train for one epoch
            train_loss = self.train_one_epoch(epoch)

            # Validate on all ranks and reduce results
            val_loss_global = self.validate_one_epoch()

            # Logging and early stopping (only on rank 0)
            self.log_epoch(epoch, train_loss, val_loss_global)

            if self.rank == 0 and self.early_stop_counter >= self.patience:
                print("⛔ Early stopping triggered.")
                break

            torch.distributed.barrier()

        # Clean up distributed processes
        dist.destroy_process_group()

    def train_one_epoch(self, epoch):
        """
        Run training for a single epoch.

        Returns:
            float: Average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        start_time = time.perf_counter()

        for k, batch in enumerate(self.train_loader):
            x1 = batch["x1"].to(self.device, non_blocking=True)
            x2 = batch["x2"].to(self.device, non_blocking=True)
            x_combined = torch.cat([x1, x2], dim=0)

            with torch.autograd.detect_anomaly(), torch.cuda.amp.autocast():
                z = self.model(x_combined)
                loss = self.criterion(z[:2], z[2:])

            # Accumulate gradients
            total_loss += loss.item()
            loss = loss / self.accumulation_steps
            self.scaler.scale(loss).backward()

            # Optimizer step every few iterations
            if (k + 1) % self.accumulation_steps == 0 or (k + 1 == len(self.train_loader)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        # Sync across processes and return average
        torch.distributed.barrier()
        return total_loss / len(self.train_loader)

    def validate_one_epoch(self):
        """
        Run validation and return synchronized average validation loss.

        Returns:
            float: Average validation loss across all processes.
        """
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                x1 = batch["x1"].to(self.device, non_blocking=True)
                x2 = batch["x2"].to(self.device, non_blocking=True)
                x_combined = torch.cat([x1, x2], dim=0)

                with torch.cuda.amp.autocast():
                    z = self.model(x_combined)
                    loss = self.criterion(z[:2], z[2:])
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)

        # Synchronize loss across all processes
        val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss_global = val_loss_tensor.item() / self.world_size

        return avg_val_loss_global

    def log_epoch(self, epoch, train_loss, val_loss):
        """
        Log and save model state if validation improves.
        Only the process with rank 0 performs logging and saving.

        Args:
            epoch (int): Current epoch index.
            train_loss (float): Average training loss.
            val_loss (float): Global validation loss.
        """
        if self.rank == 0:
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"🌀 Epoch {epoch + 1} | 🏋️ Train: {train_loss:.4f} | 🎯 Val: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"💾 Model saved to {self.save_path}")
            else:
                self.early_stop_counter += 1

            # Plot and save loss curves
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Val Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training & Validation Loss")
            plt.legend()
            plt.grid()
            plot_path = f"training_curve_seed{self.seed}_rank{self.rank}_test.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"📉 Saved training plot to {plot_path}")
        else:
            print(f"[Rank {self.rank}] Finished epoch {epoch + 1}")
