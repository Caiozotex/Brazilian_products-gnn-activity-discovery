import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os
import numpy as np

# Keep your existing masked_bce_loss
def masked_bce_loss(logits, targets):
    mask = (targets != -1).float()
    loss_unreduced = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    return (loss_unreduced * mask).sum() / mask.sum()

def evaluate_loss(model, loader, device):
    """Return average masked BCE loss over a loader."""
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y.view(-1, 12)   # ensure shape (batch_size, 12)
            loss = masked_bce_loss(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
    return total_loss / total_graphs

def pretrain_tox21(model, train_loader, optimizer, device,
                   val_loader=None, epochs=50,
                   save_dir="checkpoints/tox21",
                   patience=10, checkpoint_every=5):
    """
    Pretrain Tox21 with validation, early stopping, and checkpointing.

    Args:
        model: Tox21Model
        train_loader: DataLoader for training
        val_loader:   DataLoader for validation (if None, splits train_loader 80/20)
        epochs:       max epochs
        save_dir:     directory for checkpoints and best model
        patience:     early stopping patience
        checkpoint_every: save a snapshot every this many epochs
    """
    os.makedirs(save_dir, exist_ok=True)

    # If no val_loader provided, create one from 20% of training data
    if val_loader is None:
        dataset = train_loader.dataset   # list of Data
        n = len(dataset)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]
        train_subset = [dataset[i] for i in train_idx]
        val_subset   = [dataset[i] for i in val_idx]
        train_loader = DataLoader(
            train_subset, batch_size=train_loader.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=train_loader.batch_size, shuffle=False
        )
        print(f"Split training set: {len(train_subset)} train, {len(val_subset)} val")

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y.view(-1, 12)
            loss = masked_bce_loss(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        val_loss = evaluate_loss(model, val_loader, device)

        # Logging
        log_msg = f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
        print(log_msg)

        # Checkpointing (periodic)
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            chk_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, chk_path)
            print(f" -> Checkpoint saved: {chk_path}")

        # Early stopping and best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            # Save best model (just the encoder for easy reuse, or the whole model)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            torch.save(model.encoder.state_dict(), os.path.join(save_dir, "best_encoder.pt"))
            print(f" -> New best model (val loss {val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    print(f"Training finished. Best epoch: {best_epoch} with val loss {best_val_loss:.4f}")
    return model