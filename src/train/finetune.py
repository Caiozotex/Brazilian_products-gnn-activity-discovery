
# ------------------------------
# 6. Fine-tuning on NuBBE
# ------------------------------
import torch
import torch.nn as nn
import numpy as np
import os
from torch_geometric.loader import DataLoader
from src.models.task_models import NuBBEModel

def evaluate_nubbe_loss(model, loader, device):
    """Binary cross‑entropy loss on the NuBBE validation set."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y.view(-1, 1).float()   # shape: (batch_size, 1)
            loss = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
    return total_loss / total_graphs

def finetune_nubbe(model, train_loader, optimizer, device,
                   val_loader=None, epochs=30,
                   save_dir="checkpoints/nubbe",
                   patience=10, checkpoint_every=5):
    """
    Fine-tune a NuBBEModel (binary antioxidant/ROS classifier) starting from
    a pretrained encoder (already loaded into the model).

    Args:
        model: NuBBEModel (encoder + 1‑task head)
        train_loader: DataLoader with NuBBE graphs
        optimizer: e.g., Adam with lr=1e-4
        device: torch.device
        val_loader:   optional validation loader; if None, auto‑splits train_loader 80/20
        epochs:       maximum number of epochs
        save_dir:     directory for checkpoints and best model
        patience:     early stopping patience
        checkpoint_every: save a checkpoint every this many epochs

    Returns:
        The fine‑tuned model (best weights loaded).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Auto‑split train/val if needed
    if val_loader is None:
        dataset = train_loader.dataset   # list of Data objects
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

    criterion = nn.BCEWithLogitsLoss()
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
            targets = batch.y.view(-1, 1).float()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        val_loss = evaluate_nubbe_loss(model, val_loader, device)

        log_msg = f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
        print(log_msg)

        # Periodic checkpoint
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            chk_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, chk_path)
            print(f" -> Checkpoint saved: {chk_path}")

        # Best model tracking + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            torch.save(model.encoder.state_dict(), os.path.join(save_dir, "best_encoder.pt"))
            print(f" -> New best model (val loss {val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    print(f"Fine‑tuning finished. Best epoch: {best_epoch} with val loss {best_val_loss:.4f}")
    # Load best weights into model before returning
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    return model