import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import RNAModel
from tqdm import tqdm
import random

class RandomRNADataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # Generate random data
        self.coords = []
        self.labels = []
        self.coord_masks = []
        self.padding_masks = []
        self.confidences = []
        
        for _ in range(num_samples):
            # Random sequence length between 50 and seq_length
            L = random.randint(50, seq_length)
            
            # Generate random coordinates [L, 7, 3]
            coords = torch.randn(L, 7, 3)
            self.coords.append(coords)
            
            # Generate random labels (0-3 for A, U, G, C)
            labels = torch.randint(0, 4, (L,))
            self.labels.append(labels)
            
            # Generate random coordinate masks [L, 7]
            coord_mask = torch.ones(L, 7, dtype=torch.bool)
            # Randomly mask some coordinates
            mask_prob = 0.1
            random_mask = torch.rand(L, 7) < mask_prob
            coord_mask[random_mask] = False
            # Take only the first atom's mask for node features
            coord_mask = coord_mask[:, 0]  # [L]
            self.coord_masks.append(coord_mask)
            
            # Generate padding mask [L]
            padding_mask = torch.ones(L, dtype=torch.bool)
            self.padding_masks.append(padding_mask)
            
            # Generate confidence scores [L]
            confidence = torch.rand(L)
            self.confidences.append(confidence)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            self.coords[idx],  # [L, 7, 3]
            self.labels[idx],  # [L]
            self.coord_masks[idx],  # [L]
            self.padding_masks[idx],  # [L]
            self.confidences[idx]  # [L]
        )
    
    @staticmethod
    def collate_fn(batch):
        # Unzip the batch
        coords, labels, coord_masks, padding_masks, confidences = zip(*batch)
        
        # Get max length in this batch
        max_len = max(x.size(0) for x in coords)
        
        # Pad sequences
        padded_coords = []
        padded_labels = []
        padded_coord_masks = []
        padded_padding_masks = []
        padded_confidences = []
        
        for i in range(len(batch)):
            # Pad coordinates
            pad_len = max_len - coords[i].size(0)
            padded_coord = torch.nn.functional.pad(coords[i], (0, 0, 0, 0, 0, pad_len))
            padded_coords.append(padded_coord)
            
            # Pad labels
            padded_label = torch.nn.functional.pad(labels[i], (0, pad_len), value=-100)
            padded_labels.append(padded_label)
            
            # Pad masks
            padded_coord_mask = torch.nn.functional.pad(coord_masks[i], (0, pad_len), value=False)
            padded_coord_masks.append(padded_coord_mask)
            
            padded_padding_mask = torch.nn.functional.pad(padding_masks[i], (0, pad_len), value=False)
            padded_padding_masks.append(padded_padding_mask)
            
            padded_confidence = torch.nn.functional.pad(confidences[i], (0, pad_len), value=0.0)
            padded_confidences.append(padded_confidence)
        
        # Stack tensors
        coords = torch.stack(padded_coords)  # [B, L, 7, 3]
        labels = torch.stack(padded_labels)  # [B, L]
        coord_masks = torch.stack(padded_coord_masks)  # [B, L]
        padding_masks = torch.stack(padded_padding_masks)  # [B, L]
        confidences = torch.stack(padded_confidences)  # [B, L]
        
        # Add assertions to verify tensor shapes
        assert coords.dim() == 4 and coords.shape[2] == 7 and coords.shape[3] == 3, f"coords shape: {coords.shape}"
        assert coord_masks.dim() == 2, f"coord_masks shape: {coord_masks.shape}"
        assert padding_masks.dim() == 2, f"padding_masks shape: {padding_masks.shape}"
        assert confidences.dim() == 2, f"confidences shape: {confidences.shape}"
        
        return coords, labels, coord_masks, padding_masks, confidences

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for coords, labels, coord_mask, padding_mask, confidence in tqdm(train_loader, desc="Training"):
        # Move data to device
        coords = coords.to(device)  # [B, L, 7, 3]
        labels = labels.to(device)  # [B, L]
        coord_mask = coord_mask.to(device)  # [B, L]
        padding_mask = padding_mask.to(device)  # [B, L]
        confidence = confidence.to(device)  # [B, L]
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(coords, coord_mask, padding_mask, confidence)  # [B, L, 4]
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            print("NaN detected in logits!")
            continue
        
        # Calculate loss (ignore padding tokens)
        logits = logits.contiguous().view(-1, 4)
        labels = labels.contiguous().view(-1)
        loss = criterion(logits, labels)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print("NaN detected in loss!")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for coords, labels, coord_mask, padding_mask, confidence in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            coords = coords.to(device)  # [B, L, 7, 3]
            labels = labels.to(device)  # [B, L]
            coord_mask = coord_mask.to(device)  # [B, L]
            padding_mask = padding_mask.to(device)  # [B, L]
            confidence = confidence.to(device)  # [B, L]
            
            # Forward pass
            logits = model(coords, coord_mask, padding_mask, confidence)  # [B, L, 4]
            
            # Calculate loss (ignore padding tokens)
            logits = logits.contiguous().view(-1, 4)
            labels = labels.contiguous().view(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Calculate accuracy (ignore padding tokens)
            predictions = torch.argmax(logits.view(-1, 4), dim=-1)  # [B*L]
            mask = labels != -100
            correct += (predictions[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create argument namespace
    class Args:
        def __init__(self):
            self.node_hidden_dim_scalar = 128
            self.node_hidden_dim_vector = 16
            self.edge_hidden_dim_scalar = 32
            self.edge_hidden_dim_vector = 1
            self.num_encoder_layers = 3
            self.dropout = 0.1
            self.top_k_neighbors = 30
            self.batch_size = 32
            self.learning_rate = 0.0001  # 降低学习率
            self.epochs = 10
    
    args = Args()
    
    # Create datasets
    train_dataset = RandomRNADataset(num_samples=1000, seq_length=100)
    test_dataset = RandomRNADataset(num_samples=200, seq_length=100)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=RandomRNADataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=RandomRNADataset.collate_fn)
    
    # Initialize model
    model = RNAModel(args)
    
    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)  # 添加权重衰减
    
    best_accuracy = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluation
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 