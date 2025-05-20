import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from model import RNAModel
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import torch.nn.functional as F

class RNADataset(Dataset):
    def __init__(self, coords_dir, split='train', test_size=0.2, random_state=42):
        self.coords_dir = Path(coords_dir)
        self.seqs_dir = self.coords_dir.parent / 'seqs'
        self.split = split
        
        # Get all npy files
        self.coord_files = sorted([f for f in self.coords_dir.glob('*.npy')])
        
        # Split into train and test
        train_files, test_files = train_test_split(
            self.coord_files, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Select files based on split
        self.files = train_files if split == 'train' else test_files
        
        # Load all data
        self.coords = []
        self.labels = []
        self.coord_masks = []
        self.padding_masks = []
        self.confidences = []
        
        for coord_file in tqdm(self.files, desc=f"Loading {split} data"):
            # Load coordinates
            coords = np.load(coord_file)  # [L, 7, 3]
            coords = self.preprocess_coords(coords)
            
            # Load corresponding fasta file
            seq_id = coord_file.stem
            fasta_file = self.seqs_dir / f"{seq_id}.fasta"
            if not fasta_file.exists():
                print(f"Warning: Fasta file not found for {coord_file.name}")
                continue
                
            # Read sequence using Bio.SeqIO
            seq = str(next(SeqIO.parse(fasta_file, "fasta")).seq)
            
            # Check length consistency
            L = len(seq)
            assert coords.shape[0] == L, f"坐标长度 {coords.shape[0]} 与序列长度 {L} 不匹配 (ID: {seq_id})"
            
            # Create coordinate mask
            coord_mask = np.ones(L, dtype=bool)  # 只使用第一个原子的mask
            # Mark rows with all zeros as invalid
            zero_rows = np.all(coords == 0, axis=(1, 2))
            if np.any(zero_rows):
                coord_mask[zero_rows] = False
            
            # Convert sequence to indices
            label_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
            labels = torch.LongTensor([label_map[l] for l in seq])
            
            # Convert to tensors
            coords_tensor = torch.FloatTensor(coords)  # [L, 7, 3]
            coord_mask_tensor = torch.BoolTensor(coord_mask)  # [L]
            padding_mask = torch.ones(L, dtype=torch.bool)  # [L]
            confidence = torch.ones(L, dtype=torch.float)  # [L]
            
            # Store data
            self.coords.append(coords_tensor)
            self.labels.append(labels)
            self.coord_masks.append(coord_mask_tensor)
            self.padding_masks.append(padding_mask)
            self.confidences.append(confidence)
    
    def preprocess_coords(self, coords):
        return np.nan_to_num(coords, nan=0.0)
    
    def __len__(self):
        return len(self.coords)
    
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
        
        # Check for nan values in input
        if torch.isnan(coords).any():
            print("Warning: NaN values found in coordinates")
            coords = torch.nan_to_num(coords, nan=0.0)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(coords, coord_mask, padding_mask, confidence)  # [B, L, 4]
        
        # Check for nan values in logits
        if torch.isnan(logits).any():
            print("Warning: NaN values found in model output")
            continue
        
        # Calculate loss (ignore padding tokens)
        logits = logits.contiguous().reshape(-1, 4)
        labels = labels.contiguous().reshape(-1)
        
        # Apply softmax with numerical stability
        logits = F.log_softmax(logits, dim=-1)
        
        # Calculate loss only on non-padding tokens
        mask = labels != -100
        if mask.sum() == 0:
            print("Warning: No valid tokens in batch")
            continue
            
        loss = F.nll_loss(logits[mask], labels[mask])
        
        # Check if loss is nan
        if torch.isnan(loss):
            print("Warning: NaN loss detected")
            print(f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
            print(f"Labels stats: min={labels.min()}, max={labels.max()}")
            continue
        
        # Backward pass with gradient clipping
        loss.backward()
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
            
            # Check for nan values in input
            if torch.isnan(coords).any():
                print("Warning: NaN values found in coordinates")
                coords = torch.nan_to_num(coords, nan=0.0)
            
            # Forward pass
            logits = model(coords, coord_mask, padding_mask, confidence)  # [B, L, 4]
            
            # Check for nan values in logits
            if torch.isnan(logits).any():
                print("Warning: NaN values found in model output")
                continue
            
            # Calculate loss (ignore padding tokens)
            logits = logits.contiguous().reshape(-1, 4)
            labels = labels.contiguous().reshape(-1)
            
            # Apply softmax with numerical stability
            logits = F.log_softmax(logits, dim=-1)
            
            # Calculate loss only on non-padding tokens
            mask = labels != -100
            if mask.sum() == 0:
                print("Warning: No valid tokens in batch")
                continue
                
            loss = F.nll_loss(logits[mask], labels[mask])
            
            # Check if loss is nan
            if torch.isnan(loss):
                print("Warning: NaN loss detected")
                print(f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
                print(f"Labels stats: min={labels.min()}, max={labels.max()}")
                continue
            
            total_loss += loss.item()
            
            # Calculate accuracy (ignore padding tokens)
            predictions = torch.argmax(logits, dim=-1)  # [B*L]
            correct += (predictions[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, args, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Checkpoint loaded successfully")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
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

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to continue training')
    args = parser.parse_args()
    
    # Create datasets
    coords_dir = os.path.join("RNA_design_public", "RNA_design_public", "RNAdesignv1", "train", "coords")
    train_dataset = RNADataset(coords_dir, split='train', test_size=args.test_size, random_state=args.random_state)
    test_dataset = RNADataset(coords_dir, split='test', test_size=args.test_size, random_state=args.random_state)
    
    # Create dataloaders with collate_fn
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=RNADataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=RNADataset.collate_fn)
    
    # Initialize model
    model = RNAModel(args)
    
    # Train model
    train_model(model, train_loader, test_loader, args, checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main() 