import argparse
from model import RNAModel
from train import train_model, RNADataset
from torch.utils.data import DataLoader
import os

def get_args():
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--node_hidden_dim_scalar', type=int, default=256)
    parser.add_argument('--node_hidden_dim_vector', type=int, default=64)
    parser.add_argument('--edge_hidden_dim_scalar', type=int, default=256)
    parser.add_argument('--edge_hidden_dim_vector', type=int, default=64)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--top_k_neighbors', type=int, default=30)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Create datasets
    coords_dir = os.path.join("RNA_design_public", "RNA_design_public", "RNAdesignv1", "train", "coords")
    train_dataset = RNADataset(coords_dir, split='train', test_size=args.test_size, random_state=args.random_state)
    test_dataset = RNADataset(coords_dir, split='test', test_size=args.test_size, random_state=args.random_state)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    
    # Initialize model
    model = RNAModel(args)
    
    # Train model
    train_model(model, train_loader, test_loader, args)

if __name__ == "__main__":
    main()
