import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from gvp_src.gvp_encoder import GVPEncoder
from decoder import RNADecoder

class RNAModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = GVPEncoder(args)
        self.decoder = RNADecoder(
            input_dim=args.node_hidden_dim_scalar + args.node_hidden_dim_vector * 3,
            hidden_dim=args.node_hidden_dim_scalar,
            num_heads=8,
            num_layers=3,
            dropout=args.dropout
        )
        
    def forward(self, coords, coord_mask, padding_mask, confidence):
        # Ensure correct dimensions
        # coords: [B, L, 7, 3]
        # coord_mask: [B, L]
        # padding_mask: [B, L]
        # confidence: [B, L]
        
        # Add assertions to verify input shapes
        assert coords.dim() == 4, f"Expected coords to be 4D, got {coords.dim()}D"
        assert coord_mask.dim() == 2, f"Expected coord_mask to be 2D, got {coord_mask.dim()}D"
        assert padding_mask.dim() == 2, f"Expected padding_mask to be 2D, got {padding_mask.dim()}D"
        assert confidence.dim() == 2, f"Expected confidence to be 2D, got {confidence.dim()}D"
        
        # Check for nan values
        if torch.isnan(coords).any():
            print("Warning: NaN values in input coordinates")
            coords = torch.nan_to_num(coords, nan=0.0)
        
        # Encode
        encoded = self.encoder(coords, coord_mask, padding_mask, confidence)
        
        # Concatenate scalar and vector features
        scalar_features, vector_features = encoded
        # Reshape vector features: [B, L, V, 3] -> [B, L, V*3]
        vector_features = vector_features.reshape(vector_features.shape[0], vector_features.shape[1], -1)
        # Concatenate along feature dimension
        combined_features = torch.cat([scalar_features, vector_features], dim=-1)
        
        # Check for nan values after encoding
        if torch.isnan(combined_features).any():
            print("Warning: NaN values in encoded features")
            combined_features = torch.nan_to_num(combined_features, nan=0.0)

        # Decode
        logits = self.decoder(combined_features, padding_mask)
        
        # Check for nan values in output
        if torch.isnan(logits).any():
            print("Warning: NaN values in decoder output")
            logits = torch.nan_to_num(logits, nan=0.0)
            
        return logits

def process_coordinates(coords):
    return np.nan_to_num(coords, nan=0.0)

def predict_sequences(model, coords_dir, device):
    model.eval()
    results = []
    
    # Get all npy files
    coord_files = sorted([f for f in Path(coords_dir).glob('*.npy')])
    
    for coord_file in tqdm(coord_files, desc="Predicting sequences"):
        # Load and preprocess coordinates
        coords = np.load(coord_file)
        coords = process_coordinates(coords)
        
        # Convert to tensor
        coords = torch.FloatTensor(coords).unsqueeze(0)  # Add batch dimension
        coord_mask = torch.ones(1, coords.shape[1], dtype=torch.bool)
        padding_mask = torch.ones(1, coords.shape[1], dtype=torch.bool)
        confidence = torch.ones(1, coords.shape[1], dtype=torch.float)
        
        # Move to device
        coords = coords.to(device)
        coord_mask = coord_mask.to(device)
        padding_mask = padding_mask.to(device)
        confidence = confidence.to(device)
        
        # Get predictions
        with torch.no_grad():
            logits = model(coords, coord_mask, padding_mask, confidence)
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert predictions to sequence
        idx_to_base = {0: 'A', 1: 'U', 2: 'G', 3: 'C'}
        sequence = ''.join([idx_to_base[idx.item()] for idx in predictions[0]])
        
        # Store results
        results.append({
            'id': coord_file.stem,
            'sequence': sequence
        })
    
    return pd.DataFrame(results)

def main():
    # Get environment variables
    input_dir = os.getenv('INPUT_DIR', '/saisdata')
    output_dir = os.getenv('OUTPUT_DIR', '/saisresult')
    output_file = os.getenv('OUTPUT_FILE', 'submit.csv')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    class Args:
        def __init__(self):
            self.node_hidden_dim_scalar = 256
            self.node_hidden_dim_vector = 64
            self.edge_hidden_dim_scalar = 256
            self.edge_hidden_dim_vector = 64
            self.num_encoder_layers = 3
            self.dropout = 0.1
            self.top_k_neighbors = 30
    
    args = Args()
    model = RNAModel(args).to(device)
    
    # Load trained model
    checkpoint_path = "/app/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Checkpoint loaded successfully")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found")
        return
    
    # Process test data and generate predictions
    coords_dir = os.path.join(input_dir, "coords")
    results_df = predict_sequences(model, coords_dir, device)
    
    # Save results
    output_path = os.path.join(output_dir, output_file)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main() 