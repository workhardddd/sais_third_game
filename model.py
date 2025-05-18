import torch
import torch.nn as nn
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
            print(f"Scalar features stats: min={scalar_features.min()}, max={scalar_features.max()}, mean={scalar_features.mean()}")
            print(f"Vector features stats: min={vector_features.min()}, max={vector_features.max()}, mean={vector_features.mean()}")
            combined_features = torch.nan_to_num(combined_features, nan=0.0)

        # Decode
        logits = self.decoder(combined_features, padding_mask)
        
        # Check for nan values in output
        if torch.isnan(logits).any():
            print("Warning: NaN values in decoder output")
            print(f"Logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
            logits = torch.nan_to_num(logits, nan=0.0)
            
        return logits 