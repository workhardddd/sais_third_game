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
        # coord_mask: [B, L, 7]
        # padding_mask: [B, L]
        # confidence: [B, L]
        
        # Add assertions to verify input shapes

        
        # Encode
        encoded = self.encoder(coords, coord_mask, padding_mask, confidence)
        
        # Concatenate scalar and vector features
        scalar_features, vector_features = encoded
        # Reshape vector features: [B, L, V, 3] -> [B, L, V*3]
        vector_features = vector_features.reshape(vector_features.shape[0], vector_features.shape[1], -1)
        # Concatenate along feature dimension
        combined_features = torch.cat([scalar_features, vector_features], dim=-1)

        # Decode
        logits = self.decoder(combined_features, padding_mask)
        return logits 