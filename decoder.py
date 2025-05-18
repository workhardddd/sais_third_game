import torch
import torch.nn as nn
import torch.nn.functional as F

class RNADecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Feed forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 4)  # 4 for A, U, G, C

    def forward(self, x, mask=None):
        # Input shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Transpose for attention: [seq_len, batch_size, hidden_dim]
        x = x.transpose(0, 1)
        
        # Handle mask
        if mask is not None:
            # mask shape should be [batch_size, seq_len]
            # Convert True to 1, False to 0 for padding mask
            mask = mask.bool()
        
        for i in range(len(self.attention_layers)):
            # Self attention
            residual = x
            x = self.layer_norms1[i](x)
            x, _ = self.attention_layers[i](x, x, x, key_padding_mask=mask)
            x = residual + x
            
            # Feed forward
            residual = x
            x = self.layer_norms2[i](x)
            x = residual + self.ffn_layers[i](x)
        
        # Project to output space
        logits = self.output_proj(x)  # [seq_len, batch_size, 4]
        
        # Transpose back to [batch_size, seq_len, 4]
        logits = logits.transpose(0, 1)
        
        return logits 