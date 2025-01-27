import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model), requires_grad=True)

    def forward(self, x, axis='time'):
        if axis == 'time':
            return x + self.encoding[:x.size(0), None, :]  # Apply time encoding
        elif axis == 'location':
            return x + self.encoding[None, :x.size(1), :]  # Apply location encoding
        else:
            raise ValueError("Invalid axis for positional encoding.")


class MultiDimensionalAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiDimensionalAttention, self).__init__()
        self.time_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.feature_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # Time attention: (T, L*F, d_model)
        time_out, _ = self.time_attention(x, x, x)

        # Spatial attention: reshape to (L, T*F, d_model)
        spatial_in = x.transpose(0, 1).contiguous()
        spatial_out, _ = self.spatial_attention(spatial_in, spatial_in, spatial_in)
        spatial_out = spatial_out.transpose(0, 1).contiguous()  # Restore shape

        # Feature attention: reshape to (F, T*L, d_model)
        feature_in = x.transpose(0, 2).contiguous()
        feature_out, _ = self.feature_attention(feature_in, feature_in, feature_in)
        feature_out = feature_out.transpose(0, 2).contiguous()  # Restore shape

        # Combine outputs (e.g., via addition or concatenation)
        combined_out = time_out + spatial_out + feature_out
        return combined_out


class SimpleTransformerForecastingModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, max_time, max_locations):
        super(SimpleTransformerForecastingModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.time_pos_encoding = PositionalEncoding(d_model, max_len=max_time)
        self.location_pos_encoding = PositionalEncoding(d_model, max_len=max_locations)
        self.attention = MultiDimensionalAttention(d_model, num_heads)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (T, L, F)
        T, L, F = x.shape

        # Embed features: (T, L, F) -> (T, L, d_model)
        x = self.embedding(x)

        # Add positional encodings
        x = self.time_pos_encoding(x, axis='time')  # Time positional encoding
        x = self.location_pos_encoding(x, axis='location')  # Location positional encoding

        # Reshape for attention: flatten (L * F) per time step -> (T, L*F, d_model)
        x = x.view(T, L * F, -1)

        # Apply multi-dimensional attention
        x = self.attention(x)

        # Transformer encoder layers
        x = self.transformer_layers(x)

        # Final output projection
        x = self.fc(x)  # Map back to input_dim per feature
        x = x.view(T, L, F)  # Reshape back to original structure

        return x
