import torch
import torch.nn as nn
class CNNEncoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(CNNEncoder, self).__init__()
        
        # 64x64x1 -> 32x32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        # 32x32x32 -> 16x16x64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        # 16x16x64 -> 8x8x128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        # 8x8x128 -> 4x4x256
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, mlp_dim=512):
        super(ViTEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = 16  # 4x4
        
        # Patch embedding: 256 channels -> embed_dim
        self.patch_embed = nn.Linear(256, embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (batch, 256, 4, 4)
        batch_size = x.shape[0]
        
        # Reshape to patches: (batch, 16, 256)
        x = x.flatten(2).transpose(1, 2)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, 16, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 17, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Return CLS token
        return x[:, 0]  # (batch, embed_dim)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super(CrossAttention, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key_value):
        # query: (batch, embed_dim)
        # key_value: (batch, embed_dim)
        
        # Add sequence dimension
        query = query.unsqueeze(1)  # (batch, 1, embed_dim)
        key_value = key_value.unsqueeze(1)  # (batch, 1, embed_dim)
        
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)  # (batch, embed_dim)
        
        output = self.norm(query.squeeze(1) + attn_output)
        return output


class HybridCNNViT(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, mlp_dim=512, num_classes=2, dropout_rate=0.3):
        super(HybridCNNViT, self).__init__()
        
        # Separate CNN encoders for miRNA and mRNA
        self.mirna_cnn = CNNEncoder(dropout_rate=dropout_rate)
        self.mrna_cnn = CNNEncoder(dropout_rate=dropout_rate)
        
        # Separate ViT encoders
        self.mirna_vit = ViTEncoder(embed_dim, num_heads, num_layers, mlp_dim)
        self.mrna_vit = ViTEncoder(embed_dim, num_heads, num_layers, mlp_dim)
        
        # Cross attention
        self.mirna_cross_attn = CrossAttention(embed_dim, num_heads)
        self.mrna_cross_attn = CrossAttention(embed_dim, num_heads)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, mrna, mirna):
        # CNN encoding
        mirna_features = self.mirna_cnn(mirna)  # (batch, 256, 4, 4)
        mrna_features = self.mrna_cnn(mrna)    # (batch, 256, 4, 4)
        
        # ViT encoding - get CLS tokens
        mirna_cls = self.mirna_vit(mirna_features)  # (batch, embed_dim)
        mrna_cls = self.mrna_vit(mrna_features)     # (batch, embed_dim)
        
        # Cross attention
        mirna_attended = self.mirna_cross_attn(mirna_cls, mrna_cls)
        mrna_attended = self.mrna_cross_attn(mrna_cls, mirna_cls)
        
        # Concatenate
        combined = torch.cat([mirna_attended, mrna_attended], dim=1)
        
        # Classification
        output = self.classifier(combined)
        
        return output


