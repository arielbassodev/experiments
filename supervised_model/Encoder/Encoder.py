import torch
import torch.nn as nn
import math
import einops
from einops import rearrange, repeat, reduce
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LinearProjection(nn.Module):
    def __init__(self,input_dim, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.input_dim = input_dim
        self.linear_projection = nn.Linear(self.input_dim, self.latent_size).to(torch.float32).to(device)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.latent_size), requires_grad=True).to(torch.float32).to(device)
        self.positional_encoding = PositionalEncoding(self.latent_size)

    def forward(self, sign_batch, masks):
        sign_batch = self.linear_projection(sign_batch)
        class_token = self.class_token.expand(sign_batch.shape[0], -1, -1)
        sign_batch = torch.cat((class_token, sign_batch), dim=1)
        class_token_mask = torch.zeros(sign_batch.shape[0], 1).bool().to(device)
        masks = torch.cat((class_token_mask, masks), dim=1)
        sign_batch = self.positional_encoding(sign_batch)
        return sign_batch, masks

class VitModel(nn.Module):
    def __init__(self, d_model, input_dim,  n_heads, num_blocks):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.encoder = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True)
        self.linear_projection = LinearProjection(self.input_dim, self.d_model)
        self.transformer = nn.TransformerEncoder(self.encoder, self.num_blocks)
        self.num_classes = 500
        self.classifier = nn.Linear(self.d_model, self.num_classes).to(device)

    def forward(self,x, mask):
        x, masks = self.linear_projection(x, mask)
        x = self.transformer(x, src_key_padding_mask=masks)
        cls_token = x[:,0]
        output = self.classifier(cls_token)
        return output

#class VitTransformerEncoder(nn.Module):
