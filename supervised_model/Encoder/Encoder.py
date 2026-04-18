import torch
import torch.nn as nn
import math
import einops
from einops import rearrange, repeat, reduce

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
        self.linear_projection = nn.Linear(self.input_dim, self.latent_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.latent_size), requires_grad=True)
        self.positional_encoding = PositionalEncoding(self.latent_size)

    def forward(self, sign_batch, masks):
        sign_batch = self.linear_projection(sign_batch)
        class_token = self.class_token.expand(sign_batch.shape[0], -1, -1)
        sign_batch = torch.cat((class_token, sign_batch), dim=1)
        class_token_mask = torch.zeros(sign_batch.shape[0], 1)
        masks = torch.cat((masks, class_token_mask), dim=1)
        sign_batch = self.positional_encoding(sign_batch)
        return sign_batch, masks

class VitModel(nn.Module):
    def __init__(self, d_model, input_dim,  n_heads, num_blocks):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.encoder = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads)
        self.linear_projection = LinearProjection(self.input_dim, self.d_model)
        self.transformer = nn.TransformerEncoder(self.encoder, self.num_blocks)

    def forward(self,x, masks):
        x, masks = self.linear_projection(x, masks)
        x = self.transformer.forward(x, src_key_padding_mask=masks)
        return x

if __name__ == '__main__':
    example_x = torch.randn(512, 50,512)
    n_heads, n_layers = 4, 4
    encoder = VitModel(512, 512, 8, 4)
    example_mask = [[False]*500 + [True]*12]*50
    example_mask = torch.tensor(example_mask)
    example_y = encoder(example_x, example_mask)
    print(example_y.shape)


#class VitTransformerEncoder(nn.Module):
