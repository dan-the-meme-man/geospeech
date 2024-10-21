import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""HYPERPARAMETERS"""
lr = 1e-3
betas = (0.9, 0.98)
eps = 1e-6
weight_decay = 0.1
max_grad_norm = 1.0

d_model = 512
nhead = 8
num_layers = 6

max_length = 1024

# max grad norm
def clip_grad_norm(model):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

def get_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

class GeoWhisper(nn.Module):
    def __init__(self):
        super(GeoWhisper, self).__init__()
        self.conv1 = nn.Conv1d(3, 2, 5)
        self.conv2 = nn.Conv1d(2, 1, 5)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        
        self.register_buffer(
            'sinusoidal_positional_encoding',
            self._generate_sinusoidal_positional_encoding(max_length, d_model)
        )
        
        self.learned_positional_encoding = nn.Embedding(max_length, d_model)
        
    def _generate_sinusoidal_positional_encoding(self, max_length, d_model):
        encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, src, tgt):
        print('input and target', src.shape, tgt.shape)
        
        # convs
        src = F.gelu(self.conv1(src))
        print('1st conv', src.shape)
        src = F.gelu(self.conv2(src))
        print('2nd conv', src.shape)
        
        # reshape for transformer
        src = src.permute(2, 0, 1)
        seq_len, batch_size, _ = src.shape
        src += self.sinusoidal_positional_encoding[:seq_len, :].unsqueeze(1).expand(-1, batch_size, -1)
        print('src plus SPE', src.shape)
        
        # transformer encoder
        memory = self.transformer_encoder(src)
        print('enc out', memory.shape)
        
        # learned positional encoding
        tgt_seq_len, tgt_batch_size, _ = tgt.shape
        assert tgt_batch_size == batch_size, "Source and target batch sizes must match"
        tgt_pos = torch.arange(tgt_seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        tgt = tgt + self.learned_positional_encoding(tgt_pos).permute(1, 0, 2)
        print('tgt plus LPE', tgt.shape)
        
        # transformer decoder
        output = self.transformer_decoder(tgt, memory)
        print('dec out', output.shape)
        return output
    
    def preprocess(audio_file):
        pass