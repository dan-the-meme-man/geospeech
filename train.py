import torch

from model import clip_grad_norm, get_optimizer

def train(model):
    model.train()
    optimizer = get_optimizer(model)
    for epoch in range(10):
        for i in range(100):
            src = torch.randn(10, 3, 512)
            tgt = torch.randn(10, 1, 512)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = F.mse_loss(output, tgt)
            loss.backward()
            clip_grad_norm(model)
            optimizer.step()
            print(loss.item())