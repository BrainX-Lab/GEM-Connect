import torch
from utils import *
from torch import nn
from torch.utils.data import DataLoader

### This is ISP Loss in the paper (for one mode) ###
def selfsim_loss(input, target=None, k=1):      # input is similarity matrix
    n = input.shape[0]
    if target is None: target = torch.eye(n)
    target = torch.pow(target, k)
    return nn.functional.mse_loss(input, target)

### This is CPL Loss in the paper (for one mode) ###
def custom_decoder_loss(input, target):         # input is batch of predictions
    n = input.shape[0]
    loss = 0
    PCC_whole = 0.0
    self_norm = 0.0
    
    for i in range(n):
        PCC_whole += pearson_loss(input[i].flatten(), target[i].flatten())
        self_norm += torch.norm(input[i]-target[i],p='fro')     
    
    loss += PCC_whole / n
    loss += self_norm / n
    return loss

def eval(selfsim_loss_fn, decoder_loss_fn, k, alpha, pred_SC, pred_FC, embed_SC, embed_FC, batch_SC, batch_FC):
    batch_SC = batch_SC.float()
    batch_FC = batch_FC.float()
    
    # Get similarities
    clip_SC_sim, clip_FC_sim, clip_cross_sim   = cosine_similarities(embed_SC, embed_FC)
    truth_SC_sim, truth_FC_sim, _ =              cosine_similarities(batch_SC, batch_FC)
    
    ### This is IPCLIP Loss in the paper ###
        # L_ISP is selfsim_loss_fn (loss of prediction vs ground truth for the similarities of the same mode)
        # L_CMA is cross-entropy function (loss of prediction vs ground truth for the similarities across different modes). 
        # Note: nn.functional.cross_entropy has reduction mode = 'mean' by default, so it divides the total loss by batch_size 'm' automatically
    clip_cross_sim *= torch.exp(torch.tensor([0.07]))
    l_encoder_SC = alpha*selfsim_loss_fn(clip_SC_sim, target=truth_SC_sim, k=k) + nn.functional.cross_entropy(clip_cross_sim, torch.arange(clip_cross_sim.shape[0])) / 2          
    l_encoder_FC = alpha*selfsim_loss_fn(clip_FC_sim, target=truth_FC_sim, k=k) + nn.functional.cross_entropy(clip_cross_sim.T, torch.arange(clip_cross_sim.shape[0])) / 2
        
    ### This is CPL Loss in the paper ###
    l_decoder_SC = decoder_loss_fn(pred_SC, batch_SC)
    l_decoder_FC = decoder_loss_fn(pred_FC, batch_FC)
    
    return [l_encoder_SC, l_encoder_FC], [l_decoder_SC, l_decoder_FC]

def test(model, batch_size, dataset, encoder_loss_fn, decoder_loss_fn, k, alpha, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.Generator(device=device))
    loss_data = [0.0, 0.0]
    n_batches = 0
    
    model.eval()
    for batch_SC, batch_FC, batch_genetic in dataloader:
        pred_SC, pred_FC, embed_SC, embed_FC = model(batch_SC, batch_FC, batch_genetic)
        l_encoder_clip, l_decoder_gen = eval(encoder_loss_fn, decoder_loss_fn,
                                             k, alpha,
                                             pred_SC, pred_FC, 
                                             embed_SC, embed_FC,
                                             batch_SC, batch_FC)
        
        loss_data[0] += l_encoder_clip[0].item() + l_encoder_clip[1].item()
        loss_data[1] += l_decoder_gen[0].item() + l_decoder_gen[1].item()
        n_batches += 1
        
    return [x / n_batches for x in loss_data]

# Default values are whats used in the paper
def train(model, batch_size, dataset, optimizer, encoder_loss_fn, decoder_loss_fn, k=5.0, alpha=1.0, beta=0.2, device='cpu'):  
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.Generator(device=device))
    loss_data = [0.0, 0.0]
    n_batches = 0
    
    # passes through the model
    model.train()
    for batch_SC, batch_FC, batch_genetic in dataloader:
        pred_SC, pred_FC, embed_SC, embed_FC = model(batch_SC, batch_FC, batch_genetic)
        optimizer.zero_grad()
        l_encoder_clip, l_decoder_gen = eval(encoder_loss_fn, decoder_loss_fn,
                                             k, alpha,
                                             pred_SC, pred_FC, 
                                             embed_SC, embed_FC,
                                             batch_SC, batch_FC)
        
        loss_data[0] += l_encoder_clip[0].item() + l_encoder_clip[1].item()
        loss_data[1] += l_decoder_gen[0].item() + l_decoder_gen[1].item()
        n_batches += 1
        
        #update parameters
        loss_total = l_encoder_clip[0] + l_encoder_clip[1]
        loss_total += beta * (l_decoder_gen[0] + l_decoder_gen[1])
        loss_total.backward()
        optimizer.step()
        
    return [x / n_batches for x in loss_data]
