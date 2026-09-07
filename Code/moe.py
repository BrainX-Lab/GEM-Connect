import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

class simple_cosine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha_ = config.cons_alpha
        self.temperature = 0.78
        
    def forward(self, A, B):
        # shape (B, max_patch)
        cos_sim = F.cosine_similarity(A, B, dim=-1)
        
        # temperature control
        scaled_cos_sim = cos_sim / (1 + self.temperature)
    
        # mean
        avg_scaled_cos_sim = scaled_cos_sim.mean()
        
        return self.alpha_ * avg_scaled_cos_sim

class GeneMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.n_routed_experts = config.num_experts

        self.alpha = config.aux_loss_alpha
        self.gating_temp = config.GE_temperature # < 1 supposed to be

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim_embed
        self.gene_hide_dim = config.dim_genetic
        self.weight_token = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.weight_gene = nn.Parameter(torch.empty((self.n_routed_experts, self.gene_hide_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight_token, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_gene, a=math.sqrt(5))
    
    def forward(self, hidden_states, gene_vectors):
        ### compute gating score
        # hidden_states has shape [batch_size, n_roi * dim_embed]   
        # gene_vectors  has shape [batch_size, dim_genetic]                
        logits_h = F.linear(hidden_states, self.weight_token, None)
        logits_g = F.linear(gene_vectors, self.weight_gene, None)
        logits = (logits_h + logits_g / self.gating_temp) / (1 + 1/self.gating_temp)
        scores = logits.softmax(dim=-1)
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        #if self.training and self.alpha > 0.0:
        if self.alpha > 0.0:
            scores_for_aux = scores
            # always compute aux loss based on the naive greedy topk method (below is the seq_aux = false case)
            mask_ce = F.one_hot(topk_idx, num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

# THIS IS A FFN_GLU https://arxiv.org/pdf/2002.05202
class DMLP(nn.Module):
    def __init__(self, config, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim_embed if hidden_size is None else hidden_size
        self.intermediate_size = config.expert_intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.LeakyReLU() # ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class MyMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.top_k
        self.experts = nn.ModuleList([DMLP(config, intermediate_size = config.expert_intermediate_size) for i in range(config.num_experts)])
        self.gate = GeneMoEGate(config)
        if config.num_shared_experts is not None:
            intermediate_size = config.expert_intermediate_size * config.num_shared_experts
            self.shared_experts = DMLP(config=config, intermediate_size = intermediate_size)
            
        self.cos = simple_cosine(config)
    
    # 'hidden_states' are embeddings [batch_size, n_roi, dim_embed]
    # 'g' are gene vectors           [batch_size, dim_genetic]
    def forward(self, hidden_states, g):
        identity = hidden_states
        orig_shape = hidden_states.shape                                                    # [batch_size, n_roi, dim_embed]
        hidden_states = hidden_states.flatten(start_dim=1)                                  # [batch_size, n_roi * dim_embed]
        
        # gating function (genetic + embedding)
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states, g)                       # [batch_size, top_k]   
        flat_topk_idx = topk_idx.view(-1)                                                   # [batch_size, top_k] -> [batch_size * top_k]          
        
        hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)    # [batch_size * top_k, n_roi * dim_embed]
        y = torch.empty_like(hidden_states)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)         # [batch_size, n_roi * dim_embed]
        
        if self.training: 
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        
        if self.config.num_shared_experts:
            p = self.shared_experts(identity.flatten(start_dim=1))
            y = y + p
        
        # Cosine loss head MoE MLP vs Shared MLP
        if self.training:
            cLoss = self.cos(y, p) 
            y = AddAuxiliaryLoss.apply(y, cLoss)
            
        y =  y.view(*orig_shape)                                                            # [batch_size, n_roi, dim_embed]
        return y, topk_idx