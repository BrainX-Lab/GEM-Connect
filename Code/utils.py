import os
import torch
from scipy import stats
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import seaborn as sns

PATH = "."
PATH_DATA = os.path.join(PATH, "Data")
SC_FILE_NAME = "SC_train_370.pt"
FC_FILE_NAME = "FC_train_370.pt"
GENE_FILE_NAME = "genetics_train_370.pt"

# class for defining config parameters as object attributes
class configObject:
    def __init__(self, data):
        for key, value in data.items():
            # If the value is a dict, convert it recursively
            if isinstance(value, dict):
                value = configObject(value)
            setattr(self, key, value)

def get_config(config):
    config["MOE"]["dim_embed"] = config["CLIP"]["n_roi"] * config["CLIP"]["layer_sizes"][-1]
    config["CLIP"]["encoder_layer_sizes"] = config["CLIP"]["layer_sizes"]
    config["CLIP"]["decoder_layer_sizes"] = config["CLIP"]["layer_sizes"][::-1]
    return configObject(config)

def load_data(n_roi=148, device='cpu', SC=SC_FILE_NAME, FC=FC_FILE_NAME, GENE=GENE_FILE_NAME):
    SC_data = torch.load(os.path.join(PATH_DATA, SC), weights_only=False)    # [N, ROI, ROI]
    SC_data = SC_data.to(device=device)
    SC_data = SC_data[:,:n_roi,:n_roi]
    
    FC_data = torch.load(os.path.join(PATH_DATA, FC), weights_only=False)
    FC_data = FC_data.to(device=device)
    FC_data = FC_data[:,:n_roi,:n_roi]
    
    genetic_data = torch.load(os.path.join(PATH_DATA, GENE), weights_only=False)
    genetic_data = genetic_data.to(torch.float32).to(device=device)
    
    return SC_data, FC_data, genetic_data

def split_data(SC_data, FC_data, genetic_data, ratio=0.125, device='cpu'):
    dataset = TensorDataset(SC_data, FC_data, genetic_data)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1.0-ratio, ratio], generator=torch.Generator(device=device))
    return train_dataset, val_dataset

def normalize(input, dim=1, eps=1e-9):
    m = torch.mean(input, dim=dim, keepdim=True)
    s = torch.std(input, dim=dim, keepdim=True)
    return (input - m) / (s + eps) 

def preprocess_data(SC_data=None, FC_data=None):
    if SC_data is not None:
        SC_shape = SC_data.shape
        SC_data = SC_data.flatten(start_dim=1)
        
        SC_data = torch.log(SC_data+1)
        SC_data = normalize(SC_data)
        SC_data = SC_data.reshape(SC_shape)
        
    if FC_data is not None:
        FC_shape = FC_data.shape
        FC_data = FC_data.flatten(start_dim=1)
        FC_data = FC_data.reshape(FC_shape)
        
    return SC_data, FC_data

def pearson_loss(input, target, eps=1e-9):
    input_m = input - torch.mean(input) + eps
    target_m = target - torch.mean(target) + eps
    r_num = torch.sum(input_m * target_m)
    r_den = torch.sqrt(torch.sum(torch.pow(input_m, 2)) * torch.sum(torch.pow(target_m, 2)))
    loss = r_num / r_den
    return torch.pow(loss - 1, 2)

def cosine_similarities(input_SC, input_FC, normalize=True):
    # cosine similarities of embeddings (dot product of normalized embeddings: matrix_ij = i-th dot j-th)
    n_SC = nn.functional.normalize(input_SC.flatten(start_dim=1)) if normalize else input_SC.flatten(start_dim=1)
    n_FC = nn.functional.normalize(input_FC.flatten(start_dim=1)) if normalize else input_FC.flatten(start_dim=1)
    SC_sim = torch.matmul(n_SC, torch.t(n_SC))   
    FC_sim = torch.matmul(n_FC, torch.t(n_FC))
    cross_sim = torch.matmul(n_SC, torch.t(n_FC))
    return SC_sim, FC_sim, cross_sim

def get_measurements(n, batch_SC, batch_FC, pred_SC, pred_FC):
    def SSIM_preprocess(mat, eps=1e-6):
        ret = mat.detach()
        ret -= torch.min(ret)
        ret /= torch.max(ret) + eps
        return ret.numpy()
    
    def get_stats(mat):
        diag = torch.diagonal(mat)
        diag_m = torch.mean(diag)
        
        off_diag = mat.flatten()[1:].view(n-1, n+1)[:,:-1].flatten()
        off_diag_m = torch.mean(off_diag)
        return (diag_m.item(), off_diag_m.item())
    
    MSE  = torch.zeros((2, n, n))
    COS  = torch.zeros((2, n, n)) 
    SSIM = torch.zeros((2, n, n))
    
    for i in range(n):
        for j in range(n):
            MSE[0,i,j] = F.mse_loss(batch_SC[i], pred_SC[j]).item()
            COS[0,i,j] = F.cosine_similarity(batch_SC[i].flatten(), pred_SC[j].flatten(), dim=0).item()
            SSIM[0,i,j]= ssim(SSIM_preprocess(batch_SC[i]), SSIM_preprocess(pred_SC[j]), data_range=1.0)
                        
            MSE[1,i,j] = F.mse_loss(batch_FC[i], pred_FC[j]).item()
            COS[1,i,j] = F.cosine_similarity(batch_FC[i].flatten(), pred_FC[j].flatten(), dim=0).item()
            SSIM[1,i,j]= ssim(SSIM_preprocess(batch_FC[i]), SSIM_preprocess(pred_FC[j]), data_range=1.0)
    
    return {"MSE_SC": get_stats(MSE[0]), "COS_SC": get_stats(COS[0]), "SSIM_SC": get_stats(SSIM[0]), 
            "MSE_FC": get_stats(MSE[1]), "COS_FC": get_stats(COS[1]), "SSIM_FC": get_stats(SSIM[1])}

def get_comp_measurements(n, batch_SC, batch_FC, pred_SC, pred_FC):
    def norm(t):
        n = t - (torch.ones_like(t) * torch.min(t))
        n = n / torch.max(n)
        return torch.unsqueeze(n,1).repeat(1,3,1,1)
    
    MAE_SC = F.l1_loss(batch_SC, pred_SC)
    MAE_FC = F.l1_loss(batch_FC, pred_FC)
    
    PCC_SC = sum([stats.pearsonr(batch_SC[i].flatten(), pred_SC[i].flatten()).statistic for i in range(n)]) / n
    PCC_FC = sum([stats.pearsonr(batch_FC[i].flatten(), pred_FC[i].flatten()).statistic for i in range(n)]) / n
    
    # FID requires normalization to [0,1], expects images of (1,148,148)
    fid_SC = FrechetInceptionDistance(feature=64, input_img_size=(1, 148, 148), normalize=True)
    fid_SC.update(norm(batch_SC), real=True)
    fid_SC.update(norm(pred_SC), real=False)
    
    fid_FC = FrechetInceptionDistance(feature=64, input_img_size=(1, 148, 148), normalize=True)
    fid_FC.update(norm(batch_FC), real=True)
    fid_FC.update(norm(pred_FC), real=False)
    
    return {"MAE_SC": MAE_SC, "PCC_SC": PCC_SC, "FID_SC": fid_SC.compute(), 
            "MAE_FC": MAE_FC, "PCC_FC": PCC_FC, "FID_FC": fid_FC.compute()}

def display_similarities(CLIP_sim, SC_sim, FC_sim, cmap='viridis'):
    fig, ax = plt.subplots(1,3)
    sns.heatmap(CLIP_sim, square=True, cmap=cmap, ax=ax[0], cbar=True, vmin=0, vmax=1)
    sns.heatmap(SC_sim, square=True, cmap=cmap, ax=ax[1], cbar=True, vmin=0, vmax=1)
    sns.heatmap(FC_sim, square=True, cmap=cmap, ax=ax[2], cbar=True, vmin=0, vmax=1)
    ax[0].set_title('SC-FC')
    ax[1].set_title('SC-SC')
    ax[2].set_title('FC-FC')
    plt.savefig("img_similarities.png") 
    plt.show()

def display_img_batch(SC_data, FC_data, pred_SC, pred_FC, embed_SC, embed_FC, 
                      cmap='viridis'):
    height = len(SC_data)
    SC_max = torch.max(SC_data)
    SC_min = torch.min(SC_data)
    FC_max = torch.max(FC_data)
    FC_min = torch.min(FC_data)
    PSC_max = torch.max(pred_SC)
    PSC_min = torch.min(pred_SC)
    PFC_max = torch.max(pred_FC)
    PFC_min = torch.min(pred_FC)
    
    fig, ax = plt.subplots(height,4)
    for i in range(height):
        sns.heatmap(SC_data[i], square=True, cmap=cmap, ax=ax[i,0], cbar=True, vmin=SC_min, vmax=SC_max)
        sns.heatmap(pred_SC[i], square=True, cmap=cmap, ax=ax[i,1], cbar=True, vmin=SC_min, vmax=SC_max)     
        sns.heatmap(FC_data[i], square=True, cmap=cmap, ax=ax[i,2], cbar=True, vmin=FC_min, vmax=FC_max)
        sns.heatmap(pred_FC[i], square=True, cmap=cmap, ax=ax[i,3], cbar=True, vmin=FC_min, vmax=FC_max)   
             
    ax[0, 0].set_title('ground SC')
    ax[0, 1].set_title('predicted SC')
    ax[0, 2].set_title('ground FC')
    ax[0, 3].set_title('predicted FC')
    fig.set_size_inches(15, 19, forward=True)
    plt.subplots_adjust(left=0.1, right=0.9, 
                        top=0.9, bottom=0.1, 
                        wspace=0.3, hspace=0.3)
    plt.savefig("img_batch_result.png") 
    plt.show()
    