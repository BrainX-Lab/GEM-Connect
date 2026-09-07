from train import *
from model import Brain_CLIP
import utils
import copy
import json
import os

PATH = "."
MODEL_NAME = "moe_1PS"
TRIALS = 3

torch.cuda.set_device(1)        #NOTE: Crashes on single-GPU (device 0) or CPU-only environments. Better to use dynamic selection.
device = 'cuda'
torch.set_default_device(device)
print("device:", torch.get_default_device())

# data
SC_data, FC_data, genetic_data = load_data(device=device)
SC_data, FC_data = preprocess_data(SC_data, FC_data)
train_dataset, val_dataset = split_data(SC_data, FC_data, genetic_data, ratio=0.2, device=device)

# Default values are what are used in the paper
    # k: exponent on Truth Cosine Similarity
    # alpha: trade-off between cross-modality vs. same modality similarity losses
    # beta: trade-off between Brain-CLIP vs. regeneration/prediction losses
def train_model(config):
    # model parameters
    layer_sizes = config.CLIP.layer_sizes
    k = config.CLIP.k
    model = Brain_CLIP(config)
    
    # loss function parameters
    alpha = config.Train.alpha
    beta = config.Train.beta

    # training parameters
    batch_size = config.Train.batch_size    
    lr = config.Train.lr
    n_epochs = config.Train.max_epochs
    patience = config.Train.patience
    
    encoder_loss_fn = selfsim_loss
    decoder_loss_fn = custom_decoder_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    lr_steps = config.Train.lr_steps
    
    # print configs
    print("k:\t\t", k)
    print("alpha:\t\t", alpha)
    print("beta:\t\t", beta)
    print("lr:\t\t", lr)
    print("layer_sizes:\t", layer_sizes)

    epochs_without_improvement = 0
    lr_steps_cur = 0
    best_version = None
    min_loss_v = None
    for epoch in range(n_epochs):
        print("Epoch:", epoch)
        
        # train
        loss = train(model, batch_size=min(len(train_dataset), batch_size), 
                    dataset=train_dataset, 
                    optimizer=optimizer, 
                    encoder_loss_fn=encoder_loss_fn,
                    decoder_loss_fn=decoder_loss_fn,
                    k=k, alpha=alpha, beta=beta,
                    device=device
                    )
        
        if (patience and patience > 0):
            loss_v = test(model, batch_size=min(len(val_dataset), batch_size),
                          dataset=val_dataset,
                          encoder_loss_fn=encoder_loss_fn,
                          decoder_loss_fn=decoder_loss_fn,
                          k=k, alpha=alpha,
                          device=device
                          )
            temp = loss_v[0] + beta*loss_v[1]
            
            if not min_loss_v:
                min_loss_v = temp
                best_version = copy.deepcopy(model.state_dict())
            elif temp < min_loss_v: 
                epochs_without_improvement = 0
                min_loss_v = temp
                best_version = copy.deepcopy(model.state_dict())
            else: epochs_without_improvement += 1
            
            print("\t", loss_v[0], loss_v[1], epochs_without_improvement)
            
            if epochs_without_improvement >= patience:
                scheduler.step()
                lr_steps_cur += 1
                if lr_steps_cur >= lr_steps:
                    print("\t Training Finished")
                    break
                print("\tLR decayed to:", scheduler.get_last_lr())
                epochs_without_improvement = 0

    model.load_state_dict(best_version)
    return model

### ------------------------------------- ###
# make directory to save models
MODEL_SAVE_PATH = os.path.join(PATH, "Saved_Models", MODEL_NAME)
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# process configs
with open(os.path.join(PATH, "config.json"), 'r') as f:
    config_dict = json.load(f)
    config = utils.get_config(config_dict)
    with open(os.path.join(MODEL_SAVE_PATH, "config.json"), 'w+') as c:
        json.dump(config_dict, c)

#
for trial_idx in range(TRIALS):      
    print("-----------------------TRAINING NEW MODEL-----------------------")
    print("model:\t", MODEL_NAME)
    print("trial:\t", trial_idx)
    
    model = train_model(config)
    torch.save(model, os.path.join(MODEL_SAVE_PATH, "trial_{}".format(trial_idx)))
    
    print("\n\n\n")