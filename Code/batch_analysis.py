import os 
import torch
from utils import *
import pickle
import copy

FOLDER_NAME = "moe_1PS"
SC_FILE_NAME = "SC_test_370.pt"
FC_FILE_NAME = "FC_test_370.pt"
GENE_FILE_NAME = "genetics_test_370.pt"

device = 'cpu'
torch.set_default_device(device)
SC_data, FC_data, genetic_data = load_data(device=device, SC=SC_FILE_NAME, FC=FC_FILE_NAME, GNE=GENE_FILE_NAME)
SC_data, FC_data = preprocess_data(SC_data, FC_data)
n = len(SC_data)

PATH = os.path.join(".", "Saved_Models", FOLDER_NAME)
results = dict()
for model_name in os.listdir(PATH):
    if model_name == "config.json": continue
    print("--", model_name, "--")
    model_tag = "_".join(model_name.split("_")[:-1])
    model_path = os.path.join(PATH, model_name)
    model = torch.load(model_path, weights_only=False, map_location=device)
    
    model.eval()
    FC_pred, topk_FC = model.predict_FC(SC_data, genetic_data)
    SC_pred, topk_SC = model.predict_SC(FC_data, genetic_data)
    FC_pred = FC_pred.detach()
    SC_pred = SC_pred.detach()
    
    measures = get_measurements(n, SC_data, FC_data, SC_pred, FC_pred)
    comp_measures = get_comp_measurements(n, SC_data, FC_data, SC_pred, FC_pred)
    measures.update(comp_measures)
    
    empty_measures = {
        "MSE_SC": [],
        "COS_SC": [],
        "SSIM_SC": [],
        "MSE_FC": [],
        "COS_FC": [],
        "SSIM_FC": [],
        "MAE_SC": [],
        "PCC_SC": [],
        "FID_SC": [],
        "MAE_FC": [],
        "PCC_FC": [],
        "FID_FC": [],
        "EXPERT_SC": dict(),
        "EXPERT_FC": dict()
    }
    
    print(measures, "\n")
    
    if model_tag not in results:
        results[model_tag] = copy.deepcopy(empty_measures)
    for key in measures:
        results[model_tag][key].append(measures[key])
    
    for idx in topk_FC.flatten().tolist():
        results[model_tag]["EXPERT_FC"][idx] = results[model_tag]["EXPERT_FC"].get(idx, 0) + 1
    for idx in topk_SC.flatten().tolist():
        results[model_tag]["EXPERT_SC"][idx] = results[model_tag]["EXPERT_SC"].get(idx, 0) + 1

print("--Final Results --", FOLDER_NAME, "--")
print(results)

dbfile = open(os.path.join(".", "Analysis", "{}_analysis_output".format(FOLDER_NAME)), 'wb')
pickle.dump(results, dbfile)                    
dbfile.close()
