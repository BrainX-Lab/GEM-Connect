# GEM-Connect Documentation

## Files
| File | Purpose |
| :-- | :-- |
| [`config.json`](config.json) | Training & model hyperparameters |
| [`batch_train.py`](./Code/batch_train.py) | Main execution script for training |
| [`batch_analysis.py`](./Code/batch_analysis.py) | Post-training evaluation script |
| [`moe.py`](./Code/moe.py) | Mixture-of-Experts module |
| [`model.py`](./Code/model.py) | Overall model architecture |
| [`BNT_modules.py`](./Code/BNT_modules.py) | Transformer building blocks |
| [`train.py`](./Code/train.py) | Loss objectives & single-epoch training/validation |
| [`utils.py`](./Code/utils.py) | Utilities for data I/O, config parsing, and visualization |

## Configurable Parameters

### Encoder/Decoder
| Field | Value | Usage |
| :--   | :--   | :--   |
| `n_roi` | 148 | Number of brain regions of interest (ROIs) <br> The brain input for each subject is an `n_roi` $\times$ `n_roi` matrix|
| `num_heads` | 4 | Number of attention heads in multi-head attention |
| `layer_sizes` | 128,64,32 | In the encoder, each ROI feature is down-projected from $148\rightarrow 128 \rightarrow 64 \rightarrow 32$ <br> In the decoder, each ROI feature is up-projected from $32 \rightarrow 64 \rightarrow 128 \rightarrow 148$|
| `k` | 5 | Power exponent for cross-subject similarity matrix <br> Higher values sharpens cross-subject differences |

### Genetic Mixture-of-Experts
| Field | Value | Usage |
| :--   | :--   | :--   |
| `dim_genetic` | 834 | The genetic input for each subject is a vector of this length |
| `num_experts` | 8 | Number of routed experts |
| `top_k` | 2 | Number of active routed experts per sample |
| `num_shared_experts` | 1 | Number of always-active experts <br> Does not take away from `top_k` |
| `expert_intermediate_size` | 2368 | Hidden layer width of expert MLPs <br> $ (148\times 32)\rightarrow 2368\rightarrow (148\times 32)$ |

### Training
| Field | Value | Usage |
| :--   | :--   | :--   |
| `batch_size` | 32 | Mini-batch size |
| `alpha` | 1.0 | Encoder intra-modality vs. cross-modality loss balancing factor |
| `beta` | 0.2 | Decoder (reconstruction) loss weight |
| `lr` | 0.0001 | Initial learning rate |
| `lr_steps` | 2 | How many times to decrease the learning rate
| `patience` | 30 | Early-stopping threshold <br> How many epochs without improvement before lowering the learning rate or stopping |
| `max_epochs` | 10000 | Maximum epoch ceiling (training terminates much earlier) | 
