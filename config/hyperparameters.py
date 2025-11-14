
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 100
batch_size = 64
k_folds = 5
learning_rate = 0.001
dropout_rate = 0.5
random_seed = 42


num_electrodes_eeg = 62
num_electrodes_fnirs = 18
time_points = 200
num_classes = 4


gat_num_heads = 8
gat_out_channels = 25
gat_hidden_dim = 256
transformer_nhead = 1
transformer_dim_feedforward = 512