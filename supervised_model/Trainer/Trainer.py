from torch.utils.data import DataLoader

from Data_loading import *
from Data_loading.Data_loading import DataLoading
from Encoder import *
from lsfb_dataset import LSFBIsolConfig, LSFBIsolLandmarks

train_data = LSFBIsolLandmarks(LSFBIsolConfig(
    root="C:/Users/abassoma/Documents/Dataset/Lsfb_dataset/isol",
    split="train",
    n_labels=500,
    sequence_max_length=50
))

train_loader = DataLoading(train_data, 512)()
#test_loader  = DataLoading(test_data , 512)()

for batch in train_loader:
    f, t, mask = batch
    print(f.shape, t.shape, mask.shape)
    break
