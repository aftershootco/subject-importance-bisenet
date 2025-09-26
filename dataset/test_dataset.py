from torch.utils.data import DataLoader
from dataset_loader import load_dataset

folders = ["/Users/dhairyarora/development/Data/bg-remove-photoroom-7k", "/Users/dhairyarora/development/Data/photoroom_dataset", "/Users/dhairyarora/development/Data/photoroom_dataset_2"]
datasets = load_dataset(folders, image_size=(256, 256))

train_loader = DataLoader(datasets["train"], batch_size=2, shuffle=True)

for imgs, masks in train_loader:
    print("Image batch:", imgs.shape)   # [B,3,H,W]
    print("Mask batch:", masks.shape)   # [B,1,H,W]
    print("Unique mask values:", masks.unique())
    break