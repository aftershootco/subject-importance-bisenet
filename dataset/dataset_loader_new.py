# your_dataset_file.py
import os
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BinarySegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, num_classes=2, mode="train"):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes # Kept for consistency, though not used in one-hot
        self.mode = mode

        # Online transforms (augmentations) are kept
        # The expensive A.Resize is removed!
        if mode == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                    border_mode=0, interpolation=1, mask_interpolation=0, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else: # Validation/Test
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load preprocessed data directly as numpy arrays
        img = np.array(Image.open(img_path).convert("RGB"))

        # --- FORCE BINARY MASK: convert to grayscale and binarize (0 or 1) ---
        mask = np.array(Image.open(mask_path).convert("L"))  # single channel 0..255
        # robust binarization (works if masks are 0/255 or other scales)
        mask = (mask > 127).astype(np.uint8)

        # Apply online augmentations
        augmented = self.transform(image=img, mask=mask)
        img_tensor = augmented["image"]

        # Mask is now a tensor of shape [H, W] with values 0 or 1.
        mask_tensor = augmented["mask"].long()

        return img_tensor, mask_tensor



def load_dataset(folders, num_classes=2, val_size=0.2):
    image_paths, mask_paths = [], []

    for folder in folders:
        src_dir = os.path.join(folder, "src")
        masks_dir = os.path.join(folder, "masks")

        imgs = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
        for img_path in imgs:
            base = os.path.splitext(os.path.basename(img_path))[0]
            # The preprocessed masks are saved as .png
            mask_path = os.path.join(masks_dir, base + ".png")
            
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)

    if not image_paths:
        raise ValueError("No valid image-mask pairs found. Did you run the preprocessing script?")
        
    assert len(image_paths) == len(mask_paths), "Image/mask count mismatch!"

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=val_size, random_state=42
    )

    datasets = {
        "train": BinarySegDataset(train_imgs, train_masks, num_classes=num_classes, mode="train"),
        "val": BinarySegDataset(val_imgs, val_masks, num_classes=num_classes, mode="val"),
    }
    return datasets