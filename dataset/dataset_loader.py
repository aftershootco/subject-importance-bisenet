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
    def __init__(self, image_paths, mask_paths, image_size=(256, 256), num_classes=2, mode="train"):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.num_classes = num_classes
        self.mode = mode

        if mode == "train":
            self.transform = A.Compose([
                A.Resize(*image_size, interpolation=Image.BILINEAR),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=0,
                    interpolation=1,
                    mask_interpolation=0,
                    p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*image_size, interpolation=Image.BILINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # --- MODIFIED LOGIC: Open with Pillow to check and fix sizes ---
        try:
            img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")

            # Check for size mismatch and resize mask if necessary
            if img_pil.size != mask_pil.size:
                # Resize mask to match image using NEAREST to preserve mask values
                mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)
            
            # Convert to numpy arrays for further processing
            img = np.array(img_pil)
            mask = np.array(mask_pil)

        except Exception as e:
            print(f"Error loading or resizing file: {img_path}. Skipping. Error: {e}")
            # Return None, which will be handled by a collate_fn if needed, or raise error
            # For simplicity, we can try to return the next valid item
            return self.__getitem__((idx + 1) % len(self))
        # --- END OF MODIFIED LOGIC ---

        # Normalize mask to binary {0, 1}
        mask = np.where(mask > 128, 1, 0).astype(np.uint8)

        # Apply Albumentations
        augmented = self.transform(image=img, mask=mask)
        img = augmented["image"]
        
        # One-hot encode the mask for the loss function
        mask = augmented["mask"].long()
        h, w = mask.shape
        one_hot_mask = torch.zeros(self.num_classes, h, w)
        one_hot_mask.scatter_(0, mask.unsqueeze(0), 1)

        return img, one_hot_mask


def load_dataset(folders, image_size=(256, 256), num_classes=2, val_size=0.2):
    image_paths, mask_paths = [], []

    for folder in folders:
        src = os.path.join(folder, "src")
        masks = os.path.join(folder, "masks")

        imgs = sorted(glob.glob(os.path.join(src, "*.jpg")))
        for img in imgs:
            base = os.path.splitext(os.path.basename(img))[0]
            mask = os.path.join(masks, base + ".png")
            
            # This logic correctly skips images with missing masks
            if os.path.exists(mask):
                image_paths.append(img)
                mask_paths.append(mask)

    if not image_paths:
        raise ValueError("No valid image-mask pairs found in the provided directories. Check your paths and file names.")
        
    assert len(image_paths) == len(mask_paths), "Image/mask count mismatch after filtering!"

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=val_size, random_state=42
    )

    datasets = {
        "train": BinarySegDataset(train_imgs, train_masks, image_size, num_classes=num_classes, mode="train"),
        "val": BinarySegDataset(val_imgs, val_masks, image_size, num_classes=num_classes, mode="val"),
    }
    return datasets