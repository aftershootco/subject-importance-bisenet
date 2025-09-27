# preprocess.py
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

def preprocess_data(base_dir, output_dir, image_size):
    """
    Preprocesses images and masks and saves them to a new directory.

    - Ensures masks match image sizes.
    - Resizes both images and masks to the target `image_size`.
    - Binarizes masks to {0, 1}.
    """
    print(f"Starting preprocessing...")
    print(f"Input data directory: {base_dir}")
    print(f"Output directory for processed data: {output_dir}")
    print(f"Target image size: {image_size}")

    # Find all subdirectories (e.g., folder1, folder2, etc.)
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    if not folders:
        raise ValueError(f"No subdirectories found in {base_dir}. Expected 'src' and 'masks' folders inside subdirectories.")

    for folder in folders:
        src_path = os.path.join(base_dir, folder, "src")
        masks_path = os.path.join(base_dir, folder, "masks")

        # Create corresponding output directories
        output_src_path = os.path.join(output_dir, folder, "src")
        output_masks_path = os.path.join(output_dir, folder, "masks")
        os.makedirs(output_src_path, exist_ok=True)
        os.makedirs(output_masks_path, exist_ok=True)

        image_files = sorted(glob.glob(os.path.join(src_path, "*.jpg")))
        
        print(f"\nProcessing folder: {folder} ({len(image_files)} images)")
        for img_path in tqdm(image_files, desc=f"Preprocessing {folder}"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(masks_path, base_name + ".png")

            if not os.path.exists(mask_path):
                continue

            try:
                # --- 1. Load images ---
                img_pil = Image.open(img_path).convert("RGB")
                mask_pil = Image.open(mask_path).convert("L")

                # --- 2. Fix size mismatch (if any) ---
                if img_pil.size != mask_pil.size:
                    mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)

                # --- 3. Resize both to target size ---
                img_resized = img_pil.resize(image_size, Image.BILINEAR)
                mask_resized = mask_pil.resize(image_size, Image.NEAREST)

                # --- 4. Binarize mask ---
                mask_np = np.array(mask_resized)
                # mask_binary = np.where(mask_np > 128, 1, 0).astype(np.uint8)
                mask_final_pil = Image.fromarray(mask_np)

                # --- 5. Save processed files ---
                output_img_filepath = os.path.join(output_src_path, base_name + ".jpg")
                output_mask_filepath = os.path.join(output_masks_path, base_name + ".png")
                
                img_resized.save(output_img_filepath)
                # Save mask as PNG to preserve single-channel 8-bit format
                mask_final_pil.save(output_mask_filepath)

            except Exception as e:
                print(f"\nCould not process file {img_path}. Error: {e}. Skipping.")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline preprocessing for segmentation dataset.")
    parser.add_argument("--data", type=str, default="test", help="Path to the root data directory.")
    parser.add_argument("--output_dir", type=str, default="processed", help="Path to save the processed data.")
    parser.add_argument("--img_height", type=int, default=256, help="Target image height.")
    parser.add_argument("--img_width", type=int, default=256, help="Target image width.")
    args = parser.parse_args()

    target_size = (args.img_width, args.img_height)
    preprocess_data(args.data, args.output_dir, target_size)