# check_dataset.py
import os
import glob
from PIL import Image
from tqdm import tqdm

# --- CONFIGURE THIS ---
# Set this to the path of your parent data folder 
# (e.g., '/Users/dhairyarora/development/Data')
PARENT_DATA_DIR = '/Users/dhairyarora/development/Data'
# --------------------

def check_dataset_integrity(base_dir):
    """
    Scans through dataset folders to find image-mask pairs with
    mismatched dimensions or missing files.
    """
    print(f"🔍 Starting scan of '{base_dir}'...")
    
    # Find all subdirectories in the parent data folder
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    if not subfolders:
        print(f"❌ No subfolders found in '{base_dir}'. Please check the path.")
        return

    mismatched_files = []
    missing_masks = []

    for folder in subfolders:
        print(f"\nScanning subfolder: {os.path.basename(folder)}")
        src_dir = os.path.join(folder, "src")
        masks_dir = os.path.join(folder, "masks")

        if not os.path.exists(src_dir):
            print(f"  - Warning: 'src' folder not found in {folder}")
            continue
        if not os.path.exists(masks_dir):
            print(f"  - Warning: 'masks' folder not found in {folder}")
            continue

        # Get all .jpg images
        image_paths = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
        
        for img_path in tqdm(image_paths, desc=f"  - Checking {os.path.basename(folder)}"):
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(masks_dir, base_filename + ".png")

            # Check 1: Does the mask exist?
            if not os.path.exists(mask_path):
                missing_masks.append(img_path)
                continue

            # Check 2: Are the dimensions equal?
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size != mask.size:
                        mismatched_files.append({
                            "image": img_path,
                            "image_size": img.size,
                            "mask": mask_path,
                            "mask_size": mask.size,
                        })
            except Exception as e:
                print(f"\n🚨 Error reading file pair: {img_path} and {mask_path}. Error: {e}")

    print("\n" + "="*50)
    print("✅ Scan Complete!")
    print("="*50)

    if mismatched_files:
        print("\n❌ Found Mismatched Dimensions in the following files:")
        for item in mismatched_files:
            print(f"  - Image: {item['image']} (Size: {item['image_size']})")
            print(f"    Mask:  {item['mask']} (Size: {item['mask_size']})")
            print("-" * 20)
    else:
        print("\n👍 All image/mask pairs have matching dimensions.")

    if missing_masks:
        print("\n❌ Found images with Missing Masks:")
        for path in missing_masks:
            print(f"  - {path}")
    else:
        print("\n👍 All images have a corresponding mask.")
        
    if not mismatched_files and not missing_masks:
        print("\n🎉 Your dataset looks consistent!")


if __name__ == '__main__':
    if not os.path.isdir(PARENT_DATA_DIR):
        print(f"Error: The directory '{PARENT_DATA_DIR}' does not exist.")
        print("Please update the PARENT_DATA_DIR variable in the script.")
    else:
        check_dataset_integrity(PARENT_DATA_DIR)