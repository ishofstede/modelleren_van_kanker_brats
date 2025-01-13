import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import os

scaler = MinMaxScaler()

def load_and_normalize_image(file_path):
    """
    Load and normalize a 3D medical image using MinMaxScaler.
    """
    image = nib.load(file_path).get_fdata()
    normalized_image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return normalized_image

def load_and_process_mask(file_path):
    """
    Load and process a segmentation mask. Reassign label 4 to 3 and convert to integer type.
    """
    mask = nib.load(file_path).get_fdata().astype(np.uint8)
    mask[mask == 4] = 3  # Reassign mask values 4 to 3
    return mask

def crop_volume(volume, crop_size=(128, 128, 128), offsets=(56, 56, 13)):
    """
    Crop a 3D volume to the specified size and offsets.
    """
    x, y, z = offsets
    dx, dy, dz = crop_size
    return volume[x:x+dx, y:y+dy, z:z+dz]

def save_volume_and_mask(image, mask, img_idx, save_path):
    """
    Save the combined image and mask as .npy files if it contains at least 1% labeled data.
    """
    val, counts = np.unique(mask, return_counts=True)
    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume
        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)

        # One-hot encode the mask
        mask = to_categorical(mask, num_classes=4)
        np.save(os.path.join(save_path, f'images/image_{img_idx}.npy'), image)
        np.save(os.path.join(save_path, f'masks/mask_{img_idx}.npy'), mask)
        print(f"Saved: image_{img_idx}.npy and mask_{img_idx}.npy")
    else:
        print(f"Ignored: image_{img_idx} due to insufficient labeled data")

def process_and_save_images(image_paths, mask_paths=None, save_path=None, crop_size=(128, 128, 128), offsets=(56, 56, 13)):
    """
    Process and save a list of images and their corresponding masks (if available).
    """
    for img_idx in range(len(image_paths)):
        print(f"Processing image {img_idx + 1}/{len(image_paths)}")
        
        # Load and normalize images
        t2_image = load_and_normalize_image(image_paths[img_idx])
        t1ce_image = load_and_normalize_image(image_paths[img_idx])  # Adjust for available modalities
        flair_image = load_and_normalize_image(image_paths[img_idx])
        
        # Combine channels
        combined_image = np.stack([flair_image, t1ce_image, t2_image], axis=-1)
        
        # Crop image
        cropped_image = crop_volume(combined_image, crop_size, offsets)
        
        if mask_paths:
            # Load and process mask if masks are provided
            mask = load_and_process_mask(mask_paths[img_idx])
            cropped_mask = crop_volume(mask, crop_size, offsets)
            
            # Save image and mask
            save_volume_and_mask(cropped_image, cropped_mask, img_idx, save_path)
        else:
            # Save only the image if no masks are available
            os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
            np.save(os.path.join(save_path, f'images/image_{img_idx}.npy'), cropped_image)
            print(f"Saved: image_{img_idx}.npy (no mask)")

