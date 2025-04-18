import os
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from src.dataset_format import TCGADataset
import numpy as np

# Set your dataset root
root_dir = 'Exploration_of_UNet_architectures_on_brain_tumor_segmentation/kaggle_3m'

# Define transforms
image_transform = transforms.ToTensor()
mask_transform = transforms.ToTensor()
dataset = TCGADataset(root_dir=root_dir, transform=image_transform, mask_transform=mask_transform)
random_indices = random.sample(range(len(dataset)), 5)

# Collect the image and mask tensors for the 5 random samples
image_tensor_list = []
mask_tensor_list = []

for idx in random_indices:
    img, msk = dataset[idx]
    image_tensor_list.append(img)
    mask_tensor_list.append(msk)

# Ensure output directory exists
save_dir = 'Exploration_of_UNet_architectures_on_brain_tumor_segmentation'
os.makedirs(save_dir, exist_ok=True)

# Create a large plot with 5 triplets (each triplet is 3 images)
fig, axes = plt.subplots(5, 3, figsize=(12, 20))
fig.suptitle('Random 5 Samples - Original, Mask, Blended', fontsize=16)

# Loop through each image-mask pair
for idx, (image_tensor, mask_tensor) in enumerate(zip(image_tensor_list, mask_tensor_list)):
    image = image_tensor.squeeze().permute(1, 2, 0).numpy()
    mask = mask_tensor.squeeze().numpy()
    # Create blended image
    blended = image.copy()
    red_mask = np.zeros_like(image)
    red_mask[..., 0] = 1.0  # Red channel
    alpha = 0.4
    blended = np.where(mask[..., None] == 1, 
                       alpha * red_mask + (1 - alpha) * image, 
                       image)
    # Plot original, mask, and blended for the current sample
    axes[idx, 0].imshow(image)
    axes[idx, 0].set_title(f"Original Image {idx+1}")
    axes[idx, 0].axis('off')
    axes[idx, 1].imshow(mask, cmap='gray')
    axes[idx, 1].set_title(f"Segmentation Mask {idx+1}")
    axes[idx, 1].axis('off')
    axes[idx, 2].imshow(blended)
    axes[idx, 2].set_title(f"Image with Mask Overlay {idx+1}")
    axes[idx, 2].axis('off')

# Adjust layout and save the figure
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust to avoid title overlap
plt.savefig(os.path.join(save_dir, 'data_visual_image.png'))  # Save as PNG
plt.close()
