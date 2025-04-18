import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TCGADataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_mask_pairs = []

        # loop over each case folder
        for case_folder in os.listdir(root_dir):
            case_path = os.path.join(root_dir, case_folder)
            if not os.path.isdir(case_path):
                continue

            # load all .tif images that are not masks
            images = glob.glob(os.path.join(case_path, '*.tif'))
            for img_path in images:
                if img_path.endswith('_mask.tif'):
                    continue
                # get the matching mask path
                mask_path = img_path.replace('.tif', '_mask.tif')
                if os.path.exists(mask_path):
                    self.image_mask_pairs.append((img_path, mask_path))

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.image_mask_pairs[index]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # mask is usually single channel

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
