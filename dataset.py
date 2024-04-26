import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from maskgenerator import MaskGenerator

import numpy as np


class TerrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)
        self.mask_generator = MaskGenerator(
            64, 64, channels=1, rand_seed=None, filepath=None
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        
        mask = torch.permute(torch.from_numpy(self.mask_generator.sample()), (2,0,1)).to(torch.float32)
        
        maskedimage = image * mask
        
        # print(f"The mask has type {mask.dtype}")
        # print(f"The image has type {image.dtype}")
        # print(f"The masked image has type {maskedimage.dtype}")
        
        maskedimage.requires_grad = True
        mask.requires_grad = True
        image.requires_grad = True
        
        # print(f"The mask has shape {mask.shape}")
        # print(f"The image has shape {image.shape}")
        # print(f"The masked image has shape {maskedimage.shape}")

        return maskedimage, mask, image  # image is ground truth


if __name__ == "__main__":

    # Check __getitem__ and the mask dimension

    print("Running dataset.py\n")
    data_dir = "./data/images/"

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Resize to a specific size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.5,], std=[0.5,]),  # Normalize the image
        ]
    )

    # Create an instance of the dataset
    dataset = TerrainDataset(root_dir="./data/images/", transform=transform)

    # Extract the image and label using the index
    maskedimage, mask, image = dataset[0]

    # Convert the image tensor to a NumPy array and transpose it to the correct shape
    image_np = np.transpose(maskedimage.detach().numpy(), (1, 2, 0))

    plt.imshow(image_np, cmap='gray')
    plt.title("Input Image")
    plt.axis("off")
    plt.show()
