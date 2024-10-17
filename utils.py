import torch  # type: ignore
import os
from PIL import Image  # type: ignore
import numpy as np  # type: ignore
import torchvision.utils as vutils  # type: ignore
from torch.utils.data import Dataset  # type: ignore


class CustomImageDataset(Dataset):
    """A custom dataset class that wraps optimized images and labels."""

    def __init__(self, images, labels):
        """
        Args:
            images (Tensor): The optimized images.
            labels (Tensor): The corresponding labels for the images.
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Retrieves the image and label at the specified index."""
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def normalize_per_image(tensor):
    # Flatten the tensor along the channel, height, and width dimensions
    flat_tensor = tensor.view(tensor.size(0), -1)

    # Compute the min and max along the flattened dimension
    min_vals = flat_tensor.min(1, keepdim=True)[0]
    max_vals = flat_tensor.max(1, keepdim=True)[0]

    # Normalize the tensor
    tensor_norm = (tensor - min_vals.view(tensor.size(0), 1, 1, 1)) / (
        max_vals - min_vals
    ).view(tensor.size(0), 1, 1, 1)

    # Handle cases where the max and min are the same (e.g., flat images)
    tensor_norm[torch.isnan(tensor_norm)] = 0

    return tensor_norm


def visualize_and_save_tensor(
    adv_X, root="optimized_images/N=2", filename="adv", scale_factor=10
):
    # Ensure adv_X is normalized correctly for image data
    adv_X = (
        adv_X.clone().cpu().detach()
    )  # Detach and clone to avoid modifying the original tensor
    adv_X = adv_X.float()  # Convert to float
    adv_X = (adv_X - adv_X.min()) / (adv_X.max() - adv_X.min())

    if not os.path.exists(root):
        os.makedirs(root)
    np.save(os.path.join(root, filename + ".npy"), adv_X)
    # if adv_X.max() > 1:
    # adv_X /= 255.0  # Normalize to [0, 1] if the tensor is scaled between 0 and 255

    # Create a grid of images
    grid = vutils.make_grid(adv_X, nrow=10, padding=2, normalize=False)
    grid_np = grid.numpy().transpose((1, 2, 0))
    grid_img = Image.fromarray((grid_np * 255).astype("uint8"), "RGB")

    # Rescale the image
    original_size = grid_img.size
    new_size = (
        int(original_size[0] * scale_factor),
        int(original_size[1] * scale_factor),
    )
    grid_img = grid_img.resize(
        new_size, Image.BICUBIC
    )  # Using bicubic interpolation for a smoother result

    grid_img.save(os.path.join(root, filename + ".png"))
    print(f"Saved image grid as {filename}, with size {new_size}")


def calculate_l2_norms(original, perturbed):
    # Calculate the difference tensor
    difference = original - perturbed

    # Compute the L2 norm for each image in the batch
    l2_norms = torch.norm(
        difference, p=2, dim=(1, 2, 3)
    )  # Assuming images are in NCHW format

    return l2_norms.mean().item()


def accuracy(net, loader, device):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def get_data(retain_set, size=100, device="cuda", class_idx=None):
    images = []
    labels = []

    i = 0
    # Iterate over the first 100 indices
    while len(images) < size:
        image, label = retain_set[i]
        i += 1
        if class_idx is not None and label not in class_idx:
            continue
        images.append(image)
        labels.append(label)

    # Convert lists to tensors if necessary
    images = torch.stack(images)
    labels = torch.tensor(labels)
    print(labels)
    return images.to(device), labels.to(device)
