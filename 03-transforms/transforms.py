import torch
import matplotlib.pyplot as plt


from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

def visualize_img(img, label, label_map):
    print("Label: {}".format(label_map[label]))
    plt.figure()
    plt.title(label_map[label])
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# Download training data from open datasets.
data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Get a sample from the training data.
img, label = data[0]
visualize_img(img, label, labels_map)

# Flip the image horizontally.
# flipped_img = torch.flip(img, dims=[2])
# visualize_img(flipped_img, label, labels_map)

# # Rotate the image 90 degrees counter-clockwise.
# # k=1 means 90 degrees clockwise
# # k=2 means 180 degrees
# # k=3 means 270 degrees
# # dims=[1, 2] means rotate the image along the height and width dimensions
print(f"Original shape: {img.shape}")
rotated_img = torch.rot90(img, k=3, dims=[1, 2])
# visualize_img(rotated_img, label, labels_map)

# Resize the image to 16x16 pixels.
resized_img = torch.nn.functional.interpolate(img.unsqueeze(0), size=16, mode="bicubic")
# visualize_img(resized_img.squeeze(), label, labels_map)

# # Crop the image to the center 14x14 pixels.
cropped_img = torch.nn.functional.interpolate(img.unsqueeze(0), size=14, mode="bicubic")
# visualize_img(cropped_img.squeeze(), label, labels_map)

# # pad the image with 10 pixels of zeros on each side.
padded_img = torch.nn.functional.pad(img.unsqueeze(0), pad=(10, 10, 10, 10), mode="constant", value=0)
# visualize_img(padded_img.squeeze(), label, labels_map)

# # Add random noise to the image.
noise = torch.randn(padded_img.shape) * 0.2
noisy_img = padded_img + noise
visualize_img(noisy_img, label, labels_map)

# # transforms.Compose() allows us to apply multiple transforms to the data.
# composed = transforms.Compose([
#         transforms.Normalize((0.5,), (0.5,)), # Normalizes the image with mean 0.5 and standard deviation 0.5.
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=45),
#     ]
# )
# img_transformed = composed(img)
# visualize_img(img_transformed, label, labels_map)

# # Apply the transforms to the dataset.
# transformed_dataset = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=False,
#     transform=composed
# )