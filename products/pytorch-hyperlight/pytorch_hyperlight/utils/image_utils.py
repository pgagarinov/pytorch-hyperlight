import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def load_image_as_resized_tensor(image_path, image_size=None, use_crop=False):
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    if isinstance(image_size, list):
        orig_image_size = list(image.shape)[1:]
        scale_factor = max(
            max(
                [image_size[0] / orig_image_size[0], image_size[1] / orig_image_size[1]]
            ),
            1,
        )
        new_image_size = [int(sz * scale_factor) for sz in orig_image_size]
    else:
        new_image_size = image_size

    image = transforms.Resize(new_image_size)(image)
    if use_crop:
        image = transforms.CenterCrop(image_size)(image)
    return image


def show_image_tensors(
    image_tensor_list,
    title_list=None,
):
    n_images = len(image_tensor_list)
    plt.figure(figsize=(10, 5 * n_images))
    for i_image in range(0, n_images):
        image_tensor = image_tensor_list[i_image]
        plt.subplot(n_images, 1, i_image + 1)
        image_tensor = image_tensor.squeeze()
        image_tensor = image_tensor.permute(1, 2, 0)
        plt.imshow(image_tensor)
        plt.axis("off")
        if title_list:
            title = title_list[i_image]
            plt.title(title)
    plt.show()
