'''
 Author: Dr. Souradeep Dutta
 
 This code is provided for demonstration and instructional purposes
 for CPEN 355 at the University of British Columbia (UBC).
 
 Course website:
 https://souradeep-dutta-01.github.io/ubc-cpen-355-website
 
'''

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


def prepare_mnist_data(data_dir="data", num_samples=8, show_samples=True):
    data_path = Path(data_dir)
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform
    )

    if show_samples:
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.6, 2))
        for idx in range(num_samples):
            image, label = train_dataset[idx]
            axes[idx].imshow(image.squeeze(0), cmap="gray")
            axes[idx].set_title(str(label))
            axes[idx].axis("off")

        plt.tight_layout()
        plt.show()

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    return train_dataset, test_dataset


def collate_images(dataset, number=1, num_digits=2, seed=0):
    if not isinstance(number, int) or not isinstance(num_digits, int):
        raise ValueError("number and num_digits must be integers.")
    if number <= 0 or num_digits <= 0:
        raise ValueError("number and num_digits must be positive.")
    if number >= 10 ** num_digits:
        raise ValueError(f"number must be between 0 and {10 ** num_digits - 1}.")

    digits = f"{number:0{num_digits}d}"
    rng = np.random.default_rng(seed)

    images = []
    for digit_char in digits:
        digit = int(digit_char)
        idxs = [idx for idx, (_, label) in enumerate(dataset) if label == digit]
        if not idxs:
            raise ValueError("Could not find samples for the requested digits.")
        idx = int(rng.choice(idxs))
        img, _ = dataset[idx]
        images.append(img)

    for img in images[1:]:
        if img.shape != images[0].shape:
            raise ValueError("Images must have the same shape to collate.")

    return torch.cat(images, dim=2)


def prepare_thresholded_numbers_dataset(
    dataset,
    threshold_val,
    dataset_size,
    meta_seed,
    num_digits=None,
    output_dir="numbers_dataset",
    visualize_dataset=False,
):
    if not isinstance(threshold_val, int) or threshold_val <= 0:
        raise ValueError("threshold_val must be a positive integer.")
    if not isinstance(dataset_size, int) or dataset_size <= 0:
        raise ValueError("dataset_size must be a positive integer.")
    if dataset_size % 2 != 0:
        raise ValueError("dataset_size must be even.")
    if not isinstance(meta_seed, int):
        raise ValueError("meta_seed must be an integer.")

    if num_digits is None:
        num_digits = len(str(threshold_val))
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")

    max_number = 10 ** num_digits - 1
    if threshold_val <= 1 or threshold_val >= max_number:
        raise ValueError("threshold_val must allow numbers on both sides.")

    rng = np.random.default_rng(meta_seed)
    half = dataset_size // 2

    low_numbers = rng.integers(1, threshold_val, size=half)
    high_numbers = rng.integers(threshold_val + 1, max_number + 1, size=half)

    labels = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    class_dirs = {
        0: output_path / "class_0",
        1: output_path / "class_1",
    }
    for directory in class_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()

    sample_idx = 0
    for number in low_numbers:
        seed = int(rng.integers(0, 2 * dataset_size))
        image = collate_images(dataset, int(number), num_digits=num_digits, seed=seed)
        if visualize_dataset:
            to_pil(image).save(class_dirs[0] / f"sample_{sample_idx:05d}.jpg")
        labels.append(0)
        sample_idx += 1

    for number in high_numbers:
        seed = int(rng.integers(0, 2 * dataset_size))
        image = collate_images(dataset, int(number), num_digits=num_digits, seed=seed)
        if visualize_dataset:
            to_pil(image).save(class_dirs[1] / f"sample_{sample_idx:05d}.jpg")
        labels.append(1)
        sample_idx += 1

    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    return class_dirs, labels_tensor


def prepare_numbers_dataset_in_range(
    dataset,
    low_range_lower_limit,
    low_range_upper_limit,
    high_range_lower_limit,
    high_range_upper_limit,
    dataset_size,
    meta_seed,
    num_digits=None,
    output_dir="numbers_dataset",
    visualize_dataset=False,
):
    if not all(isinstance(v, int) for v in [
        low_range_lower_limit,
        low_range_upper_limit,
        high_range_lower_limit,
        high_range_upper_limit,
        dataset_size,
        meta_seed,
    ]):
        raise ValueError("Range limits, dataset_size, and meta_seed must be integers.")
    if any(v <= 0 for v in [
        low_range_lower_limit,
        low_range_upper_limit,
        high_range_lower_limit,
        high_range_upper_limit,
    ]):
        raise ValueError("Range limits must be positive.")
    if dataset_size <= 0 or dataset_size % 2 != 0:
        raise ValueError("dataset_size must be a positive even integer.")
    if low_range_lower_limit > low_range_upper_limit:
        raise ValueError("low range lower limit must be <= upper limit.")
    if high_range_lower_limit > high_range_upper_limit:
        raise ValueError("high range lower limit must be <= upper limit.")

    if num_digits is None:
        num_digits = max(
            len(str(low_range_upper_limit)),
            len(str(high_range_upper_limit)),
        )
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")

    max_number = 10 ** num_digits - 1
    if low_range_upper_limit > max_number or high_range_upper_limit > max_number:
        raise ValueError("Range limits must fit within num_digits.")

    rng = np.random.default_rng(meta_seed)
    half = dataset_size // 2

    low_numbers = rng.integers(low_range_lower_limit, low_range_upper_limit + 1, size=half)
    high_numbers = rng.integers(high_range_lower_limit, high_range_upper_limit + 1, size=half)

    labels = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    class_dirs = {
        0: output_path / "class_0",
        1: output_path / "class_1",
    }
    for directory in class_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()

    sample_idx = 0
    for number in low_numbers:
        seed = int(rng.integers(0, 2 * dataset_size))
        image = collate_images(dataset, int(number), num_digits=num_digits, seed=seed)
        if visualize_dataset:
            to_pil(image).save(class_dirs[0] / f"sample_{sample_idx:05d}.jpg")
        labels.append(0)
        sample_idx += 1

    for number in high_numbers:
        seed = int(rng.integers(0, 2 * dataset_size))
        image = collate_images(dataset, int(number), num_digits=num_digits, seed=seed)
        if visualize_dataset:
            to_pil(image).save(class_dirs[1] / f"sample_{sample_idx:05d}.jpg")
        labels.append(1)
        sample_idx += 1

    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    return class_dirs, labels_tensor


def main():
    test_collated_image = False 
    prepare_numbers_train_dataset = True 
    prepare_numbers_test_dataset = True
    prepare_numbers_range_dataset = True

    if test_collated_image :
        train_dataset, _ = prepare_mnist_data(show_samples=False)
        combined = collate_images(train_dataset, 509, 3, seed = 1)
        plt.figure(figsize=(4, 2))
        plt.imshow(combined.squeeze(0), cmap="gray")
        plt.axis("off")
        plt.title("Collated MNIST image")
        plt.tight_layout()
        plt.savefig("sample_image.png", dpi=150)

    if prepare_numbers_train_dataset : 
        train_dataset, _ = prepare_mnist_data(show_samples=False)
        class_dirs, labels = prepare_thresholded_numbers_dataset(
            train_dataset,
            threshold_val=500,
            dataset_size=100,
            meta_seed=0,
            output_dir="numbers_dataset/train",
            visualize_dataset=True
        )
        print("Thresholded dataset images:", class_dirs)
        print("Thresholded dataset labels:", labels.shape)

    if prepare_numbers_test_dataset:
        _, test_dataset = prepare_mnist_data(show_samples=False)
        class_dirs, labels = prepare_thresholded_numbers_dataset(
            test_dataset,
            threshold_val=500,
            dataset_size=40,
            meta_seed=1,
            output_dir="numbers_dataset/test",
            visualize_dataset=True
        )
        print("Thresholded test images:", class_dirs)
        print("Thresholded test labels:", labels.shape)

    if prepare_numbers_range_dataset:
        train_dataset, _ = prepare_mnist_data(show_samples=False)
        class_dirs, labels = prepare_numbers_dataset_in_range(
            train_dataset,
            low_range_lower_limit=100,
            low_range_upper_limit=199,
            high_range_lower_limit=800,
            high_range_upper_limit=899,
            dataset_size=40,
            meta_seed=2,
            output_dir="numbers_dataset/distribution_shift",
            visualize_dataset=True,
        )
        print("Range dataset images:", class_dirs)
        print("Range dataset labels:", labels.shape)


if __name__ == "__main__":
    main()
