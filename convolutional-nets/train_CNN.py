from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    ''' Your code goes here'''
    pass


def neural_network_model():
    return SimpleCNN()


def create_dataloader(data_dir, batch_size=32, shuffle=True):
    data_path = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 84)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




def train_model(model, data_dir="numbers_dataset/train", epochs=100, batch_size=32, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = create_dataloader(data_dir, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            ''' Your code goes here'''
            pass

    return model


def test_model(model, data_dir="numbers_dataset/test", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = create_dataloader(data_dir, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    accuracy = correct / max(1, total)
    print(f"Test accuracy: {accuracy:.4f}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-model",
        type=str,
        default="False",
        help="Set to True to train the model before testing.",
    )
    args = parser.parse_args()

    train_model_flag = args.train_model.strip().lower() in {"1", "true", "yes", "y"}

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cnn.pt"
    
    if train_model_flag:
        model = neural_network_model()
        model = train_model(model)
        torch.save(model.state_dict(), model_path)

    loaded_model = neural_network_model()
    loaded_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    test_model(loaded_model, "numbers_dataset/test")


if __name__ == "__main__":
    main()
