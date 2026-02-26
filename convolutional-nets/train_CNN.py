from pathlib import Path
import argparse
from xml.parsers.expat import model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_CLASSES = 2

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # medium dropout to help with generalization
            nn.Linear(256, 64),  
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


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
    # adding a scheduler to help with training 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Set all the gradients to zero before the backward pass
            optimizer.zero_grad(set_to_none=True)
            # Send our images through the model to get the logits which are raw scores, then calculate the loss and backpropagate
            logits = model(images)
            # Sees how wrong our predictions are
            loss = criterion(logits, labels)
            # Send our loss through the backwards pass to calculate the gradients for all the parameters in our model
            loss.backward()
            # Updates the weights of our model (basically what we did when we did param.data -= eta * param.grad but now by optimizer)
            optimizer.step()
            
            # Add up our counters
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            pass
        
        acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch}, Loss: {running_loss/total:.4f}, Acc: {acc:.4f}")
        
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
