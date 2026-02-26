import os
import torch
import matplotlib.pyplot as plt
from train_CNN import neural_network_model, create_dataloader, NUM_CLASSES

def evaluate_on_folder(model, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load our data from the right directory 
    loader = create_dataloader(data_dir, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return (correct / total) * 100 if total > 0 else 0

# Main function to run for this file 
# Load the trained model
model_path = "models/cnn.pt"
model = neural_network_model()

# Make sure the architecture in neural_network_model() matches what you trained!
model.load_state_dict(torch.load(model_path, map_location="cpu"))

# Loop through D1 to D6 and collect accuracies
dataset_names = [f"D{i}" for i in range(1, 7)]
accuracies = []

print("Starting Stress Test")
for name in dataset_names:
    path = f"numbers_dataset/{name}"
    if os.path.exists(path):
        acc = evaluate_on_folder(model, path)
        accuracies.append(acc)
        print(f"Accuracy for {name}: {acc:.2f}%")
    else:
        print(f"Warning: {path} not found!")
        accuracies.append(0)

# Create the Bar Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(dataset_names, accuracies, color='skyblue', edgecolor='navy')

# Formatting for the deliverable
plt.xlabel('Dataset Range Configuration')
plt.ylabel('Test Accuracy (%)')
plt.title('CNN Performance under Distribution Shift (D1 - D6)')
plt.ylim(0, 105) # Give some space for labels

# Add accuracy labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("distribution_shift.png")
plt.show()

print("Plot saved as distribution_shift.png")