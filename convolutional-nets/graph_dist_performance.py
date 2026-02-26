import os
import torch
import matplotlib.pyplot as plt
from train_CNN import neural_network_model, test_model

model_path = "models/cnn.pt"
model = neural_network_model()
model.load_state_dict(torch.load(model_path, map_location="cpu"))

dataset_names = [f"D{i}" for i in range(1, 7)]
accuracies = []

# Loop and test model on each dataset and store accuracies
print("Starting Stress Test...")
for name in dataset_names:
    path = f"numbers_dataset/{name}"
    if os.path.exists(path):
        # Just use test model from train_CNN 
        acc = test_model(model, path)
        accuracies.append(acc)
    else:
        accuracies.append(0)
        
        
# Create the Bar Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(dataset_names, accuracies, color='skyblue', edgecolor='navy')

# Adding the percentages on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', 
             ha='center', va='bottom', fontweight='bold')

plt.xlabel('Dataset Range Configuration')
plt.ylabel('Test Accuracy (%)')
plt.title('CNN Performance under Distribution Shift')
plt.ylim(0, 110) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and Show
plt.savefig("distribution_shift.png")
plt.show()
print("Graph saved as distribution_shift.png")