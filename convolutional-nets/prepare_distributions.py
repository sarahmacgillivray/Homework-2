import os
import shutil
from prepare_data import prepare_numbers_dataset_in_range, prepare_mnist_data

# First, we need to load the MNIST dataset object to pass it into the function
_, test_dataset = prepare_mnist_data(show_samples=False)

# Define the ranges as (low_min, low_max, high_min, high_max)
ranges = [
    (1, 500, 501, 999),   # D1
    (100, 500, 501, 899), # D2
    (200, 500, 501, 799), # D3
    (300, 500, 501, 699), # D4
    (400, 500, 501, 599), # D5
    (450, 500, 501, 549)  # D6
]

base_path = "numbers_dataset/distribution_shift"

for i, (l_min, l_max, h_min, h_max) in enumerate(ranges, 1):
    print(f"Generating D{i}...")
    
    # Call with the specific positional arguments required by your prepare_data.py
    prepare_numbers_dataset_in_range(
        test_dataset,
        l_min, l_max, # low range limits
        h_min, h_max, # high range limits
        dataset_size=40,
        meta_seed=i, # Unique seed for each dataset
        output_dir=base_path,
        visualize_dataset=True
    )
    
    # Move the folder to a unique name
    new_path = f"numbers_dataset/D{i}"
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.rename(base_path, new_path)

print("All 6 datasets generated in numbers_dataset/D1 through D6!")