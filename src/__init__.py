import os

# Define the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the path to the data directory outside of src
data_dir = os.path.join(project_root, 'data')

# List of subfolders to create
subfolders = ['raw', 'processed', 'outputs', 'interim']

# Create subfolders if they don't exist
for folder in subfolders:
    folder_path = os.path.join(data_dir, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
