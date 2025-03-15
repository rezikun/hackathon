import argparse
import os
import yaml
from ultralytics import YOLO
from datetime import datetime


def make_paths_absolute(dataset_path):
    """Parses the dataset.yaml file and ensures image paths are absolute."""
    base_dir = os.path.dirname(os.path.abspath(dataset_path))

    with open(dataset_path, 'r') as f:
        dataset = yaml.safe_load(f)

    # Check if paths exist and fix them to absolute based on dataset.yaml's location
    for split in ['train', 'val', 'test']:
        if split in dataset:
            path = dataset[split]
            # If the path is relative, make it absolute
            if not os.path.isabs(path):
                dataset[split] = os.path.join(base_dir, path.replace("../", ""))  # Convert relative to absolute path

    # Write the corrected dataset.yaml with absolute paths
    with open(dataset_path, 'w') as f:
        yaml.dump(dataset, f)

    return dataset_path


def train(dataset_path, run_name, epochs, resume_from=None):
    model = YOLO('yolo11s.pt')

    # If no custom name is provided, use the current datetime
    if not run_name:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Make sure paths in dataset.yaml are absolute
    corrected_dataset_path = make_paths_absolute(dataset_path)

    # If resuming, make sure to specify the checkpoint path
    resume_checkpoint = None
    if resume_from:
        if os.path.isdir(resume_from):
            # Look for the latest checkpoint in the directory
            checkpoint_files = [f for f in os.listdir(resume_from) if f.endswith('.pt')]
            checkpoint_files.sort(reverse=True)  # Sort by filename to get the latest one
            if checkpoint_files:
                resume_checkpoint = os.path.join(resume_from, checkpoint_files[0])
            else:
                print("Warning: No checkpoints found in the specified directory.")
        else:
            resume_checkpoint = resume_from

    results = model.train(
        data=corrected_dataset_path,  # Use the corrected dataset path with absolute paths
        epochs=epochs,
        device=0,
        save_period=5,
        project="/content/gdrive/MyDrive/Runs",
        name=run_name,
        resume=resume_checkpoint  # Resume training from the checkpoint (if available)
    )
    print(results)

    # Evaluate the model's performance on the validation set
    results = model.val()
    print(results)

    # Export the model to ONNX format
    success = model.export(format="onnx")
    print(success)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO model with custom dataset and run name.")

    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset.yaml file.")
    parser.add_argument("--name", type=str, default="", help="Custom name for the training run.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training (default: 200).")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume training from a checkpoint.")

    args = parser.parse_args()

    train(args.dataset, args.name, args.epochs, args.resume_from)
