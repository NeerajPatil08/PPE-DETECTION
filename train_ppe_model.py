#!/usr/bin/env python3
"""
Train a YOLO model for PPE detection using the downloaded dataset.
"""

import os
import argparse
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import shutil
import json
from datetime import datetime

def save_training_state(args, epoch, best_metrics, save_dir):
    """Save training state for resume functionality."""
    state = {
        'epoch': epoch,
        'args': vars(args),
        'best_metrics': best_metrics,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    state_file = checkpoint_dir / 'last_training_state.json'
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)

def load_training_state():
    """Load previous training state if it exists."""
    state_file = Path('checkpoints') / 'last_training_state.json'
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return None

def get_user_input(prompt, default=None):
    """Get user input with optional default value."""
    if default is not None:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO model for PPE detection')
    
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img', type=int, default=640,
                        help='Image size')
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='Initial weights path')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project name')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Optimizer (SGD, Adam, AdamW)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--save-period', type=int, default=10,
                        help='Save checkpoint every x epochs')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load previous training state if resuming
    previous_state = None
    if args.resume:
        previous_state = load_training_state()
        if previous_state:
            print("\n=== Resuming Previous Training ===")
            print(f"Previous training date: {previous_state['date']}")
            print(f"Completed epochs: {previous_state['epoch']}")
            print(f"Best metrics: {previous_state['best_metrics']}")
            print("================================\n")
            
            # Ask user about remaining epochs
            remaining_epochs = args.epochs - previous_state['epoch']
            if remaining_epochs > 0:
                print(f"Original training plan: {args.epochs} epochs")
                print(f"Completed epochs: {previous_state['epoch']}")
                print(f"Remaining epochs: {remaining_epochs}")
                
                while True:
                    response = get_user_input(
                        "Do you want to continue with the remaining epochs? (yes/no/custom)", 
                        default="yes"
                    ).lower()
                    
                    if response == "yes":
                        args.epochs = previous_state['epoch'] + remaining_epochs
                        break
                    elif response == "no":
                        print("Exiting training...")
                        return
                    elif response == "custom":
                        while True:
                            try:
                                custom_epochs = int(get_user_input(
                                    "Enter the number of additional epochs to train"
                                ))
                                if custom_epochs > 0:
                                    args.epochs = previous_state['epoch'] + custom_epochs
                                    break
                                else:
                                    print("Please enter a positive number.")
                            except ValueError:
                                print("Please enter a valid number.")
                        break
                    else:
                        print("Please enter 'yes', 'no', or 'custom'")
            
            # Update args with previous training settings
            for key, value in previous_state['args'].items():
                if key not in ['resume', 'epochs']:  # Don't override resume flag and epochs
                    setattr(args, key, value)
    
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA is available with {num_gpus} GPU(s)")
        print(f"Current device: {current_device} - {device_name}")
        
        # Set device to GPU if available
        if args.device == '':
            args.device = '0'  # Default to first GPU
        
        # Check if specified device is valid
        if args.device != 'cpu' and not args.device.isdigit() and not all(d.isdigit() for d in args.device.split(',')):
            print(f"Warning: Invalid device '{args.device}'. Using default device '0'.")
            args.device = '0'
    else:
        print("CUDA is not available. Training will use CPU, which may be slow.")
        args.device = 'cpu'
    
    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Data: {args.data}")
    print(f"Total epochs to run: {args.epochs}")
    if previous_state:
        print(f"Starting from epoch: {previous_state['epoch'] + 1}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Initial weights: {args.weights}")
    print(f"Device: {args.device}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Initial learning rate: {args.lr0}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Data augmentation: {'Yes' if args.augment else 'No'}")
    print(f"Resume training: {'Yes' if args.resume else 'No'}")
    print("=============================\n")
    
    # Confirm training configuration
    response = get_user_input("Do you want to proceed with this configuration? (yes/no)", default="yes").lower()
    if response != "yes":
        print("Training cancelled.")
        return
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    
    # Load model
    if args.resume and previous_state:
        # Look for the last checkpoint
        last_checkpoint = checkpoint_dir / 'last.pt'
        if last_checkpoint.exists():
            print(f"Loading checkpoint from {last_checkpoint}")
            model = YOLO(str(last_checkpoint))
            start_epoch = previous_state['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            model = YOLO(args.weights)
            start_epoch = 0
    else:
        model = YOLO(args.weights)
        start_epoch = 0
    
    # Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        augment=args.augment,
        resume=args.resume,
        exist_ok=True,  # Allow overwriting existing experiment
        save_period=args.save_period  # Save checkpoint every save_period epochs
    )
    
    # Save final state
    save_training_state(args, args.epochs, results.results_dict, args.project)
    
    # Print results
    print("\n=== Training Results ===")
    print(f"Best model saved at: {results.best}")
    print("========================")
    
    # Copy best model to models directory
    best_model_path = Path(results.best)
    if best_model_path.exists():
        dest_path = Path('models') / 'best_ppe_model.pt'
        shutil.copy(best_model_path, dest_path)
        print(f"Copied best model to {dest_path}")
        
        # Also save as a checkpoint
        checkpoint_path = checkpoint_dir / 'best.pt'
        shutil.copy(best_model_path, checkpoint_path)
        print(f"Saved best model checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main() 