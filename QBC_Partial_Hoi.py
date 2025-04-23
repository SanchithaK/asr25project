import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
import os
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import pandas as pd
import matplotlib.pyplot as plt
import heapq

import boto3

os.makedirs("results", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""## Data Class"""

class CellSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # (H, W)

        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize
        image = image.astype('float32') / 255.0
        mask = (mask > 0).astype('float32')  # Binary mask

        # Convert to CHW format for PyTorch
        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)
        mask = torch.tensor(mask).unsqueeze(0)    # (1, H, W)

        # Pad so that divisible by 32
        image = pad_to_multiple(image)
        mask = pad_to_multiple(mask)

        return image, mask, self.image_filenames[idx]

def pad_to_multiple(x, multiple=32):
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h))

def unpad_to_shape(x, original_h, original_w):
    return x[..., :original_h, :original_w]

"""## Load Data"""
train_ds = CellSegmentationDataset("../../Data/images_train", "../../Data/masks_train")
val_ds =  CellSegmentationDataset("../../Data/images_val", "../../Data/masks_val")
test_ds = CellSegmentationDataset("../../Data/images_test", "../../Data/masks_test")

# Do this on full dataset
"""train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)
test_loader = DataLoader(test_ds, batch_size=1)"""

"""## This is done cause of compute issues"""

# Randomly pick 10 indices from the training dataset
subset_indices = random.sample(range(len(train_ds)), 10)

# Create a subset dataset
train_subset = Subset(train_ds, subset_indices)

train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)

subset_indices = random.sample(range(len(test_ds)), 10)
test_subset = Subset(test_ds, subset_indices)
test_loader =  DataLoader(test_subset, batch_size=1)

"""## UNet Model Definition"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
    activation="sigmoid"
).to(device)

"""## Training"""

# loss_fn = smp.losses.DiceLoss(mode='binary')
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     for imgs, masks, _ in train_loader:
#         imgs, masks = imgs.to(device), masks.to(device)

#         preds = model(imgs)
#         loss = loss_fn(preds, masks)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

"""## Evaluation/Testing"""

# model.eval()
# test_dice = 0.0
# with torch.no_grad():
#     for img, mask, _ in test_loader:
#         img, mask = img.to(device), mask.to(device)
#         pred = model(img)
#         pred_bin = (pred > 0.5).float()

#         intersection = (pred_bin * mask).sum()
#         union = pred_bin.sum() + mask.sum()
#         dice = (2 * intersection) / (union + 1e-8)
#         test_dice += dice.item()

# print(f"Test Dice Score: {test_dice / len(test_loader):.4f}")

"""## Visualization of test predictions"""

# os.makedirs("results", exist_ok=True)

# def show_prediction(img, mask, filename, save=True):
#     model.eval()
#     with torch.no_grad():
#         pred = model(img.unsqueeze(0).to(device))
#         pred_bin = (pred > 0.5).float().squeeze().cpu().numpy()

#     pred_unpadded = unpad_to_shape(pred_bin, 520, 704)
#     img_unpadded = unpad_to_shape(img.squeeze(0), 520, 704)
#     mask_unpadded = unpad_to_shape(mask.squeeze(0), 520, 704)

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     axs[0].imshow(img_unpadded, cmap='gray')
#     axs[0].set_title("Input Image")
#     axs[0].axis('off')

#     axs[1].imshow(mask_unpadded, cmap='gray')
#     axs[1].set_title("Ground Truth")
#     axs[1].axis('off')

#     axs[2].imshow(pred_unpadded, cmap='gray')
#     axs[2].set_title("Predicted Mask")
#     axs[2].axis('off')

#     plt.tight_layout()

#     if save:
#         # Ensure filename is safe
#         base_name = os.path.splitext(os.path.basename(filename))[0]
#         save_path = f"results/{base_name}_prediction.png"
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"Saved prediction to {save_path}")
#     else:
#         plt.show()
#     plt.close(fig)

# for test in test_ds:
#     img, mask, fname = test
#     show_prediction(img, mask, filename=fname)

"""# Passive Learning Style vs QBC Training"""

# Cleaned version

def evaluate_model_on_subset(dataset, subset_indices, test_loader, epochs=5):
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=4, shuffle=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1, activation="sigmoid").to(device)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training
    model.train()
    for _ in range(epochs):
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Evaluation on training set after last epoch
    model.eval()
    train_dice_scores = []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            preds_bin = (preds > 0.5).float()
            intersection = (preds_bin * masks).sum()
            union = preds_bin.sum() + masks.sum()
            dice = (2 * intersection) / (union + 1e-8)
            train_dice_scores.append(dice.item())
    final_train_dice = np.mean(train_dice_scores)

    # Evaluation on test set
    model.eval()
    test_dice_scores = []
    with torch.no_grad():
        for img, mask, _ in test_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred_bin = (pred > 0.5).float()
            inter = (pred_bin * mask).sum()
            union = pred_bin.sum() + mask.sum()
            dice = (2 * inter) / (union + 1e-8)
            test_dice_scores.append(dice.item())
    final_test_dice = np.mean(test_dice_scores)
    return final_train_dice, final_test_dice

"""initial_size = 100
increment = 200"""
initial_size = 2
increment = 1
#max_size = int(0.5 * len(train_ds))
max_size = int(0.5 * len(train_subset))
n_simulations = 5

# I commented these 2 lines
#all_indices = list(range(len(train_ds)))
#dataset_sizes = list(range(initial_size, max_size + 1, increment))

all_indices = list(range(len(train_subset)))
dataset_sizes = list(range(initial_size, max_size + 1, increment))

# QBC Part:
def get_fisher_information_scores(model, dataset, unlabeled_indices):
    model.eval()
    fisher_scores = []
    loss_fn = torch.nn.BCELoss()  # Binary Cross Entropy Loss for binary segmentation

    for idx in unlabeled_indices:
        img, _, _ = dataset[idx]
        img = img.unsqueeze(0).to(device)
        img.requires_grad = True  # Make the image tensor require gradients

        # Get pseudo-label using the current model
        with torch.no_grad():
            pred = model(img)

        # Compute the loss between the prediction and the pseudo-label
        loss = loss_fn(pred, pred)

        # Backward pass to get gradients w.r.t. the model's output
        model.zero_grad()
        loss.backward()

        # Compute Fisher score as the sum of squared gradients for each parameter
        fisher_score = 0.0
        for param in model.parameters():
            if param.grad is not None:
                fisher_score += (param.grad ** 2).sum().item()  # Sum of squared gradients

        fisher_scores.append((fisher_score, idx))

    return fisher_scores


def get_qbc_scores(committee, dataset, unlabeled_indices):
    committee_preds = []
    
    for model in committee:
        model.eval()
        preds = []
        with torch.no_grad():
            for idx in unlabeled_indices:
                img, _, _ = dataset[idx]
                img = img.unsqueeze(0).to(device)
                pred = model(img).cpu().numpy()
                preds.append(pred.squeeze())
        committee_preds.append(np.array(preds))  # (N_unlabeled, H, W)

    committee_preds = np.stack(committee_preds, axis=0)  # (C, N, H, W)
    var_map = np.var(committee_preds, axis=0)  # (N, H, W)
    mean_variance = var_map.mean(axis=(1, 2))  # per sample
    return list(zip(mean_variance, unlabeled_indices))

def select_batch_using_fisher_and_qbc(committee, dataset, unlabeled_indices, batch_size, fisher_weight=1.0, qbc_weight=1.0):
    # Compute Fisher Information scores
    fisher_scores = []
    for model in committee:
        fisher_scores.extend(get_fisher_information_scores(model, dataset, unlabeled_indices))
    
    fisher_scores.sort(reverse=True)
    fisher_scores = {idx: score for score, idx in fisher_scores}

    # Compute QBC Disagreement scores
    qbc_scores = get_qbc_scores(committee, dataset, unlabeled_indices)
    qbc_scores = {idx: score for score, idx in qbc_scores}

    # Combine Fisher Information and QBC scores (weighted sum)
    combined_scores = []
    for idx in unlabeled_indices:
        fisher_score = fisher_scores.get(idx, 0)
        qbc_score = qbc_scores.get(idx, 0)
        combined_score = fisher_weight * fisher_score + qbc_weight * qbc_score
        combined_scores.append((combined_score, idx))
    
    # Sort based on combined score
    combined_scores.sort(reverse=True)
    
    # Select top `batch_size` samples
    selected = [idx for _, idx in combined_scores[:batch_size]]
    return selected

"""initial_size = 100
query_size = 100"""
initial_size = 2
query_size = 1
#max_size = int(0.5 * len(train_ds))
max_size = int(0.8 * len(train_subset))
n_simulations = 5

train_results, test_results = {}, {}

for sim in range(n_simulations):
    random.seed(sim)
    all_indices = list(range(len(train_ds)))
    random.shuffle(all_indices)
    labeled_indices = all_indices[:initial_size]
    unlabeled_indices = all_indices[initial_size:]

    while len(labeled_indices) <= max_size:
        print(f"Training on {len(labeled_indices)} samples...")

        train_dice, test_dice = evaluate_model_on_subset(train_ds, labeled_indices, test_loader)
        print(f" Train Dice = {train_dice:.4f}", f" Test Dice = {test_dice:.4f}")

        train_results.setdefault(len(labeled_indices), []).append(train_dice)
        test_results.setdefault(len(labeled_indices), []).append(test_dice)

        if len(labeled_indices) + query_size > max_size:
            break

        # Train committee models
        committee = []
        for _ in range(3):  # size of committee
            model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1, activation="sigmoid").to(device)
            loader = DataLoader(Subset(train_ds, labeled_indices), batch_size=4, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = smp.losses.DiceLoss(mode='binary')
            model.train()
            for epoch in range(3):  # small number of epochs
                for imgs, masks, _ in loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds = model(imgs)
                    loss = loss_fn(preds, masks)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            committee.append(model)

        # Select batch using combined Fisher and QBC scores
        selected = select_batch_using_fisher_and_qbc(committee, train_ds, unlabeled_indices, query_size)

        labeled_indices += selected
        unlabeled_indices = list(set(unlabeled_indices) - set(selected))
# End QBC Part


# Passive Learning Part
pl_train_results, pl_test_results = {}, {}
for sim in range(n_simulations):
    random.seed(sim)
    shuffled_indices = all_indices.copy()
    random.shuffle(shuffled_indices)
    for size in dataset_sizes:
        current_subset = shuffled_indices[:size]  # always includes previous ones
        print(f"  Training on {size} samples...", end="")
        #train_dice, test_dice = evaluate_model_on_subset(train_ds, current_subset, test_loader)
        train_dice, test_dice = evaluate_model_on_subset(train_subset, current_subset, test_loader)
        print(f" Train Dice = {train_dice:.4f}", f" Test Dice = {test_dice:.4f}")
        pl_train_results.setdefault(size, []).append(train_dice)
        pl_test_results.setdefault(size, []).append(test_dice)


# Function to plot train/test dice scores
def plot_train_test(train_results, test_results, dataset_sizes, title, filename_prefix, color_train, color_test):
    train_means = np.array([np.mean(train_results[s]) for s in dataset_sizes])
    train_stds = np.array([np.std(train_results[s]) for s in dataset_sizes])
    test_means = np.array([np.mean(test_results[s]) for s in dataset_sizes])
    test_stds = np.array([np.std(test_results[s]) for s in dataset_sizes])

    plt.figure(figsize=(8, 6))
    plt.plot(dataset_sizes, train_means, '-o', label='Train Dice', color=color_train)
    plt.plot(dataset_sizes, test_means, '-o', label='Test Dice', color=color_test)
    plt.fill_between(dataset_sizes, train_means - train_stds, train_means + train_stds, alpha=0.3, color=color_train)
    plt.fill_between(dataset_sizes, test_means - test_stds, test_means + test_stds, alpha=0.3, color=color_test)
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Dice Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"{filename_prefix}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")
    return filename

# Dataset sizes (must match keys in results dicts)
dataset_sizes = sorted(train_results.keys())

# === Generate and Save Plots ===
file1 = plot_train_test(train_results, test_results, dataset_sizes,
                        "QBC: Dice Scores vs Training Set Size",
                        "QBC_DiceScore_hoi", color_train="blue", color_test="orange")

file2 = plot_train_test(pl_train_results, pl_test_results, dataset_sizes,
                        "Passive Learning - QBC: Dice Scores vs Training Set Size",
                        "PassiveLearningQBC__hoi_DiceScores", color_train="green", color_test="red")

# === Comparison Plot ===
def plot_combined_comparison(dataset_sizes, us_train, us_test, pl_train, pl_test):
    def get_stats(data):
        return np.array([np.mean(data[s]) for s in dataset_sizes]), np.array([np.std(data[s]) for s in dataset_sizes])
    
    us_train_mean, us_train_std = get_stats(us_train)
    us_test_mean, us_test_std = get_stats(us_test)
    pl_train_mean, pl_train_std = get_stats(pl_train)
    pl_test_mean, pl_test_std = get_stats(pl_test)

    plt.figure(figsize=(10, 7))
    # Uncertainty
    plt.plot(dataset_sizes, us_train_mean, '-o', label='QBC Train', color='blue')
    plt.fill_between(dataset_sizes, us_train_mean - us_train_std, us_train_mean + us_train_std, alpha=0.2, color='blue')
    plt.plot(dataset_sizes, us_test_mean, '-o', label='QBC Test', color='orange')
    plt.fill_between(dataset_sizes, us_test_mean - us_test_std, us_test_mean + us_test_std, alpha=0.2, color='orange')
    # Passive
    plt.plot(dataset_sizes, pl_train_mean, '-s', label='PL Train', color='green')
    plt.fill_between(dataset_sizes, pl_train_mean - pl_train_std, pl_train_mean + pl_train_std, alpha=0.2, color='green')
    plt.plot(dataset_sizes, pl_test_mean, '-s', label='PL Test', color='red')
    plt.fill_between(dataset_sizes, pl_test_mean - pl_test_std, pl_test_mean + pl_test_std, alpha=0.2, color='red')

    plt.title("Comparison (QBC): Dice Scores vs Training Set Size")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Dice Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = "ComparisonQBC__hoi_DiceScores.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")
    return filename

file3 = plot_combined_comparison(dataset_sizes, train_results, test_results, pl_train_results, pl_test_results)

# === Save CSVs ===
csv_files = []
def save_df(data, name):
    df = pd.DataFrame(data)
    filename = f"{name}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")
    csv_files.append(filename)
    return filename

save_df(train_results, "QBCSamplingTrainDiceScores")
save_df(test_results, "QBCSamplingTestDiceScores")
save_df(pl_train_results, "PassiveLearningQBCTrainDiceScores")
save_df(pl_test_results, "PassiveLearningQBCTestDiceScores")


print("Saved train/test Dice scores to CSV")

#torch.save(model.state_dict(), "resnet34_model_all_data.pt")

# I commented from here
BUCKET_NAME = 'asr25project'

# Initialize the boto3 S3 client
s3 = boto3.client('s3')

# Upload all files in the 'results/' folder
# results_dir = 'results'
# for filename in os.listdir(results_dir):
#     local_path = os.path.join(results_dir, filename)
#     s3_path = f"results/{filename}"  # You can customize this path in the bucket
#     if os.path.isfile(local_path):
#         print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_path}")
#         s3.upload_file(local_path, BUCKET_NAME, s3_path)


# To here

# Upload all plots and CSVs
BUCKET_NAME = 'asr25data'
s3 = boto3.client('s3')

all_files = [file1, file2, file3] + csv_files
for fname in all_files:
    s3.upload_file(fname, BUCKET_NAME, f"results/{fname}")
    print(f"Uploaded {fname} to s3://{BUCKET_NAME}/{fname}")

# Upload all files in the 'results/' folder
# results_dir = 'results'
# for filename in os.listdir(results_dir):
#     local_path = os.path.join(results_dir, filename)
#     s3_path = f"results/{filename}"  # You can customize this path in the bucket
#     if os.path.isfile(local_path):
#         print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_path}")
#         s3.upload_file(local_path, BUCKET_NAME, s3_path)

# I commented this
#os.system('sudo shutdown now')
