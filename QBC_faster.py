import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import segmentation_models_pytorch as smp

import cv2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

USE_WARM_START = True
RESET_EVERY_N = 3  

name_extension = "passive_learning_partial_training_local"
model_dir = f"{name_extension}/models"
results_dir = f'{name_extension}/results'
title_prefix = "Passive Learning"
plot_dir = f"{name_extension}/plots"
plots_title_prefix = "Passive Learning"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

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

## Use a small subset due to local compute limitations
# Randomly pick 10 indices from the training and testing dataset
set_all_seeds(0)
subset_indices = random.sample(range(len(train_ds)), 10)

test_subset_indices = random.sample(range(len(test_ds)), 10)
test_subset = Subset(test_ds, test_subset_indices)
test_subset_loader =  DataLoader(test_subset, batch_size=1, num_workers = 0)

"""## UNet Model Definition"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def show_prediction(model, img, mask, results_dir, filename, save=True):
    model.eval()
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred_bin = (pred > 0.5).float().squeeze().cpu().numpy()

    pred_unpadded = unpad_to_shape(pred_bin, 520, 704)
    img_unpadded = unpad_to_shape(img.squeeze(0), 520, 704)
    mask_unpadded = unpad_to_shape(mask.squeeze(0), 520, 704)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img_unpadded, cmap='gray')
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(mask_unpadded, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(pred_unpadded, cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    plt.tight_layout()

    if save:
        save_path = f"{results_dir}/{filename}_prediction.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved prediction to {save_path}")
    else:
        plt.show()
    plt.close(fig)

""" # QBC Training"""

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

# Initialize your committee of models
def create_committee(n_models=5):
    committee = []
    for _ in range(n_models):
        model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1, activation="sigmoid").to(device)
        committee.append(model)
    return committee

# Create the committee before the training loop
committee = create_committee(n_models=3)  # Create a committee with 5 models (adjust as needed)

# Pass committee into the select_batch_using_fisher_and_qbc function

initial_size = 1
batch_size = 1
max_size = int(0.8 * len(subset_indices))
n_simulations = 3

all_indices = subset_indices
dataset_sizes = list(range(initial_size, max_size + 1, batch_size))

train_results, test_results = {}, {}

for sim in range(n_simulations):
    set_all_seeds(sim)
    unlabeled_indices = all_indices.copy()
    labeled_indices = []  # Initially empty labeled set
    
    warm_model = None
    for i, size in enumerate(dataset_sizes):
        # Select the next batch of samples based on QBC and Fisher scores
        if size > initial_size:
            new_batch_indices = select_batch_using_fisher_and_qbc(committee, train_ds, unlabeled_indices, batch_size=batch_size)
            labeled_indices.extend(new_batch_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_batch_indices]
        
        print(f"  Training on {len(labeled_indices)} samples...", end="")
        
        # Reset model every N steps in warm-start mode
        reset_model = USE_WARM_START and RESET_EVERY_N > 0 and i % RESET_EVERY_N == 0
        
        if reset_model:
            warm_model = None
            current_subset = labeled_indices  # full subset up to this point
        else:
            current_subset = labeled_indices  # only new data since partial training
        
        train_dice, test_dice, warm_model = evaluate_model_on_subset(train_ds, current_subset, test_subset_loader, warm_model=warm_model if USE_WARM_START else None, seed=sim)
        
        ## Prediction
        for img, mask, fname in test_subset:
            base_name = os.path.splitext(os.path.basename(fname))[0]
            file_name = f"{base_name}_sim_{sim}_train_size_{size}"
            show_prediction(warm_model, img, mask, results_dir, filename=file_name)
            break  # Only one image, just to check

        model_path = f"{model_dir}/model_sim{sim}_size{size}.pt"
        torch.save(warm_model.to('cpu').state_dict(), model_path)
        print(f"Saved model to {model_path}")
        print(f" Train Dice = {train_dice:.4f}", f" Test Dice = {test_dice:.4f}")
        train_results.setdefault(size, []).append(train_dice)
        test_results.setdefault(size, []).append(test_dice)

"""# Passive Learning Style Training"""

def evaluate_model_on_subset(dataset, subset_indices, test_loader, epochs=5, warm_model=None, seed = 0):
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=4, shuffle=True, num_workers = 0)
    set_all_seeds(seed)
    model = warm_model if warm_model else smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1, activation="sigmoid").to(device)
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

    return final_train_dice, final_test_dice, model

"""# Passive Learning
initial_size = 1
increment = 1
max_size = int(0.8 * len(subset_indices))
n_simulations = 3

all_indices = subset_indices
dataset_sizes = list(range(initial_size, max_size + 1, increment))

train_results, test_results = {}, {}
for sim in range(n_simulations):
    set_all_seeds(sim)
    shuffled_indices = all_indices.copy()
    random.shuffle(shuffled_indices)
    
    warm_model = None
    for i, size in enumerate(dataset_sizes):
        # Reset model every N steps in warm-start mode
        reset_model = USE_WARM_START and RESET_EVERY_N > 0 and i % RESET_EVERY_N == 0
        
        if reset_model:
            warm_model = None
            current_subset = shuffled_indices[:size]  # full subset up to this point
        else:
            start_idx = size - increment if size != initial_size else 0
            current_subset = shuffled_indices[start_idx:size]  # only new data since partial training
        print(f"  Training on {size} samples...", end="")
        train_dice, test_dice, warm_model = evaluate_model_on_subset(train_ds, current_subset, test_subset_loader, warm_model=warm_model if USE_WARM_START else None, seed = sim)
        
        ## Prediction
        for img, mask, fname in test_subset:
            base_name = os.path.splitext(os.path.basename(fname))[0]
            file_name = f"{base_name}_sim_{sim}_train_size_{size}"
            show_prediction(warm_model, img, mask, results_dir, filename=file_name)
            break  # Only one image, just to check

        model_path = f"{model_dir}/model_sim{sim}_size{size}.pt"
        torch.save(warm_model.to('cpu').state_dict(), model_path)
        print(f"Saved model to {model_path}")
        print(f" Train Dice = {train_dice:.4f}", f" Test Dice = {test_dice:.4f}")
        train_results.setdefault(size, []).append(train_dice)
        test_results.setdefault(size, []).append(test_dice)"""


# Plotting

# QBC
# Plotting QBC Results
means_qbc = np.array([np.mean(qbc_results[s]) for s in dataset_sizes])  # Mean QBC Score
stds_qbc = np.array([np.std(qbc_results[s]) for s in dataset_sizes])  # Standard Deviation of QBC Score

# Plot QBC Mean vs Training Set Size
plt.plot(dataset_sizes, means_qbc, '-o', label='QBC Score (Mean)', color='green')
plt.fill_between(dataset_sizes, means_qbc - stds_qbc, means_qbc + stds_qbc, color='green', alpha=0.3)
plt.title(f"{plots_title_prefix}: Mean QBC Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean QBC Score")
plt.grid(True)
plt.savefig(f"{plot_dir}/MeanQBCScore.png", bbox_inches='tight')
plt.show()

# Plot QBC Mean and Train/Test Dice Score for Comparison
plt.figure(figsize=(8, 6))
plt.plot(dataset_sizes, means_qbc, label='QBC Score (Mean)', color='green', marker='o')
plt.plot(dataset_sizes, means_train, label='Train Dice (Mean)', color='blue', marker='o')
plt.plot(dataset_sizes, means_test, label='Test Dice (Mean)', color='orange', marker='o')
plt.fill_between(dataset_sizes, means_qbc - stds_qbc, means_qbc + stds_qbc, color='green', alpha=0.3)
plt.fill_between(dataset_sizes, means_train - stds_train, means_train + stds_train, color='blue', alpha=0.3)
plt.fill_between(dataset_sizes, means_test - stds_test, means_test + stds_test, color='orange', alpha=0.3)

# Labels and legend
plt.title(f"{plots_title_prefix}: Mean QBC Score vs Training Set Size (with Train/Test Dice)")
plt.xlabel("Training Set Size")
plt.ylabel("Scores")
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)

# Save or show
plt.tight_layout()
plt.savefig(f"{plot_dir}/MeanQBCTrainTestDiceScore.png", dpi=300)
plt.show()

print("Saved QBC Score Figures")

# Save the QBC results to CSV
qbc_df = pd.DataFrame(qbc_results)
qbc_df.to_csv(f"{plot_dir}/QBCScores.csv", index=False)

print("Saved QBC scores to CSV")

"""# Passive
means_train = np.array([np.mean(train_results[s]) for s in dataset_sizes])
stds_train = np.array([np.std(train_results[s]) for s in dataset_sizes])
plt.plot(dataset_sizes, means_train, '-o')
plt.fill_between(dataset_sizes, means_train - stds_train, means_train + stds_train, alpha=0.3)
plt.title(f"{plots_title_prefix}: Mean Training Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Train Set Dice Score")
plt.grid(True)
plt.savefig(f"{plot_dir}/MeanTrainingDiceScore.png", bbox_inches='tight')
plt.show()

means_test = np.array([np.mean(test_results[s]) for s in dataset_sizes])
stds_test = np.array([np.std(test_results[s]) for s in dataset_sizes])
plt.plot(dataset_sizes, means_test, '-o')
plt.fill_between(dataset_sizes, means_test - stds_test, means_test + stds_test, alpha=0.3)
plt.title(f"{plots_title_prefix}: Mean Test Set Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Test Set Dice Score")
plt.grid(True)
plt.savefig(f"{plot_dir}/MeanTestDiceScore.png", bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(dataset_sizes, means_train, label='Train Dice (Mean)', color='blue', marker='o')
plt.plot(dataset_sizes, means_test, label='Test Dice (Mean)', color='orange', marker='o')
plt.fill_between(dataset_sizes, means_train - stds_train, means_train + stds_train, color='blue', alpha=0.3)
plt.fill_between(dataset_sizes, means_test - stds_test, means_test + stds_test, color='orange', alpha=0.3)

# Labels and legend
plt.title(f"{plots_title_prefix}: Mean Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Dice Score")
plt.legend()
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)

# Save or show
plt.tight_layout()
plt.savefig(f"{plot_dir}/MeanBothDiceScore.png", dpi=300)
plt.show()

print("Saved Figures")

train_df = pd.DataFrame(train_results)
train_df.to_csv(f"{plot_dir}/TrainDiceScores.csv", index=False)

test_df = pd.DataFrame(test_results)
test_df.to_csv(f"{plot_dir}/TestDiceScores.csv", index=False)

print("Saved train/test Dice scores to CSV")"""

