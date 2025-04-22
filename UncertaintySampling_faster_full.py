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

import boto3

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

USE_WARM_START = True
RESET_EVERY_N = 3  

name_extension = "uncertainty_sampling_full_training_local"
model_dir = f"{name_extension}/models"
results_dir = f'{name_extension}/results'
title_prefix = "Uncertainty Sampling FUll Learning"
plot_dir = f"{name_extension}/plots"
plots_title_prefix = "Uncertainty Sampling Full Learning"

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

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=4, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

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


"""# Model eval code"""

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

""" # QBC Training"""

def get_fisher_information_scores(model, dataset, unlabeled_indices):
    model.eval()
    fisher_scores = []
    loss_fn = torch.nn.BCELoss()
    epsilon = 1e-10

    for idx in unlabeled_indices:
        img, _, _ = dataset[idx]
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pseudo_label = model(img)

        img.requires_grad = True  # Still not necessary unless doing gradient w.r.t. input

        # Forward pass with gradient tracking
        pred = model(img)
        loss = loss_fn(pred, pseudo_label.detach())

        model.zero_grad()
        loss.backward()

        fisher_score = 0.0
        for param in model.parameters():
            if param.grad is not None:
                fisher_score += (param.grad ** 2).sum().item()

        fisher_scores.append((fisher_score, idx))

    return fisher_scores

def get_uncertainty_scores(model, dataset, unlabeled_indices):
    model.eval()
    uncertainties = []
    epsilon = 1e-10  # Small value to avoid log(0)

    with torch.no_grad():
        for idx in unlabeled_indices:
            img, _, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)

            # Predict probabilities using sigmoid
            pred = model(img).sigmoid()  # Shape: (1, 1, H, W)

            # Remove batch and channel dimensions -> shape: (H, W)
            probs = pred.squeeze().detach().cpu().numpy()

            # Compute pixel-wise entropy
            pixel_entropy = -probs * np.log(probs + epsilon) - (1 - probs) * np.log(1 - probs + epsilon)

            # Aggregate entropy (mean over all pixels)
            uncertainty = np.mean(pixel_entropy)

            uncertainties.append((uncertainty, idx))

    return uncertainties

def select_batch_using_fisher_and_uncertainty(model, dataset, unlabeled_indices, query_size, fisher_weight=1.0, uncertainty_weight=1.0):
    # Compute Fisher Information scores
    fisher_scores = get_fisher_information_scores(model, dataset, unlabeled_indices)

    # Compute Uncertainty scores
    uncertainty_scores = get_uncertainty_scores(model, dataset, unlabeled_indices)

    # Combine Fisher Information and Uncertainty scores
    combined_scores = []
    fisher_scores_dict = dict(fisher_scores)
    uncertainty_scores_dict = dict(uncertainty_scores)

    for idx in unlabeled_indices:
        fisher_score = fisher_scores_dict.get(idx, 0)
        uncertainty_score = uncertainty_scores_dict.get(idx, 0)
        
        # Combine both scores (you can tweak the weights as needed)
        combined_score = fisher_weight * fisher_score + uncertainty_weight * uncertainty_score
        combined_scores.append((combined_score, idx))

    # Sort by combined score (higher score means more uncertain and impactful)
    combined_scores.sort(reverse=True)
    
    # Select top `query_size` samples
    selected = [idx for _, idx in combined_scores[:query_size]]
    
    return selected


# Pass committee into the select_batch_using_fisher_and_qbc function

initial_size = 100
batch_size = 200
max_size = int(0.5 * len(train_ds))
n_simulations = 5

all_indices = list(range(len(train_ds)))
dataset_sizes = list(range(initial_size, max_size + 1, batch_size))

train_results, test_results = {}, {}

for sim in range(n_simulations):
    set_all_seeds(sim)
    unlabeled_indices = all_indices.copy()
    labeled_indices = []  # Initially empty labeled set

    warm_model = None
    for i, size in enumerate(dataset_sizes):
        reset_model = USE_WARM_START and RESET_EVERY_N > 0 and i % RESET_EVERY_N == 0

        if reset_model or warm_model is None:
            warm_model = None  # Reset model so a new one is initialized in evaluate_model_on_subset

        # Select the training data
        if i == 0:
            # Randomly select initial labeled batch
            labeled_indices = random.sample(unlabeled_indices, initial_size)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in labeled_indices]
        else:
            # Train model on current labeled set to use it for batch selection
            current_subset = labeled_indices
            _, _, warm_model = evaluate_model_on_subset(
                train_ds, current_subset, test_loader,
                warm_model=None, seed=sim
            )

            # Select new batch using Fisher + Uncertainty strategy
            new_batch_indices = select_batch_using_fisher_and_uncertainty(
                warm_model, train_ds, unlabeled_indices, query_size=batch_size
            )
            labeled_indices.extend(new_batch_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_batch_indices]

        # Train model on updated labeled set
        current_subset = labeled_indices
        print(f"  Training on {len(current_subset)} samples...", end="")

        train_dice, test_dice, warm_model = evaluate_model_on_subset(
            train_ds, current_subset, test_loader,
            warm_model=warm_model if USE_WARM_START else None, seed=sim
        )

        # Save model
        model_path = f"{model_dir}/model_sim{sim}_size{size}.pt"
        torch.save(warm_model.to('cpu').state_dict(), model_path)
        print(f" Saved model to {model_path}")
        print(f" Train Dice = {train_dice:.4f} | Test Dice = {test_dice:.4f}")

        # Log results
        train_results.setdefault(size, []).append(train_dice)
        test_results.setdefault(size, []).append(test_dice)

"""# Passive Learning Style Training"""


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
means_train = np.array([np.mean(train_results[s]) for s in dataset_sizes])
stds_train = np.array([np.std(train_results[s]) for s in dataset_sizes])
plt.plot(dataset_sizes, means_train, '-o')
plt.fill_between(dataset_sizes, means_train - stds_train, means_train + stds_train, alpha=0.3)
plt.title(f"{plots_title_prefix}: Mean Training Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Train Set Dice Score")
plt.grid(True)
plt.savefig(f"{plot_dir}/MeanTrainingDiceScore_US_Hoi_Full.png", bbox_inches='tight')
#plt.show()

means_test = np.array([np.mean(test_results[s]) for s in dataset_sizes])
stds_test = np.array([np.std(test_results[s]) for s in dataset_sizes])
plt.plot(dataset_sizes, means_test, '-o')
plt.fill_between(dataset_sizes, means_test - stds_test, means_test + stds_test, alpha=0.3)
plt.title(f"{plots_title_prefix}: Mean Test Set Dice Score vs Training Set Size")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Test Set Dice Score")
plt.grid(True)
plt.savefig(f"{plot_dir}/MeanTestDiceScore_US_Hoi_Full.png", bbox_inches='tight')
#plt.show()


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
plt.savefig(f"{plot_dir}/MeanBothDiceScore_US_Hoi_Full.png", dpi=300)
#plt.show()

print("Saved Figures")

train_df = pd.DataFrame(train_results)
train_df.to_csv(f"{plot_dir}/TrainDiceScores_US_Hoi_Full.csv", index=False)

test_df = pd.DataFrame(test_results)
test_df.to_csv(f"{plot_dir}/TestDiceScores_US_Hoi_Full.csv", index=False)

print("Saved US train/test Dice scores to CSV")

# To save to s3 bucket:
BUCKET_NAME = 'asr25data'

# Initialize the boto3 S3 client
s3 = boto3.client('s3')

# Upload individual files
#s3.upload_file('resnet34_model_all_data.pt', BUCKET_NAME, 'resnet34_model_all_data.pt')
for filename in os.listdir(plot_dir):
    local_path = os.path.join(plot_dir, filename)
    s3_path = f"{plot_dir}/{filename}"
    if os.path.isfile(local_path):
        print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_path}")
        s3.upload_file(local_path, BUCKET_NAME, s3_path)

# Upload all files in the results_dir folder
# for filename in os.listdir(results_dir):
#     local_path = os.path.join(results_dir, filename)
#     s3_path = f"{name_extension}/results/{filename}"
#     if os.path.isfile(local_path):
#         print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_path}")
#         s3.upload_file(local_path, BUCKET_NAME, s3_path)

for filename in os.listdir(model_dir):
    local_path = os.path.join(model_dir, filename)
    s3_path = f"{model_dir}/{filename}"
    if os.path.isfile(local_path):
        print(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_path}")
        s3.upload_file(local_path, BUCKET_NAME, s3_path)
    
#os.system('sudo shutdown now')

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

