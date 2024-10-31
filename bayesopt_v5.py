import os
import gc
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
import random

from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from IPython.display import display
    
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.model_selection import KFold


import torch
import numpy as np
import random
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from ax.service.managed_loop import optimize


class RandomMirrorAndRotate:
    def __init__(self):
        # Define the possible transformations for tensors
        self.transformations = [
            lambda x: x,  # Identity (no transformation)
            F.hflip,      # Horizontal flip
            F.vflip,      # Vertical flip
            lambda x: torch.rot90(x, 1, [2, 3]),  # Rotate 90 degrees
            lambda x: torch.rot90(x, 2, [2, 3]),  # Rotate 180 degrees
            lambda x: torch.rot90(x, 3, [2, 3]),  # Rotate 270 degrees
            lambda x: torch.rot90(F.hflip(x), 1, [2, 3]),  # Horizontal flip + Rotate 90 degrees
            lambda x: torch.rot90(F.vflip(x), 1, [2, 3])   # Vertical flip + Rotate 90 degrees
        ]
        
    def __call__(self, img):
        # Randomly select one of the transformations
        transform = random.choice(self.transformations)
        return transform(img)


# Check if CUDA (GPU) is available
cuda_device = 'cuda:1'  # most people use cuda:0 so let's try to use the other GPU on the machine
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, device='cpu'):
        self.data = data
        self.device = device
        self.preload_to_device()

    def preload_to_device(self):
        self.data = [(image.to(self.device), group, torch.tensor(features).float().to(self.device)) for image, group, features in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, group, features = self.data[index]
        return image, group, features


# Define the image transforms
image_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.RandomAutocontrast(p=1.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def make_dataset(data_folder, N=1, verbose=False):

    # Set the random seed
    random.seed(16)

    # Create a list to store the data tuples
    this_data = []

    # Get a list of subfolders in the data folder
    subfolders = os.listdir(data_folder)
    sample_keys = {k: i for i, k in enumerate(subfolders)}

    # Sort subfolders based on the index after splitting by '_', and [-3] represent the UTS
    subfolders.sort(key=lambda x: float(x.split('_')[-3]))

    # Group subfolders based on i % 5
    grouped_subfolders = [[] for _ in range(5)]
    for i, subfolder in enumerate(subfolders):
        index = i // (len(subfolders)//5)
        if index >=5:
            index -= 1
        grouped_subfolders[index].append(subfolder)
    if verbose:
        print(grouped_subfolders)

    chunk_keys = {}
    for i, gs in enumerate(grouped_subfolders):
        for sf in gs:
            chunk_keys[sf] = i

    # Randomly select one subfolder from each group
    for _ in range(len(subfolders) // 5 +1):
        for k, group in enumerate(grouped_subfolders):
            if group:
                selected_subfolder = random.choice(group)
                group.remove(selected_subfolder)
                folder_path = os.path.join(data_folder, selected_subfolder)

                # Load the CSV data
                csv_data = None
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".csv"):
                        csv_path = os.path.join(folder_path, file_name)
                        csv_data = pd.read_csv(csv_path)
                        break

                # Load the image data and combine it with the CSV data
                num = 0
                image_names = [image_name for image_name in os.listdir(folder_path) if image_name.endswith(".jpg")]
                image_names.sort()
                # Select every fifth image
                for i, image_name in enumerate(image_names):
                    if i % N != 0:
                        continue
                    num+=1
                    # print(image_name)
                    image_path = os.path.join(folder_path, image_name)
                    image_data = Image.open(image_path).convert("RGB")  # Convert image to RGB mode
                    image_data = image_transforms(image_data)

                    # Find the corresponding row in the CSV data
                    if csv_data is not None:
                        # image_index = csv_data[csv_data["Image Name"] == image_name].index[0]
                        # image_features = csv_data.iloc[image_index, 1:].values.astype(float)
                        image_features = csv_data.loc[csv_data["Image Name"] == image_name, "UTS (MPa)"].values[0].astype(float)
                    else:
                        image_features = None

                    # Add data to the list
                    this_data.append((image_data, (chunk_keys[selected_subfolder], sample_keys[selected_subfolder]), image_features))
                if verbose:
                    print(f'Number of images in folder {selected_subfolder}: {num}')

    return CustomDataset(this_data, device=cuda_device)

# Remember to change data folder path
TRAIN_VAL_PATH = "./Images/Train_val"
TEST_PATH = "./Images/Test"

# GLOBAL Variable
# Epoch = 30  # original value
# Epoch = 64  # more epochs might mean better generalization with augmentation?
Epoch = 32  # it seems like more epochs leads to overfitting?
# Epoch = 100  # tried this, it was wasteful; most runs optimized around 25
date = '240710'

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Define the number of epochs
num_epochs = Epoch

# Initialize KFold object
n_splits = 5
kf = StratifiedGroupKFold(n_splits=n_splits)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Define the objective function
def objective_function(params):

    # params['N_skip'] = 10  # HARDCODE
    N_skip = params['N_skip']

    Lr = params['lr']
    # Batch_size = params['batch_size']
    Batch_size = 32  # seems like either 16 or 32
    print(f"Evaluating parameters: {params}")

    params['use_h_flip'] = True
    params['use_v_flip'] = True
    params['use_rotation'] = False

    transforms_list = [transforms.CenterCrop(224)]
    if (params['use_h_flip'] and params['use_v_flip'] and params['use_rotation']):
        # all 8 possible states
        transforms_list.append( RandomMirrorAndRotate() )
    else:
        # we will defalut to using horizontal augmentation
        if ('use_h_flip' not in params) or ('use_h_flip' in params and params['use_h_flip']):
            transforms_list.append( transforms.RandomHorizontalFlip(p=0.5) )
        # other augmentations need to be explicitly enabled
        if 'use_v_flip' in params and params['use_v_flip']:
            transforms_list.append( transforms.RandomVerticalFlip(p=0.5) )
        if 'use_rotation' in params and params['use_rotation']:
            transforms_list.append( transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5) )

    if params['use_contrast']:
        transforms_list.append( transforms.RandomAutocontrast(p=1.0) )

    transforms_list.append( transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) )
    online_transforms = transforms.Compose(transforms_list)

    # build validation/testing transforms (no augmentation)
    val_xform_list = [transforms.CenterCrop(224), ]
    if params['use_contrast']:
        val_xform_list.append( transforms.RandomAutocontrast(p=1.0) )
    val_xform_list.append( transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) )    
    offline_transforms = transforms.Compose(val_xform_list)

    # Create dataset with chosen "N" (how many images to skip)
    train_val_dataset = make_dataset(TRAIN_VAL_PATH, N=N_skip)
    test_dataset = make_dataset(TEST_PATH, N=N_skip)

    val_loss_all_fold = []
    test_loss_all_fold = []
    min_epoch_all_fold = []

    test_preds_history = []

    uts_label = [it[1][0] for it in train_val_dataset]  # record the UTS chunk # when making the dataset
    sample_id = [it[1][1] for it in train_val_dataset]  # record the sample # when making the dataset

    # FOR CHECKING THE CONTENTS OF EACH FOLD:
    # for fold, (train_index, val_index) in enumerate(kf.split(train_val_dataset, uts_label, sample_id)):
    #     val_dataset = Subset(train_val_dataset, val_index)
    #     val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, worker_init_fn=seed_worker)
    #     samples_in_fold = []
    #     for i, (images, group, labels) in enumerate(val_loader):
    #         samples_in_fold += group[1].cpu().numpy().tolist()
    #     print(sorted(set(samples_in_fold)))
    # raise RuntimeError('Fence.')

    for fold, (train_index, val_index) in enumerate(kf.split(train_val_dataset, uts_label, sample_id)):

        # print(f"Fold {fold+1}/{n_splits}")
        set_seed(42)

        # Create train and validation datasets for this fold
        train_dataset = Subset(train_val_dataset, train_index)
        val_dataset = Subset(train_val_dataset, val_index)

        # Create train and validation loaders for this fold
        train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, worker_init_fn=seed_worker)

        # Load pre-trained ResNext50 model
        # model = models.swin_t(weights='IMAGENET1K_V1')
        # in_features = model.head.in_features
        # model.head = nn.Linear(in_features, 1)

        # model = models.resnext50_32x4d(weights='IMAGENET1K_V1')

        model = models.resnet18(weights='IMAGENET1K_V1')        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

        # Move model to GPU
        model.to(device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Lr)

        # Initialize lists to store training and validation losses for this fold
        val_loss_history = []
        test_loss_history = []
        test_preds_best = None

        # Train the model for this fold
        pbar = tqdm.tqdm(range(num_epochs))
        for epoch in pbar:
            model.train()
            set_seed(42)
            for i, (images, group, labels) in enumerate(train_loader):

                images = online_transforms(images)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate the model on the validation set after each epoch
            model.eval()
            with torch.no_grad():
                val_loss = 0
                k = 0
                preds_val = []
                true_labels_val = []
                for images, group, labels in val_loader:
                    images = offline_transforms(images)

                    # Forward pass
                    outputs = model(images)
                    val_loss += criterion(outputs.squeeze(), labels).item() * len(images)
                    k += len(images)
                    preds_val.append(outputs.squeeze())
                    true_labels_val.append(labels)

                val_loss /= k
                preds_val = torch.cat(preds_val, dim=0).detach().cpu().numpy()
                true_labels_val = torch.cat(true_labels_val, dim=0).detach().cpu().numpy()

                # do the same for test loss; we'll grab the value from the lowest val at the end
                test_loss = 0
                k = 0
                preds_test = []
                true_labels_test = []
                for images, group, labels in test_loader:
                    # Apply offline transforms and move inputs and labels to GPU
                    images = offline_transforms(images)

                    # Forward pass
                    outputs = model(images)
                    test_loss += criterion(outputs.squeeze(), labels).item() * len(images)
                    k += len(images)
                    preds_test.append(outputs.squeeze())
                    true_labels_test.append(labels)

                test_loss /= k
                preds_test = torch.cat(preds_test, dim=0).detach().cpu().numpy()
                true_labels_test = torch.cat(true_labels_test, dim=0).detach().cpu().numpy()

            # print(f'Epoch: {ep} Validation Loss: {val_loss}')
            val_loss_history.append(val_loss)
            test_loss_history.append(test_loss)

            if val_loss == np.min(val_loss_history):
                test_preds_best = preds_test.copy()
                # save the model
                torch.save(model, f'resnet18-v5-fold{fold+1}.pth')

            pbar.set_postfix_str(f"Train {loss.item():.3e}, Val {val_loss:.3e}, Test {test_loss:.3e}")

        min_epoch = np.argmin(val_loss_history)
        val_loss_all_fold.append(val_loss_history[min_epoch])
        min_epoch_all_fold.append(min_epoch + 1)
        # use the best validation epoch for the test evaluation
        test_loss_all_fold.append(test_loss_history[min_epoch])
        test_preds_history.append(test_preds_best)

    folds_median_test = np.median(np.vstack(test_preds_history), axis=0)
    median_test_mse = np.mean((folds_median_test - true_labels_test)**2)

    folds_mean_test = np.mean(np.vstack(test_preds_history), axis=0)
    mean_test_mse = np.mean((folds_mean_test - true_labels_test)**2)

    mean_val_loss = np.mean(val_loss_all_fold)
    std_val_loss = np.std(val_loss_all_fold)

    print(f'Validation for five folds: {val_loss_all_fold}')
    print(f'Lowest validation loss for each fold occurred at epochs: {min_epoch_all_fold}')
    print(f'Mean validation loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}')
    print(f'Mean test loss: {np.mean(test_loss_all_fold):.4f} ± {np.std(test_loss_all_fold):.4f}')
    print(f'Mean test over folds: {mean_test_mse:.4f}')
    # print(f'Median test over folds: {median_test_mse:.4f}')

    ucb_val_loss = mean_val_loss + std_val_loss
    max_val_loss = np.max(val_loss_all_fold)

    loss_obj = max_val_loss
    # loss_obj = ucb_val_loss

    print(f'Validation loss function: {loss_obj:.4f}')

    if loss_obj != loss_obj:
        loss_obj = 1e3  # default value for NaN (why is it NaN?)
    print(f'Objective function: {loss_obj:.4f}')

    return loss_obj, min_epoch_all_fold


def main():

    # Run only the latest, best result
    # NOTE that we implicitly used H and V flips, see objective fn above
    best_parameters = {'lr': 0.0009761248347350309, 'N_skip': 2, 'use_contrast': False}
    best_max_val_loss, best_epochs_each_fold = objective_function(best_parameters)
    raise RuntimeError('Encountered deliberate runtime fence')

    # Store the best parameters and the best epochs for each fold
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
            # {"name": "N_skip", "type": "range", "bounds": [1, 20], "value_type": "int"},
            {"name": "N_skip", "type": "choice", "values": [2, 4, 5, 10, 20], "value_type": "int"},
            # {"name": "batch_size", "type": "range", "bounds": [16, 64], "value_type": "int"},
            # {"name": "batch_size", "type": "choice", "values": [16, 24, 32], "value_type": "int"},
            # {"name": "use_h_flip", "type": "choice", "values": [True, False], "value_type": "bool"},
            # {"name": "use_v_flip", "type": "choice", "values": [True, False], "value_type": "bool"},
            # {"name": "use_rotation", "type": "choice", "values": [True, False], "value_type": "bool"},
            {"name": "use_contrast", "type": "choice", "values": [True, False], "value_type": "bool"},
        ],
        evaluation_function=lambda params: objective_function(params)[0],  # Return only max_val_loss for optimization
        objective_name='max_val_loss',
        minimize=True,
        total_trials=64,
    )

    # Retrieve the best parameters and the best epochs for each fold
    best_max_val_loss, best_epochs_each_fold = objective_function(best_parameters)

    print('\n')
    print(f"Best parameters: {best_parameters}")
    print(f"Mean validation loss: {values}")
    print(f"Best epochs for each fold: {best_epochs_each_fold}")
    print('\n')

    # Set the best parameters from Ax-platform
    best_params = {
        'lr': best_parameters['lr'],
        'N_skip': best_parameters['N_skip'],
    }

    print(best_params)
    print(best_epochs_each_fold)

if __name__ == "__main__":
    main()
