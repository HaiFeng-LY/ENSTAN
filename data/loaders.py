import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler


def interpolate_fnirs_data(data, new_num_samples=200):
    """
    Interpolate fnirs data to a new number of samples.

    Parameters:
    ----------
    data : np.ndarray
        Original fnirs data with shape (18, 11)
    new_num_samples : int
        New number of samples per channel

    Returns:
    ----------
    np.ndarray
        Interpolated fnirs data with shape (18, new_num_samples)
    """
    # Ensure input data has correct shape
    if data.shape != (18, 11):
        raise ValueError("Input data must have shape (18, 11).")

    # Initialize empty array for interpolated data
    interpolated_data = np.empty((18, new_num_samples))

    # Interpolate each channel
    for i in range(18):
        # Original x-axis (11 points from 0 to 10)
        x = np.linspace(0, 10, 11)
        # New x-axis (200 points from 0 to 10)
        x_new = np.linspace(0, 10, new_num_samples)
        # Use linear interpolation
        f = interp1d(x, data[i, :], kind='linear')
        interpolated_data[i, :] = f(x_new)

    return interpolated_data


def normalize_features(features):
    """
    Normalize features using StandardScaler.

    Parameters:
    ----------
    features : np.ndarray
        Feature array with shape (num_samples, num_channels, num_features)

    Returns:
    ----------
    np.ndarray
        Normalized feature array
    """
    scaler = StandardScaler()
    # Reshape for normalization: (channels * time_points) as features
    original_shape = features.shape
    features_reshaped = features.reshape(original_shape[0], -1)
    features_normalized = scaler.fit_transform(features_reshaped)
    # Reshape back to original shape
    features_normalized = features_normalized.reshape(original_shape)
    return features_normalized


def load_subject_data(person_id, event_ids, data_path):
    """
    Load subject data from MATLAB files and preprocess.

    Parameters:
    ----------
    person_id : int
        Subject ID
    event_ids : list
        List of event IDs to load
    data_path : str
        Base path to data files

    Returns:
    ----------
    tuple
        (dataset_eeg_hbo, dataset_eeg_hbr, dataset_hbr_hbo, dataset_label)
    """

    dataset_eeg_hbo = []
    dataset_eeg_hbr = []
    dataset_hbr_hbo = []
    dataset_label = []

    for event_id in event_ids:
        try:
            # Construct file paths
            datafile_eeg = f'{data_path}Feature_folder/Raw_1s/EEG/EEG_RAW_no_fre_channel_{person_id}_{event_id}'
            datafile_hbo = f'{data_path}Feature_folder/Raw_1s/HBO/HBO_RAW_{person_id}_{event_id}'
            datafile_hbr = f'{data_path}Feature_folder/Raw_1s/HBR/HBR_RAW_{person_id}_{event_id}'
            datafile_label = f'{data_path}Feature_folder/Raw_1s/LABEL/RAW_label_{person_id}_{event_id}'

            # Load data from MATLAB files
            EEG_RAW = sio.loadmat(datafile_eeg)['data']  # (-1, 62, 1000)
            HBO_RAW_11 = sio.loadmat(datafile_hbo)['data']  # (-1, 18, 11)
            HBR_RAW_11 = sio.loadmat(datafile_hbr)['data']  # (-1, 18, 11)
            Y = sio.loadmat(datafile_label)['data']
            Y = np.array(Y).reshape(np.array(Y).shape[1], 1)

            # Process each sample
            for sample_idx in range(EEG_RAW.shape[0]):
                # EEG data processing (downsample from 1000 to 200 time points)
                feature_eeg = EEG_RAW[sample_idx, :, ::5]  # (-1, 62, 200)

                # fNIRS data processing (interpolate from 11 to 200 time points)
                feature_hbo_11 = HBO_RAW_11[sample_idx, :, :]
                feature_hbr_11 = HBR_RAW_11[sample_idx, :, :]
                feature_hbo = interpolate_fnirs_data(feature_hbo_11)  # (-1, 18, 200)
                feature_hbr = interpolate_fnirs_data(feature_hbr_11)  # (-1, 18, 200)

                # Normalize features
                feature_eeg = normalize_features(feature_eeg)
                feature_hbo = normalize_features(feature_hbo)
                feature_hbr = normalize_features(feature_hbr)

                # Create multimodal combinations
                eeg_hbo = np.concatenate((feature_eeg, feature_hbo), axis=0)  # (80, 200)
                eeg_hbr = np.concatenate((feature_eeg, feature_hbr), axis=0)  # (80, 200)
                hbr_hbo = np.concatenate((feature_hbr, feature_hbo), axis=0)  # (36, 200)
                lab = Y[sample_idx]

                # Append to datasets
                dataset_eeg_hbo.append(eeg_hbo)
                dataset_eeg_hbr.append(eeg_hbr)
                dataset_hbr_hbo.append(hbr_hbo)
                dataset_label.append(lab)

        except Exception as e:
            print(f"Error loading data for subject {person_id}, event {event_id}: {e}")
            continue

    print(f"Loaded {len(dataset_eeg_hbo)} samples for subject {person_id}")
    return dataset_eeg_hbo, dataset_eeg_hbr, dataset_hbr_hbo, dataset_label


def create_data_loaders(datasets, batch_size, k_folds=5, random_state=77):
    """
    Create data loaders for training and testing with k-fold cross validation.

    Parameters:
    ----------
    datasets : tuple
        Tuple containing (eeg_hbo, eeg_hbr, hbr_hbo, labels) datasets
    batch_size : int
        Batch size for data loaders
    k_folds : int
        Number of folds for cross validation
    random_state : int
        Random state for reproducibility

    Returns:
    ----------
    list
        List of tuples containing (train_loader, test_loader) for each fold
    """

    dataset_eeg_hbo, dataset_eeg_hbr, dataset_hbr_hbo, dataset_label = datasets

    # Convert to tensors
    dataset_eeg_hbo = [torch.from_numpy(item).float() for item in dataset_eeg_hbo]
    dataset_eeg_hbr = [torch.from_numpy(item).float() for item in dataset_eeg_hbr]
    dataset_hbr_hbo = [torch.from_numpy(item).float() for item in dataset_hbr_hbo]
    dataset_label = [torch.from_numpy(item).long() for item in dataset_label]

    # Stack tensors
    dataset_eeg_hbo = torch.stack(dataset_eeg_hbo)
    dataset_eeg_hbr = torch.stack(dataset_eeg_hbr)
    dataset_hbr_hbo = torch.stack(dataset_hbr_hbo)
    dataset_label = torch.stack(dataset_label)

    # Initialize KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    data_loaders = []

    # Create data loaders for each fold
    for train_idx, test_idx in kf.split(dataset_eeg_hbo):
        # Split data
        train_eeg_hbo = dataset_eeg_hbo[train_idx]
        test_eeg_hbo = dataset_eeg_hbo[test_idx]
        train_eeg_hbr = dataset_eeg_hbr[train_idx]
        test_eeg_hbr = dataset_eeg_hbr[test_idx]
        train_hbr_hbo = dataset_hbr_hbo[train_idx]
        test_hbr_hbo = dataset_hbr_hbo[test_idx]
        train_label = dataset_label[train_idx]
        test_label = dataset_label[test_idx]

        # Create TensorDatasets
        train_dataset = TensorDataset(train_eeg_hbo, train_eeg_hbr, train_hbr_hbo, train_label)
        test_dataset = TensorDataset(test_eeg_hbo, test_eeg_hbr, test_hbr_hbo, test_label)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        data_loaders.append((train_loader, test_loader))

    print(f"Created {len(data_loaders)} folds with batch size {batch_size}")
    return data_loaders


def create_single_data_loader(datasets, batch_size, train_ratio=0.8, random_state=77):
    """
    Create single train-test split data loaders (alternative to k-fold).

    Parameters:
    ----------
    datasets : tuple
        Tuple containing (eeg_hbo, eeg_hbr, hbr_hbo, labels) datasets
    batch_size : int
        Batch size for data loaders
    train_ratio : float
        Ratio of data to use for training
    random_state : int
        Random state for reproducibility

    Returns:
    ----------
    tuple
        (train_loader, test_loader)
    """

    dataset_eeg_hbo, dataset_eeg_hbr, dataset_hbr_hbo, dataset_label = datasets

    # Convert to tensors
    dataset_eeg_hbo = torch.stack([torch.from_numpy(item).float() for item in dataset_eeg_hbo])
    dataset_eeg_hbr = torch.stack([torch.from_numpy(item).float() for item in dataset_eeg_hbr])
    dataset_hbr_hbo = torch.stack([torch.from_numpy(item).float() for item in dataset_hbr_hbo])
    dataset_label = torch.stack([torch.from_numpy(item).long() for item in dataset_label])

    # Calculate split index
    num_samples = len(dataset_eeg_hbo)
    split_idx = int(num_samples * train_ratio)

    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Shuffle indices
    indices = torch.randperm(num_samples)

    # Split data
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_eeg_hbo = dataset_eeg_hbo[train_indices]
    test_eeg_hbo = dataset_eeg_hbo[test_indices]
    train_eeg_hbr = dataset_eeg_hbr[train_indices]
    test_eeg_hbr = dataset_eeg_hbr[test_indices]
    train_hbr_hbo = dataset_hbr_hbo[train_indices]
    test_hbr_hbo = dataset_hbr_hbo[test_indices]
    train_label = dataset_label[train_indices]
    test_label = dataset_label[test_indices]

    # Create TensorDatasets
    train_dataset = TensorDataset(train_eeg_hbo, train_eeg_hbr, train_hbr_hbo, train_label)
    test_dataset = TensorDataset(test_eeg_hbo, test_eeg_hbr, test_hbr_hbo, test_label)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Created single split: {len(train_dataset)} training, {len(test_dataset)} test samples")
    return train_loader, test_loader

