import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

def interpolate_fnirs_data(data, new_num_samples=200):
    """
    Interpolate fnirs data to a new number of samples.
    Parameters:
    data (np.ndarray): Original fnirs data with shape (18, 11).
    new_num_samples (int): New number of samples per channel.
    Returns:
    np.ndarray: Interpolated fnirs data with shape (18, new_num_samples).
    """

    if data.shape != (18, 11):
        raise ValueError("Input data must have shape (18, 11).")

    interpolated_data = np.empty((18, new_num_samples))

    for i in range(18):

        x = np.linspace(0, 10, 11)

        x_new = np.linspace(0, 10, new_num_samples)

        f = interp1d(x, data[i, :], kind='linear')
        interpolated_data[i, :] = f(x_new)
    return interpolated_data



def normalize_features(features):

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    return features_normalized


def separate_channels(s_eeg_hbo, s_eeg_hbr):
    """
    从s_eeg_hbo和s_eeg_hbr中分离出EEG、HbO、HbR通道
    s_eeg_hbo: (batch, 80, 200)
    s_eeg_hbr: (batch, 80, 200)
    """

    eeg_from_hbo = s_eeg_hbo[:, :62, :]  # (batch, 62, 200)
    eeg_from_hbr = s_eeg_hbr[:, :62, :]  # (batch, 62, 200)


    eeg_combined = (eeg_from_hbo + eeg_from_hbr) / 2  # (batch, 62, 200)


    hbo_channels = s_eeg_hbo[:, 62:, :]  # (batch, 18, 200)


    hbr_channels = s_eeg_hbr[:, 62:, :]  # (batch, 18, 200)

    return eeg_combined, hbo_channels, hbr_channels


def compute_single_modality_adjacency(data, modality_name):

    batch_size, channels, time_points = data.shape


    correlation_matrices = []

    for i in range(batch_size):
        # 计算单个样本的相关性矩阵
        sample_data = data[i].cpu().numpy()  # (channels, time_points)
        correlation_matrix = np.corrcoef(sample_data)  # (channels, channels)
        correlation_matrices.append(correlation_matrix)


    avg_correlation = np.mean(correlation_matrices, axis=0)


    np.fill_diagonal(avg_correlation, 0)
    adjacency_matrix = np.abs(avg_correlation)

    return adjacency_matrix
