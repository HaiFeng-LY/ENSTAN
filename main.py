import torch
import numpy as np
import pandas as pd
import os
from config.hyperparameters import *
from models.DMGAT import DMGAT
from models.TACA import TACA
from models.ENSTAN import ENSTAN_SSTEA
from utils.data_processing import *
from training.train_utils import train_epoch, valid_epoch
from data.loaders import load_subject_data, create_data_loaders
import pandas as pd
import os
import time
from sklearn.model_selection import KFold
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from config.hyperparameters import *
from utils.data_processing import *
from training.train_utils import train_epoch, valid_epoch
from data.loaders import load_subject_data, create_data_loaders
from utils.metrics import calculate_metrics, print_metrics_summary, calculate_cross_validation_metrics


def main():
    """Main training function"""

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load adjacency matrices
    try:
        EEG_fNIRS_filename = 'dis_whole_EEG_fNIRS_adj_matrix.pt'
        data_EEG_fNIRS = torch.load(EEG_fNIRS_filename)
        Adj_EEG_fNIRS = data_EEG_fNIRS['dis_data']

        HbO_HbR_filename = 'dis_whole_hbo_hbr_adj_matrix.pt'
        data_HbO_HbR = torch.load(HbO_HbR_filename)
        Adj_HbO_HbR = data_HbO_HbR['dis_data']

        print("Adjacency matrices loaded successfully")
    except Exception as e:
        print(f"Error loading adjacency matrices: {e}")
        return

    # Define subject IDs and event IDs
    person_id = [5, 12, 25]
    event_id = list(range(1, 61))  # 1 to 60

    # Initialize results storage
    all_val_cor = []
    all_val_kappa = []
    all_val_f1 = []
    all_cms = []
    all_train_time = []

    # Cross-validation setup
    splits = KFold(n_splits=k_folds, shuffle=True, random_state=77)
    foldperf = {}

    # Main training loop across subjects
    for i, subject_id in enumerate(person_id):
        print(f"\n{'=' * 60}")
        print(f"Training for Subject {subject_id} ({i + 1}/{len(person_id)})")
        print(f"{'=' * 60}")

        # Load subject data
        try:
            dataset_eeg_hbo, dataset_eeg_hbr, dataset_hbr_hbo, dataset_label = load_subject_data(
                subject_id, event_id, data_path='E:\\haifeng\\codegpu\\tyut3_code\\'
            )
            print(f"Loaded data for subject {subject_id}: {len(dataset_eeg_hbo)} samples")
        except Exception as e:
            print(f"Error loading data for subject {subject_id}: {e}")
            continue

        # Convert to tensors
        dataset_eeg_hbo = [torch.from_numpy(item) for item in dataset_eeg_hbo]
        dataset_eeg_hbr = [torch.from_numpy(item) for item in dataset_eeg_hbr]
        dataset_hbr_hbo = [torch.from_numpy(item) for item in dataset_hbr_hbo]
        dataset_label = [torch.from_numpy(item) for item in dataset_label]

        # Subject-specific results
        subject_val_corrects = []
        subject_f1_scores = []
        subject_kappa_scores = []
        subject_best_acc = 0

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset_eeg_hbo)):
            print(f'\nFold {fold + 1}/{k_folds} for Subject {subject_id}')
            print(f'Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}')

            # Create train/validation datasets
            train_eeg_hbo = torch.stack([dataset_eeg_hbo[i] for i in train_idx])
            test_eeg_hbo = torch.stack([dataset_eeg_hbo[i] for i in val_idx])
            train_eeg_hbr = torch.stack([dataset_eeg_hbr[i] for i in train_idx])
            test_eeg_hbr = torch.stack([dataset_eeg_hbr[i] for i in val_idx])
            train_hbr_hbo = torch.stack([dataset_hbr_hbo[i] for i in train_idx])
            test_hbr_hbo = torch.stack([dataset_hbr_hbo[i] for i in val_idx])
            train_label = torch.stack([dataset_label[i] for i in train_idx])
            test_label = torch.stack([dataset_label[i] for i in val_idx])

            # Create data loaders
            train_dataset = TensorDataset(train_eeg_hbo, train_eeg_hbr, train_hbr_hbo, train_label)
            test_dataset = TensorDataset(test_eeg_hbo, test_eeg_hbr, test_hbr_hbo, test_label)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model
            graph_encoder = DMGAT(
                in_channels=time_points,
                num_ele_eeg_fnirs=80,
                num_ele_hbo_hbr=36,
                num_heads=gat_num_heads,
                hidden_dim=gat_hidden_dim,
                out_channels=gat_out_channels,
                A_init_eeg_fnirs=Adj_EEG_fNIRS,
                A_init_hbo_hbr=Adj_HbO_HbR,
                dropout=dropout_rate,
                devices=device
            )

            encoder_layer = TACA(
                d_model_eeg=62,
                d_model_hbo=18,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout_rate
            )

            classification_model = ENSTAN_SSTEA(
                graph_encoder,
                encoder_layer,
                num_classes=num_classes
            ).to(device)

            # Define loss function and optimizer
            criterion_emo_class = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(classification_model.parameters(), lr=learning_rate)

            # Training history
            history = {
                'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [],
                'confusion_matrix': [], 'kappa': [], 'F1_score': [], 'train_time': []
            }

            # Training loop
            fold_val_corrects = []

            for epoch in tqdm(range(num_epochs), desc=f"Fold {fold + 1} Epochs"):
                start_epoch_time = time.time()

                # Train for one epoch
                train_loss, train_correct = train_epoch(
                    classification_model, train_loader, test_loader, optimizer,
                    criterion_emo_class, epoch, fold, device
                )

                train_time = time.time() - start_epoch_time

                # Validate
                val_loss, val_correct, val_cm, val_kapp, val_f1 = valid_epoch(
                    classification_model, train_loader, test_loader, device
                )

                fold_val_corrects.append(val_correct)
                subject_f1_scores.append(val_f1)
                subject_kappa_scores.append(val_kapp)

                # Update history
                history['train_loss'].append(train_loss)
                history['test_loss'].append(val_loss)
                history['train_acc'].append(train_correct)
                history['test_acc'].append(val_correct)
                history['confusion_matrix'].append(val_cm)
                history['kappa'].append(val_kapp)
                history['F1_score'].append(val_f1)
                history['train_time'].append(train_time)

                # Print progress
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}: "
                          f"Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}, "
                          f"Train Acc: {train_correct:.4f}, Test Acc: {val_correct:.4f}")

            # Store fold performance
            foldperf[f'fold{fold + 1}'] = history

            # Calculate fold metrics
            max_acc = np.max(fold_val_corrects)
            avg_acc = np.mean(fold_val_corrects)
            std_acc = np.std(fold_val_corrects)
            max_f1 = np.max(subject_f1_scores)
            avg_f1 = np.mean(subject_f1_scores)
            max_kappa = np.max(subject_kappa_scores)
            avg_kappa = np.mean(subject_kappa_scores)

            print(f"\nFold {fold + 1} Results:")
            print(f"Max Accuracy: {max_acc:.4f}")
            print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"Max F1: {max_f1:.4f}")
            print(f"Average F1: {avg_f1:.4f}")
            print(f"Max Kappa: {max_kappa:.4f}")
            print(f"Average Kappa: {avg_kappa:.4f}")

            # Update best accuracy
            if max_acc > subject_best_acc:
                subject_best_acc = max_acc

            # Clean up
            del classification_model
            torch.cuda.empty_cache()

        # Calculate subject-level metrics
        testl_f, tl_f, testa_f, ta_f, cm_f, kappa_f, f1_f, train_time_f = [], [], [], [], [], [], [], []

        for f in range(1, k_folds + 1):
            tl_f.append(np.mean(foldperf[f'fold{f}']['train_loss']))
            testl_f.append(np.mean(foldperf[f'fold{f}']['test_loss']))
            ta_f.append(np.mean(foldperf[f'fold{f}']['train_acc']))
            testa_f.append(np.mean(foldperf[f'fold{f}']['test_acc']))
            cm_f.append(foldperf[f'fold{f}']['confusion_matrix'])
            kappa_f.append(np.mean(foldperf[f'fold{f}']['kappa']))
            f1_f.append(np.mean(foldperf[f'fold{f}']['F1_score']))
            train_time_f += foldperf[f'fold{f}']['train_time']

        # Store subject results
        all_val_cor.append(testa_f)
        all_val_kappa.append(kappa_f)
        all_val_f1.append(f1_f)

        if history['confusion_matrix']:
            cumulative_cm = np.sum(history['confusion_matrix'], axis=0)
            average_cm = cumulative_cm / (num_epochs + 1)
            all_cms.append(average_cm)

        # Print subject summary
        print(f"\nSubject {subject_id} Summary:")
        print(f"Best testing accuracy: {subject_best_acc:.4f}")
        print(f"Average Test Accuracy: {np.mean(testa_f):.4f} ± {np.std(testa_f):.4f}")
        print(f"Kappa: {np.mean(kappa_f):.4f}")
        print(f"F1-Scores: {np.mean(f1_f):.4f}")
        print(f"Total Training Time: {np.mean(train_time_f) * num_epochs * k_folds:.2f} seconds")

        # Save subject results to CSV
        results_dict = {
            'ID': str(subject_id),
            'Best_Testing_Accuracy': subject_best_acc,
            'Average_Test_Losses': np.mean(testl_f),
            'Average_Test_Accuracy': np.mean(testa_f),
            'Test_Std_Accuracy': np.std(testa_f),
            'Kappa': np.mean(kappa_f),
            'F1_Scores': np.mean(f1_f),
            'Total_Training_Time': np.mean(train_time_f) * num_epochs * k_folds,
        }

        df_results = pd.DataFrame([results_dict])
        results_folder = 'results'
        file_path = os.path.join(results_folder, 'DyGAT_Adv_EEG_HBO_HBR.csv')

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        if os.path.isfile(file_path):
            existing_df = pd.read_csv(file_path)
            df_results = pd.concat([existing_df, df_results], ignore_index=True)

        df_results.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")

    # Final summary across all subjects
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY ACROSS ALL SUBJECTS")
    print(f"{'=' * 60}")

    if all_val_cor:
        print(f"Average Accuracy across all subjects: {np.mean(all_val_cor):.4f} ± {np.std(all_val_cor):.4f}")
        print(f"Average Kappa across all subjects: {np.mean(all_val_kappa):.4f}")
        print(f"Average F1-Scores across all subjects: {np.mean(all_val_f1):.4f}")
        print(f"Total Training Time across all subjects: {np.mean(all_train_time):.2f} seconds")

    if all_cms:
        all_cms_array = np.array(all_cms)
        average_cm_all_subjects = np.mean(all_cms_array, axis=0)
        print(f"\nAverage Confusion Matrix across all subjects:\n{average_cm_all_subjects}")
    else:
        print("No confusion matrices available for averaging")


if __name__ == "__main__":
    main()