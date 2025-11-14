import torch
import torch.nn.functional as F
import numpy as np

def train_epoch(model,train_loader, test_loader, optimizer, criterion_class, epoch, fold, device):
    model.train()
    train_loss, total_correct, total_kappa, total_f1, total_predictions = 0.0, 0, 0.0, 0.0, 0

    for batch_idx, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):

        s_eeg_hbo, s_eeg_hbr, s_hbr_hbo, s_label = train_data
        s_eeg_hbo, s_eeg_hbr, s_hbr_hbo = s_eeg_hbo.to(device), s_eeg_hbr.to(device), s_hbr_hbo.to(device)

        s_label = s_label.to(device).squeeze().long()

        t_eeg_hbo, t_eeg_hbr, t_hbr_hbo, _ = test_data
        t_eeg_hbo, t_eeg_hbr, t_hbr_hbo = t_eeg_hbo.to(device), t_eeg_hbr.to(device), t_hbr_hbo.to(device)


        p = float(batch_idx + epoch * len(train_loader)) / (num_epochs * len(train_loader))
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        optimizer.zero_grad()
        class_output, _ = model(s_eeg_hbo, s_eeg_hbr, s_hbr_hbo, t_eeg_hbo, t_eeg_hbr, t_hbr_hbo,alpha,mode='train')
        err_s_label = criterion_class(class_output.float(), s_label)

        err = err_s_label

        torch.cuda.empty_cache()
        err.backward()

        if model.graph_layer.gat_eo.A.grad is None:
            A_eo_grad = torch.zeros_like(model.graph_layer.gat_eo.A)
        else:
            A_eo_grad = model.graph_layer.gat_eo.A.grad
        model.graph_layer.gat_eo.update_adjacency_matrix(A_eo_grad,learning_rate=lr)

        if model.graph_layer.gat_er.A.grad is None:
            A_er_grad = torch.zeros_like(model.graph_layer.gat_er.A)
        else:
            A_er_grad = model.graph_layer.gat_er.A.grad
        model.graph_layer.gat_er.update_adjacency_matrix(A_er_grad, learning_rate=lr)

        if model.graph_layer.gat_ro.A.grad is None:
            A_ro_grad = torch.zeros_like(model.graph_layer.gat_ro.A)
        else:
            A_ro_grad = model.graph_layer.gat_ro.A.grad
        model.graph_layer.gat_ro.update_adjacency_matrix(A_ro_grad, learning_rate=lr)

        adjacency_matrix_eo = model.graph_layer.gat_eo.A.detach().cpu().numpy()
        adjacency_matrix_er = model.graph_layer.gat_er.A.detach().cpu().numpy()
        adjacency_matrix_ro = model.graph_layer.gat_ro.A.detach().cpu().numpy()

        adjacency_matrix_eo_normalized = normalize_matrix(adjacency_matrix_eo)
        adjacency_matrix_er_normalized = normalize_matrix(adjacency_matrix_er)
        adjacency_matrix_ro_normalized = normalize_matrix(adjacency_matrix_ro)


        eeg_data, hbo_data, hbr_data = separate_channels(s_eeg_hbo, s_eeg_hbr)


        adjacency_matrix_eeg = compute_single_modality_adjacency(eeg_data, "EEG")
        adjacency_matrix_hbo = compute_single_modality_adjacency(hbo_data, "HbO")
        adjacency_matrix_hbr = compute_single_modality_adjacency(hbr_data, "HbR")

        adjacency_matrix_eeg_normalized = normalize_matrix(adjacency_matrix_eeg)
        adjacency_matrix_hbo_normalized = normalize_matrix(adjacency_matrix_hbo)
        adjacency_matrix_hbr_normalized = normalize_matrix(adjacency_matrix_hbr)

        topomap_eeg_address = f'D:\\2code\\TYUT_III\\Figure_folder\\degree_centrality_topomap\\{fold + 1}\\EEG_Topomap_{epoch + 1}.png'
        plot_topomap_from_adjacency_matrix_EEG(adjacency_matrix_eeg_normalized, fold, epoch, topomap_eeg_address)
        topomap_HbO_address = f'D:\\2code\\TYUT_III\\Figure_folder\\degree_centrality_topomap\\{fold + 1}\\HbO_Topomap_{epoch + 1}.png'
        plot_topomap_from_adjacency_matrix_NIRS(adjacency_matrix_hbo_normalized, fold, epoch, topomap_HbO_address)
        topomap_HbR_address = f'D:\\2code\\TYUT_III\\Figure_folder\\degree_centrality_topomap\\{fold + 1}\\HbR_Topomap_{epoch + 1}.png'
        plot_topomap_from_adjacency_matrix_NIRS(adjacency_matrix_hbr_normalized, fold, epoch, topomap_HbR_address)
        topomap_eeg_hbo_address = f'D:\\2code\\TYUT_III\\Figure_folder\\degree_centrality_topomap\\{fold+1}\\EEG_HBO_Topomap_{epoch+1}.png'
        plot_topomap_from_adjacency_matrix(adjacency_matrix_eo_normalized,fold,epoch,topomap_eeg_hbo_address )
        topomap_eeg_hbr_address = f'D:\\2code\\TYUT_III\\Figure_folder\\degree_centrality_topomap\\{fold + 1}\\EEG_HBR_Topomap_{epoch + 1}.png'
        plot_topomap_from_adjacency_matrix(adjacency_matrix_er_normalized, fold, epoch, topomap_eeg_hbr_address)

        plt.figure(figsize=(10, 8))
        sns.heatmap(adjacency_matrix_eo_normalized, annot=False, cmap='viridis')
        plt.title('EEG_HBO Adjacency Matrix Heatmap')
        plt.savefig(
            f'D:\\2code\\TYUT_III\\att_matrix\\{fold}\\EEG_HBO_Adjacency_Matrix_Heatmap_{epoch+1}.png')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(adjacency_matrix_er_normalized, annot=False, cmap='viridis')
        plt.title('EEG_HBR Adjacency Matrix Heatmap')
        plt.savefig(
            f'D:\\2code\\TYUT_III\\att_matrix\\{fold}\\EEG_HBR_Adjacency_Matrix_Heatmap_{epoch+1}.png')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(adjacency_matrix_ro_normalized, annot=False, cmap='viridis')
        plt.title('HBO_HBR Adjacency Matrix Heatmap')
        plt.savefig(
            f'D:\\2code\\TYUT_III\\att_matrix\\{fold}\\HBO_HBR_Adjacency_Matrix_Heatmap_{epoch+1}.png')
        plt.close()

        save_top_percent_coords(adjacency_matrix_eo_normalized,
                                f'D:\\2code\\TYUT_III\\att_matrix\\{fold}\\EEG_HBO_Top_10_Percent_Coords_{epoch + 1}.csv',percent=0.5)
        save_top_percent_coords(adjacency_matrix_er_normalized,
                                f'D:\\2code\\TYUT_III\\att_matrix\\{fold}\\EEG_HBR_Top_10_Percent_Coords_{epoch + 1}.csv',percent=0.5)
        save_top_percent_coords(adjacency_matrix_ro_normalized,
                                f'D:\\2code\\TYUT_III\\att_matrix\\{fold}\\HBO_HBR_Top_10_Percent_Coords_{epoch + 1}.csv',percent=0.5)

        optimizer.step()


        train_loss += err.item()
        scores, predictions = torch.max(class_output, 1)
        batch_correct = (predictions == s_label).sum().item()
        total_correct += batch_correct
        total_predictions += len(s_label)


    train_acc = total_correct / total_predictions if total_predictions > 0 else 0

    return train_loss / len(train_loader), train_acc

def valid_epoch(model, train_loader,test_loader, device):
    model.eval()

    val_loss, val_correct, val_total = 0.0, 0, 0
    val_cm = np.zeros((4, 4), dtype=int)

    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for batch_idx, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):

            s_eeg_hbo, s_eeg_hbr, s_hbr_hbo, s_label = train_data
            s_eeg_hbo, s_eeg_hbr, s_hbr_hbo = s_eeg_hbo.to(device), s_eeg_hbr.to(device), s_hbr_hbo.to(device)
            s_label = s_label.to(device).squeeze().long()


            t_eeg_hbo, t_eeg_hbr, t_hbr_hbo, test_lab = test_data
            t_eeg_hbo, t_eeg_hbr, t_hbr_hbo = t_eeg_hbo.to(device), t_eeg_hbr.to(device), t_hbr_hbo.to(device)
            test_lab = test_lab.to(device).squeeze().long()

            p = float(batch_idx + epoch * len(test_loader)) / (num_epochs * len(test_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            class_output = model(s_eeg_hbo, s_eeg_hbr, s_hbr_hbo,t_eeg_hbo, t_eeg_hbr, t_hbr_hbo,
                                 alpha,mode='test')
            _, predicted = torch.max(class_output.data, 1)

            val_total  += test_lab.size(0)
            val_correct += (predicted == test_lab).sum().item()

            loss = F.cross_entropy(class_output, test_lab)
            val_loss += loss.item()

            batch_cm = confusion_matrix(test_lab.cpu().numpy(), predicted.cpu().numpy())
            if batch_cm.shape != (4, 4):
                continue
            val_cm += batch_cm

            y_true_all.extend(test_lab.cpu().numpy())
            y_pred_all.extend(predicted.cpu().numpy())

    val_acc = val_correct / val_total
    val_kappa = cohen_kappa_score(y_true_all, y_pred_all)
    val_f1 = f1_score(y_true_all, y_pred_all, average='macro')

    return val_loss/len(test_loader),val_acc,val_cm,val_kappa,val_f1
