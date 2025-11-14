import torch
import torch.nn as nn

def diff_loss(diff, S, Falpha):
    if len(S.shape) == 4:
        # batch input
        return  torch.mean(torch.log(torch.sum(torch.mean(torch.sqrt(diff ** 2), axis=3) * S, axis=(1, 2))))
    else:
        return  torch.mean(torch.log(torch.matmul(S, torch.sqrt(diff ** 2)) + 1))

def F_norm_loss(S, Falpha):
    if len(S.shape) > 2:

        squared_sum = torch.sum(S ** 2, dim=(-2, -1))

        return Falpha * torch.sqrt(torch.sum(squared_sum))
    else:

        return Falpha * torch.sqrt(torch.sum(S ** 2))

def L1_regularization(matrix, lambda_l1):
    l1_loss = torch.sum(torch.abs(matrix))
    return lambda_l1 * l1_loss

def pcc_matrix(tensor):
    Batch, Node, Time = tensor.shape
    dot_products_sum = []
    for batch in range(Batch):
        per_pcc = 1 - np.corrcoef(tensor[batch, :, :].cpu().numpy())
        dot_products_sum.append(per_pcc)
    tensor_result = torch.tensor(dot_products_sum, dtype=torch.float32)
    return tensor_result