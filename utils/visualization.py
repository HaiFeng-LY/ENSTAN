import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_heatmap(input_data, cmap='coolwarm', annot=True, figsize=(10, 8)):
    mean_data = input_data.mean(dim=0)
    mean_data_np = mean_data.numpy()

    plt.figure(figsize=figsize)
    sns.heatmap(mean_data_np, annot=annot, cmap=cmap)
    plt.xlabel('seq2')
    plt.ylabel('seq1')
    plt.title('Heatmap of Mean Data')
    plt.show()


def generate_attn_mask():

    attn_mask = torch.ones(200, 11, dtype=torch.bool)

    for q in range(200):
        for k in range(11):

            if q / 200.0 > k / 11.0:
                attn_mask[q, k] = False
    return attn_mask



def save_top_percent_coords(matrix, filename, percent=0.5):
    # 确保percent是一个有效的百分比
    if not (0 < percent <= 100):
        raise ValueError("percent must be between 0 and 100")

        # 忽略对角线元素
    np.fill_diagonal(matrix, 0)

    # 计算绝对值并扁平化，同时记录原始索引
    flat_abs = np.abs(matrix).flatten()
    indices = np.indices(matrix.shape).reshape(2, -1).T  # 注意这里的转置，以便每行是一个坐标

    # 找到前percent%的绝对值最高的索引
    threshold = np.percentile(flat_abs, 100 - percent)
    selected_indices = np.where(flat_abs >= threshold)[0]

    # 提取这些索引对应的坐标和值
    selected_coords = indices[selected_indices]
    selected_values = flat_abs[selected_indices]

    # 创建一个DataFrame来保存坐标和值
    df = pd.DataFrame(np.column_stack((selected_coords, selected_values)), columns=['X', 'Y', 'Value'])

    # 保存到CSV
    df.to_csv(filename, index=False)


def normalize_matrix(matrix):
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
