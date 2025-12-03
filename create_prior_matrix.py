# create_prior_matrix.py
import numpy as np
import os

def create_prior_matrix_identity():
    """创建单位矩阵作为先验矩阵"""
    num_nodes = 100
    
    # 创建单位矩阵
    prior_matrix = np.eye(num_nodes, dtype=np.float32)
    
    # 保存
    os.makedirs('cache/deepshap/prior_matrix', exist_ok=True)
    np.save('cache/deepshap/prior_matrix/matrix.npy', prior_matrix)
    
    print(f"✅ 创建单位矩阵: shape={prior_matrix.shape}")
    print(f"   对角线: 全1")
    print(f"   非对角线: 全0")
    print(f"   保存到: cache/deepshap/prior_matrix/matrix.npy")
    
    return prior_matrix

create_prior_matrix_identity()