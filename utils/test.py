import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import torch.optim as optim
import time
from utils.utils import get_loss

def test(model, dataset, path, max, min, args):
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    data_max = max
    data_min = min

    model.load_state_dict(torch.load(path))
    model.to(args.device)
    
    criterion = nn.MSELoss()

    model.eval()
    metrics_test = {
        3: {'mae': 0, 'rmse': 0, 'mape': 0},
        6: {'mae': 0, 'rmse': 0, 'mape': 0},
        12: {'mae': 0, 'rmse': 0, 'mape': 0}
    }

    with torch.no_grad():
        for img_tensor, text_tensor, x_samples_tensor, y_samples_tensor in test_loader:
            img_tensor = img_tensor.to(args.device)
            text_tensor = text_tensor.to(args.device)
            x_samples_tensor = x_samples_tensor.to(args.device)
            y_samples_tensor = y_samples_tensor.to(args.device)

            pred_3, pred_6, pred_12, m_pred_3, m_pred_6, m_pred_12 = model(img_tensor, text_tensor, x_samples_tensor, 1, args.l)

            new_values = {
                3: get_loss(pred_3, m_pred_3, y_samples_tensor[..., :3], data_max, data_min, args.alpha),
                6: get_loss(pred_6, m_pred_6, y_samples_tensor[..., :6], data_max, data_min, args.alpha),
                12: get_loss(pred_12, m_pred_12, y_samples_tensor, data_max, data_min, args.alpha)
            }
            for t, (mae, rmse, mape) in new_values.items():
                for metric, value in zip(['mae', 'rmse', 'mape'], [mae, rmse, mape]):
                    metrics_test[t][metric] += value
            

    for t, (mae, rmse, mape) in new_values.items():
        for metric, value in zip(['mae', 'rmse', 'mape'], [mae, rmse, mape]):
            metrics_test[t][metric] /= len(test_loader)

    print(f"TEST--STEP3--MAE: {metrics_test[3]['mae']:.4f}, RMSE: {metrics_test[3]['rmse']:.4f}, MAPE: {metrics_test[3]['mape']:.4f}")
    print(f"TEST--STEP6--MAE: {metrics_test[6]['mae']:.4f}, RMSE: {metrics_test[6]['rmse']:.4f}, MAPE: {metrics_test[6]['mape']:.4f}")
    print(f"TEST--STEP12--MAE: {metrics_test[12]['mae']:.4f}, RMSE: {metrics_test[12]['rmse']:.4f}, MAPE: {metrics_test[12]['mape']:.4f}")


    t3=time.time()

    return t3
