import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torch.optim import lr_scheduler
from utils.utils import get_loss
from tqdm import tqdm

def train(model, train_dataset, val_dataset,  max, min, args):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    data_max=max
    data_min=min

    model.to(args.device)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')
    
    for e in range(args.epoch):
        model.train()
        metrics_train = {
            3: {'mae': 0, 'rmse': 0, 'mape': 0},
            6: {'mae': 0, 'rmse': 0, 'mape': 0},
            12: {'mae': 0, 'rmse': 0, 'mape': 0, 'loss': 0}
        }

        n_iter = 0
        
        for i_batch, (img_tensor, text_tensor, x_samples_tensor, y_samples_tensor) in enumerate(tqdm(train_loader, desc=f"Training Epoch {e+1}/{args.epoch}")):
            img_tensor = img_tensor.to(args.device)
            text_tensor = text_tensor.to(args.device)
            x_samples_tensor = x_samples_tensor.to(args.device)
            y_samples_tensor = y_samples_tensor.to(args.device)

            pred_3, pred_6, pred_12, m_pred_3, m_pred_6, m_pred_12 = model(img_tensor, text_tensor, x_samples_tensor, e, args.l)  

            loss_1 = criterion(pred_12, y_samples_tensor[...,:12])
            loss_2 = criterion(m_pred_12, y_samples_tensor[...,:12])
            loss = loss_1 * args.beta + loss_2 * (1 - args.beta)

        
            new_values = {
                3: get_loss(pred_3, m_pred_3, y_samples_tensor[..., :3], data_max, data_min, args.beta),
                6: get_loss(pred_6, m_pred_6, y_samples_tensor[..., :6], data_max, data_min, args.beta),
                12: get_loss(pred_12, m_pred_12, y_samples_tensor, data_max, data_min, args.beta)
            }
            for t, (mae, rmse, mape) in new_values.items():
                for metric, value in zip(['mae', 'rmse', 'mape'], [mae, rmse, mape]):
                    metrics_train[t][metric] += value
            
            metrics_train[12]['loss'] += loss.item()
            
            n_iter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for t, (mae, rmse, mape) in new_values.items():
            for metric, value in zip(['mae', 'rmse', 'mape'], [mae, rmse, mape]):
                metrics_train[t][metric] /= n_iter

        model.eval()
        metrics_val = {
            3: {'mae': 0, 'rmse': 0, 'mape': 0},
            6: {'mae': 0, 'rmse': 0, 'mape': 0},
            12: {'mae': 0, 'rmse': 0, 'mape': 0, 'loss': 0}
        }
        metrics_test = {
            3: {'mae': 0, 'rmse': 0, 'mape': 0},
            6: {'mae': 0, 'rmse': 0, 'mape': 0},
            12: {'mae': 0, 'rmse': 0, 'mape': 0}
        }

        with torch.no_grad():
            for img_tensor, text_tensor, x_samples_tensor, y_samples_tensor in tqdm(val_loader, desc=f"Validation Epoch {e+1}/{args.epoch}"):
                img_tensor = img_tensor.to(args.device)
                text_tensor = text_tensor.to(args.device)
                x_samples_tensor = x_samples_tensor.to(args.device)
                y_samples_tensor = y_samples_tensor.to(args.device)

                pred_3, pred_6, pred_12, m_pred_3, m_pred_6, m_pred_12 = model(img_tensor, text_tensor, x_samples_tensor, 1, args.l)
                
                loss_1 = criterion(pred_12, y_samples_tensor[...,:12])
                loss_2 = criterion(m_pred_12, y_samples_tensor[...,:12])
                loss = loss_1 * args.beta + loss_2 * (1 - args.beta)

                new_values = {
                    3: get_loss(pred_3, m_pred_3, y_samples_tensor[..., :3], data_max, data_min, args.beta),
                    6: get_loss(pred_6, m_pred_6, y_samples_tensor[..., :6], data_max, data_min, args.beta),
                    12: get_loss(pred_12, m_pred_12, y_samples_tensor, data_max, data_min, args.beta)
                }
                for t, (mae, rmse, mape) in new_values.items():
                    for metric, value in zip(['mae', 'rmse', 'mape'], [mae, rmse, mape]):
                        metrics_val[t][metric] += value
                metrics_val[12]['loss'] += loss.item()

        
            avg_val_loss = metrics_val[12]['loss'] / len(val_loader)
            for t, (mae, rmse, mape) in new_values.items():
                for metric, value in zip(['mae', 'rmse', 'mape'], [mae, rmse, mape]):
                    metrics_val[t][metric] /= len(val_loader)

            t2=time.time()

            # save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                path=args.best_model_path + '/{}_{}_{}_{}.pth'.format(args.data, args.lr, args.his_len, args.pred_len, args.l)
                torch.save(model.state_dict(), path)
                print(f"Model saved at epoch {e}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered!")
                    break

            
        print(f"Epoch [{e}/{args.epoch}], TRAIN--STEP3--MAE: {metrics_train[3]['mae']:.4f}, RMSE: {metrics_train[3]['rmse']:.4f}, MAPE: {metrics_train[3]['mape']:.4f}, VAL----MAE: {metrics_val[3]['mae']:.4f}, RMSE: {metrics_val[3]['rmse']:.4f}, MAPE: {metrics_val[3]['mape']:.4f}")
        print(f"Epoch [{e}/{args.epoch}], TRAIN--STEP6--MAE: {metrics_train[6]['mae']:.4f}, RMSE: {metrics_train[6]['rmse']:.4f}, MAPE: {metrics_train[6]['mape']:.4f}, VAL----MAE: {metrics_val[6]['mae']:.4f}, RMSE: {metrics_val[6]['rmse']:.4f}, MAPE: {metrics_val[6]['mape']:.4f}")
        print(f"Epoch [{e}/{args.epoch}], TRAIN--STEP12--MAE: {metrics_train[12]['mae']:.4f}, RMSE: {metrics_train[12]['rmse']:.4f}, MAPE: {metrics_train[12]['mape']:.4f}, VAL----MAE: {metrics_val[12]['mae']:.4f}, RMSE: {metrics_val[12]['rmse']:.4f}, MAPE: {metrics_val[12]['mape']:.4f}")
        
        scheduler.step()

    
    return t2
