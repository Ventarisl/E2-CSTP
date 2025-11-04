import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import os
import re
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class CMoSTDataset(Dataset):
    def __init__(self, image_dir, text_dir, time_series_path, his_len, pred_len, lat_range, lon_range, transform=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.time_series_path = time_series_path
        self.his_len = his_len
        self.pred_len = pred_len
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.lat_range = lat_range
        self.lon_range = lon_range

        self.image_files = self._parse_files(image_dir, r"relief_(\d+)([NS])_(\d+)([EW])\.png")
        self.text_files = self._parse_files(text_dir, r"llm_(\d+)([NS])_(\d+)([EW])\.txt")

        self.time_series_data = np.load(time_series_path).astype(np.float32)  # Shape: (t, n)
        self.t, self.n = self.time_series_data.shape
        self.nums = self.t - self.his_len - self.pred_len

        train_end = int(self.nums * 0.8)
        val_end = int(self.nums * 0.9)

        train_data_part = self.time_series_data[:train_end + self.his_len]
        self.train_data_min = np.min(train_data_part).astype(np.float32)
        self.train_data_max = np.max(train_data_part).astype(np.float32)

        self.normalized_data = (self.time_series_data - self.train_data_min) / (self.train_data_max - self.train_data_min)


        self.geo_to_index = self._generate_geo_index()

        self.matched_samples = self._match_samples()

    def _parse_files(self, directory, pattern):
        files = {}
        regex = re.compile(pattern)
        for file in os.listdir(directory):
            match = regex.match(file)
            if match:
                lat, lat_dir, lon, lon_dir = match.groups()
                lat = int(lat) * (-1 if lat_dir == "S" else 1)
                lon = int(lon) * (-1 if lon_dir == "W" else 1)
                if self.lat_range[0] <= lat < self.lat_range[1] and self.lon_range[0] <= lon < self.lon_range[1]:
                    files[(lat, lon)] = os.path.join(directory, file)
        return files

    def _generate_geo_index(self):
        index_map = {}

        latitudes = np.linspace(self.lat_range[0], self.lat_range[1]-1, self.lat_range[1]-self.lat_range[0])
        longitudes = np.linspace(self.lon_range[0], self.lon_range[1]-1, self.lon_range[1]-self.lon_range[0])
        count = 0
        for lat in latitudes:
            for lon in longitudes:
                index_map[(int(lat), int(lon))] = count
                count += 1
        return index_map

    def _match_samples(self):
        matched_samples = []
        for coord in self.image_files.keys():
            if coord in self.text_files and coord in self.geo_to_index:
                matched_samples.append(coord)
        return matched_samples

    def __len__(self):
        return len(self.matched_samples)

    def __getitem__(self, idx):
        lat, lon = self.matched_samples[idx]

        image_path = self.image_files[(lat, lon)]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img_tensor = self.transform(img)

        text_path = self.text_files[(lat, lon)]
        with open(text_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        text_tensor = torch.tensor([ord(c) for c in text_data[-200:]])

        time_series_index = self.geo_to_index[(lat, lon)]
        time_series = self.normalized_data[:, time_series_index]

        x_samples = []
        y_samples = []
        for start_idx in range(self.t - self.his_len - self.pred_len):
            x_samples.append(time_series[start_idx:start_idx + self.his_len])
            y_samples.append(time_series[start_idx + self.his_len:start_idx + self.his_len + self.pred_len])

        x_samples = np.array(x_samples)
        y_samples = np.array(y_samples)
        x_samples_tensor = torch.tensor(x_samples, dtype=torch.float32)
        y_samples_tensor = torch.tensor(y_samples, dtype=torch.float32)

        img_tensor = img_tensor.repeat(x_samples_tensor.shape[0], 1, 1, 1).squeeze(1)
        text_tensor = text_tensor.repeat(x_samples_tensor.shape[0], 1)

        return img_tensor, text_tensor, x_samples_tensor, y_samples_tensor, self.train_data_max, self.train_data_min
    
class CMoSTTrainDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        self.train_num_time_steps = int(original_dataset.nums*0.8)
        self.val_num_time_steps = int(original_dataset.nums*0.1)
        self.test_num_time_steps = original_dataset.nums - self.train_num_time_steps - self.val_num_time_steps
        self.train_data_max=0
        self.train_data_min=0

        self._reorganize_data()

    def _reorganize_data(self):
        img_list = []
        text_list = []
        x_samples_list = []
        y_samples_list = []

        for i in range(self.num_nodes): 
            img_tensor, text_tensor, x_samples_tensor, y_samples_tensor, self.train_data_max, self.train_data_min = self.original_dataset[i]
            
            img_list.append(img_tensor[:1]) 
            text_list.append(text_tensor[:1])
            x_samples_list.append(x_samples_tensor[:self.train_num_time_steps]) 
            y_samples_list.append(y_samples_tensor[:self.train_num_time_steps]) 

        self.img_data = torch.stack(img_list, dim=1)
        self.text_data = torch.stack(text_list, dim=1)
        self.x_samples_data = torch.stack(x_samples_list, dim=1)
        self.y_samples_data = torch.stack(y_samples_list, dim=1)
    
    def __len__(self):
        return self.train_num_time_steps

    def __getitem__(self, idx):
        img_tensor = self.img_data[0]
        text_tensor = self.text_data[0]
        x_samples_tensor = self.x_samples_data[idx]
        y_samples_tensor = self.y_samples_data[idx] 

        return img_tensor, text_tensor, x_samples_tensor, y_samples_tensor
    
class CMoSTValDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        self.train_num_time_steps = int(original_dataset.nums*0.8)
        self.val_num_time_steps = int(original_dataset.nums*0.1)
        self.test_num_time_steps = original_dataset.nums - self.train_num_time_steps - self.val_num_time_steps
        self.train_data_max=0
        self.train_data_min=0

        self._reorganize_data()

    def _reorganize_data(self):
        img_list = []
        text_list = []
        x_samples_list = []
        y_samples_list = []

        for i in range(self.num_nodes):
            img_tensor, text_tensor, x_samples_tensor, y_samples_tensor, self.train_data_max, self.train_data_min = self.original_dataset[i]
            
            img_list.append(img_tensor[:1])
            text_list.append(text_tensor[:1])
            x_samples_list.append(x_samples_tensor[self.train_num_time_steps:self.train_num_time_steps+self.val_num_time_steps])
            y_samples_list.append(y_samples_tensor[self.train_num_time_steps:self.train_num_time_steps+self.val_num_time_steps])

        self.img_data = torch.stack(img_list, dim=1)
        self.text_data = torch.stack(text_list, dim=1)
        self.x_samples_data = torch.stack(x_samples_list, dim=1)
        self.y_samples_data = torch.stack(y_samples_list, dim=1)
    
    def __len__(self):
        return self.val_num_time_steps
    
    def __getitem__(self, idx):
        img_tensor = self.img_data[0]
        text_tensor = self.text_data[0]
        x_samples_tensor = self.x_samples_data[idx]
        y_samples_tensor = self.y_samples_data[idx] 

        return img_tensor, text_tensor, x_samples_tensor, y_samples_tensor

class CMoSTTestDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        self.train_num_time_steps = int(original_dataset.nums*0.8)
        self.val_num_time_steps = int(original_dataset.nums*0.1)
        self.test_num_time_steps = original_dataset.nums - self.train_num_time_steps - self.val_num_time_steps
        self.train_data_max=0
        self.train_data_min=0

        self._reorganize_data()

    def _reorganize_data(self):
        img_list = []
        text_list = []
        x_samples_list = []
        y_samples_list = []

        for i in range(self.num_nodes): 
            img_tensor, text_tensor, x_samples_tensor, y_samples_tensor, self.train_data_max, self.train_data_min = self.original_dataset[i]
            
            img_list.append(img_tensor[:1])
            text_list.append(text_tensor[:1])
            x_samples_list.append(x_samples_tensor[self.train_num_time_steps+self.val_num_time_steps:])
            y_samples_list.append(y_samples_tensor[self.train_num_time_steps+self.val_num_time_steps:])

        self.img_data = torch.stack(img_list, dim=1) 
        self.text_data = torch.stack(text_list, dim=1)
        self.x_samples_data = torch.stack(x_samples_list, dim=1)
        self.y_samples_data = torch.stack(y_samples_list, dim=1)
    
    def __len__(self):
        return self.test_num_time_steps
    
    def __getitem__(self, idx):
        img_tensor = self.img_data[0]
        text_tensor = self.text_data[0]
        x_samples_tensor = self.x_samples_data[idx]
        y_samples_tensor = self.y_samples_data[idx] 

        return img_tensor, text_tensor, x_samples_tensor, y_samples_tensor

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str ='cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

def get_loss(pred, m_pred, y, max, min, beta):
    pred = (pred * (max - min) + min).flatten().cpu().detach().numpy()
    m_pred = (m_pred * (max - min) + min).flatten().cpu().detach().numpy()
    y = (y * (max - min) + min).flatten().cpu().numpy()

    mae_1 = mean_absolute_error(pred, y)
    rmse_1 = np.sqrt(mean_squared_error(pred, y))
    mape_1 = mean_absolute_percentage_error(pred, y)

    mae_2 = mean_absolute_error(m_pred, y)
    rmse_2 = np.sqrt(mean_squared_error(m_pred, y))
    mape_2 = mean_absolute_percentage_error(m_pred, y)

    mae=mae_1*beta+mae_2*(1-beta)
    rmse=rmse_1*beta+rmse_2*(1-beta)
    mape=mape_1*beta+mape_2*(1-beta)
    
    return mae, rmse, mape




    
    
