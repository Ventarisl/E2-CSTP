import os
import warnings
import torchvision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['DGL_WARNLEVEL'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="unrecognized nn.Module: SiLU")
warnings.filterwarnings("ignore", message="lambda_max is not provided")
torchvision.disable_beta_transforms_warning()

import random
import numpy as np
import torch
import argparse
import configparser
import time
from models.CMoST import CMoST
from utils.train import train
from utils.test import test
from utils.utils import CMoSTTrainDataset, CMoSTValDataset, CMoSTTestDataset, CMoSTDataset
from torch.cuda.amp import autocast, GradScaler

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default='config/Terra.conf')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    parser.add_argument('--image_dir', type=str, default=config['file']['image_dir'])
    parser.add_argument('--text_dir', type=str, default=config['file']['text_dir'])
    parser.add_argument('--time_series_path', type=str, default=config['file']['time_series_path'])

    parser.add_argument('--data', type = str, default=config['data']['data'])
    parser.add_argument('--num_nodes', type = int, default=config['data']['num_nodes'])
    parser.add_argument('--his_len', type = int, default=config['data']['his_len'])
    parser.add_argument('--pred_len', type = int, default=config['data']['pred_len'])
    parser.add_argument('--matrix', default=None)

    parser.add_argument('--lr', type=float, default=config['train']['lr'])
    parser.add_argument('--sigma', type=float, default=config['train']['sigma'])
    parser.add_argument('--dropout', type = float, default=config['train']['dropout'])
    parser.add_argument('--device', type = str, default=config['train']['device'])
    parser.add_argument('--epoch', type = int, default=config['train']['epoch'])
    parser.add_argument('--log_interval', type = int, default=config['train']['log_interval'])
    parser.add_argument('--batch_size', type = int, default=config['train']['batch_size'])
    parser.add_argument('--best_model_path', type = str, default=config['train']['best_model_path'])
    parser.add_argument('--patience', type = int, default=config['train']['patience'])

    parser.add_argument('--beta', type = float, default=config['param']['beta'])
    parser.add_argument('--l', type = float, default=config['param']['l'])
    parser.add_argument('--d_model', type = int, default=config['param']['d_model'])
    parser.add_argument('--state_size', type = int, default=config['param']['state_size'])
    parser.add_argument('--t_dim', type = int, default=config['param']['t_dim'])
    parser.add_argument('--text_dim', type = int, default=config['param']['text_dim'])    
    parser.add_argument('--img_dim', type = int, default=config['param']['img_dim'])
    parser.add_argument('--hidden_dim', type = int, default=config['param']['hidden_dim'])
    parser.add_argument('--feature_dim', type = int, default=config['param']['feature_dim'])
    parser.add_argument('--context_dim', type = int, default=config['param']['context_dim'])
    parser.add_argument('--confounder_dim', type = int, default=config['param']['confounder_dim'])

    opt, unknown = parser.parse_known_args()

    print('Parameters:')
    for key, value in vars(opt).items():
        print(f"{key}: {value}")
    
    return opt

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ =="__main__":
    set_random_seed(42)

    opt = parse_opt()

    if opt.data=='Terra':
        dataset = CMoSTDataset(image_dir=opt.image_dir,
                                text_dir=opt.text_dir,
                                time_series_path=opt.time_series_path,
                                his_len=opt.his_len,
                                pred_len=opt.pred_len,
                                lat_range=(50,60),
                                lon_range=(-8,2))

        train_dataset=CMoSTTrainDataset(dataset)
        val_dataset=CMoSTValDataset(dataset)
        test_dataset=CMoSTTestDataset(dataset)
        max=dataset.train_data_max
        min=dataset.train_data_min
        model=CMoST(opt)
    
        
    t1=time.time()
    t2 = train(model, train_dataset, val_dataset, max, min, opt)
    
    print(f"Total Train and Val time: {t2-t1:.2f}")

    path=opt.best_model_path + '/{}_{}_{}_{}.pth'.format(opt.data, opt.lr, opt.his_len, opt.pred_len)

    t3 = test(model, test_dataset, path, max, min, opt)
    print(f"Total Test time: {t3-t2:.2f}")


