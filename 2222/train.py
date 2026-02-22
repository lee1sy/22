import os
import torch
import numpy as np
from torch import nn
from tools.runner import Trainer
from tools.utils import load_config, check_path, check_dir
from modules.GS import GaussianFusionNet, set_global_seed
from torch.utils.data import DataLoader
from dataset.NuScenesDataset import TripletDataset, DatabaseQueryDataset, collate_fn
from torchvision.transforms import transforms

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    cfg = load_config('config/config.yaml')
    model_cfg = cfg.get('model', {})
    set_global_seed(model_cfg.get('seed', 42))

    data_root_dir = cfg['data']['data_root_dir']
    database_path = cfg['data']['database_path']
    train_query_path = cfg['data']['train_query_path']
    test_query_path = cfg['data']['test_query_path']
    val_query_path = cfg['data']['val_query_path']
    info_path = cfg['data']['info_path']
    gaussian_path = cfg['data'].get('gaussian_path', None)

    nonTrivPosDistThres = cfg['runner']['nonTrivPosDistThres']
    posDistThr = cfg['runner']['posDistThr']
    nNeg = 10  
    nNegSample = cfg['runner']['nNegSample']
    margin = cfg['runner']['margin']
    resize = cfg['runner']['resize']
    lr = cfg['runner']['lr']
    step_size = cfg['runner']['step_size']
    gamma = cfg['runner']['gamma']
    num_epochs = cfg['runner']['num_epochs']
    freeze_visual = cfg['runner'].get('freeze_visual', True)  # 默认冻结
    
    num_workers_train = 16  
    num_workers_test = 16
    
    ckpt_dir = cfg['runner']['ckpt_dir']
    result_dir = cfg['runner']['result_dir']
    cache_dir = cfg['runner']['cache_dir']
    log_dir = cfg['runner']['log_dir']

    check_path(data_root_dir, database_path, train_query_path, test_query_path, val_query_path, info_path)
    check_dir(ckpt_dir, result_dir, cache_dir, log_dir)

    img_transforms = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = TripletDataset(
        data_root_dir, database_path, train_query_path, info_path, cache_dir,
        img_transforms, nNeg, nNegSample, nonTrivPosDistThres, posDistThr, margin,
        gaussian_path=gaussian_path, resize=resize
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=1, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers_train, pin_memory=True
    )

    whole_train_set = DatabaseQueryDataset(
        data_root_dir, database_path, train_query_path, info_path,
        img_transforms, nonTrivPosDistThres,
        gaussian_path=gaussian_path, resize=resize
    )
    whole_train_loader = DataLoader(
        dataset=whole_train_set, batch_size=8, shuffle=False,
        num_workers=num_workers_test, pin_memory=True
    )

    whole_val_set = DatabaseQueryDataset(
        data_root_dir, database_path, val_query_path, info_path,
        img_transforms, nonTrivPosDistThres,
        gaussian_path=gaussian_path, resize=resize
    )
    whole_val_loader = DataLoader(
        dataset=whole_val_set, batch_size=8, shuffle=False,
        num_workers=num_workers_test, pin_memory=True
    )

    model = GaussianFusionNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = Trainer(
        model=model, 
        train_loader=train_loader, 
        whole_train_loader=whole_train_loader, 
        whole_val_set=whole_val_set, 
        whole_val_loader=whole_val_loader, 
        device=device,
        num_epochs=num_epochs, 
        ckpt_dir=ckpt_dir, 
        cache_dir=cache_dir, 
        log_dir=log_dir,
        lr=lr, 
        step_size=step_size, 
        gamma=gamma, 
        margin=margin,
        freeze_visual=freeze_visual
    )
    
    trainer.train()

if __name__ == '__main__':
    main()