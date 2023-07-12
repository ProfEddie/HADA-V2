import json
import joblib
import argparse
import numpy as np
from Retrieval_Utils import i2t, t2i, evaluate_recall
from Utils import write_to_file
import torch
# import mlflow
import Utils as ut
from Controller_Hyp import Controller as Ctr
from Controller_Hyp import Controller as Ctr_Both
from lavis.models import load_model_and_preprocess, load_model, load_preprocess
from lavis.datasets.builders import load_dataset

# mlflow.set_tracking_uri('http://localhost:1409')

DATASET_NAME = 'flickr30k'
if DATASET_NAME == 'flickr30k':
    itr_dataset = load_dataset(DATASET_NAME,vis_path='/mnt/data/itr_dataset/dataset/flickr30k_images/')
if DATASET_NAME == 'mscoco':
    itr_dataset = load_dataset("coco_retrieval",vis_path='/mnt/data/itr_dataset/dataset/coco_images/')

def get_model_dict(config):
    print(f"Loading {config['model_1_name']}-{config['model_1_type']} Preprocessor ...")
    model_1, vis_processors_1, txt_processors_1 = load_model_and_preprocess(name=config['model_1_name'], 
                                                                            model_type=config['model_1_type'], 
                                                                            is_eval=True, 
                                                                            device=config['device'])
    model_1_dict = {'model': model_1, 'vis_processors': vis_processors_1, 'txt_processors': txt_processors_1}

    model_2, vis_processors_2, txt_processors_2 = load_model_and_preprocess(name=config['model_2_name'], 
                                                                            model_type=config['model_2_type'], 
                                                                            is_eval=True, 
                                                                            device=config['device'])
    model_2_dict = {'model': model_2, 'vis_processors': vis_processors_2, 'txt_processors': txt_processors_2}
    return model_1_dict, model_2_dict

def run_train(args):
    print(f"RUN TRAIN")
    config_path = args.config_path
    config_name = config_path.split('/')[-1][:-4]
    dataset_name = config_path.split('/')[-2]
    config = ut.load_config(config_path)
    config['util_norm'] = False
    config['dataset_name'] = dataset_name
    config['config_path'] = config_path
    config['out_dir'] = f"{config['out_dir']}/{dataset_name}"
    model_1_dict, model_2_dict = get_model_dict(config)
    
    train_dataset = ut.Retrieval_Dataset(config=config, dataset=itr_dataset, 
                                         model_1_dict=model_1_dict, model_2_dict=model_2_dict, 
                                         type_dataset='train')
    val_dataset = ut.Retrieval_Dataset(config=config, dataset=itr_dataset, 
                                       model_1_dict=model_1_dict, model_2_dict=model_2_dict, 
                                       type_dataset='val')
    
    niters = int(int(np.ceil(len(train_dataset) / config['batch_size'])))
    
    if config['Tmax'] > 0:
        config['Tmax'] = config['Tmax'] * niters
    
    if 'both' in config_path.lower():
        
        print("Using both backbones to fuse")
        controller = Ctr_Both(config)
    else:
        print("Using 1 dominant backbones to fuse")
        controller = Ctr(config)
    
    total_para = controller.count_parameters()
    print(f"Trainable Paras: {total_para}")
    controller.train(dataset_train=train_dataset,  dataset_val=val_dataset,
                     num_epoch=config['num_epoch'], model_name=config_name)
    

def run_evaluate(args):
    config_path = args.config_path
    config_name = config_path.split('/')[-1][:-4]
    dataset_name = config_path.split('/')[-2]
    
    print(f"PERFORM EVALUATE")
    config = ut.load_config(config_path)
    config['out_dir'] = f"{config['out_dir']}/{dataset_name}"    
    save_path = f"{config['out_dir']}/{config_name}/best.pth.tar"
    config['config_path'] = config_path
    config['util_norm'] = False
    config['dataset_name'] = dataset_name
    model_1_dict, model_2_dict = get_model_dict(config)
    
    test_dataset = ut.Retrieval_Dataset(config=config, dataset=itr_dataset, 
                                        model_1_dict=model_1_dict, model_2_dict=model_2_dict, 
                                        type_dataset='test')
    
    if 'both' in config_path.lower():
        print("Using both backbones to fuse")
        controller = Ctr_Both(config)
    else:
        print("Using 1 dominant backbones to fuse")
        controller = Ctr(config)
        
    controller.load_model(save_path)
    controller.eval_mode()
    
    apply_temp = True if controller.temp > 0 else False
    with torch.no_grad():
        r, loss_rall = controller.evaluate_multimodal(test_dataset, apply_temp, return_sim=False)
        r1i, r5i, r10i, r1t, r5t, r10t = r
        
    info_txt = f"R1i: {r1i}\nR5i: {r5i}\nR10i: {r10i}\n"
    info_txt += f"R1t: {r1t}\nR5t: {r5t}\nR10t: {r10t}\n"
    info_txt += f"Ri: {r1i+r5i+r10i}\nRt: {r1t+r5t+r10t}\n"
    info_txt += f"Rall: {r1i+r5i+r10i+r1t+r5t+r10t}\n"
    info_txt += f"LoRe: {loss_rall}\n"
    write_to_file(f"{config['out_dir']}/{config_name}/TestReport.log", info_txt)     
    print(info_txt)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='HADA_m/Config/C5.yml', help='yml file of the config')
    # parser.add_argument('-md', '--model_type', type=str, default='LiFu_m', help='structure of the model')
    parser.add_argument('-rm', '--run_mode', type=str, default='train', help='train: train and test\ntest: only test')
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    print(f"CONFIG: {CONFIG_PATH.split('/')[-1]}")
    if args.run_mode == 'train':
        run_train(args)
        run_evaluate(args)
    if args.run_mode == 'test':
        run_evaluate(args)