from lavis.models import load_model_and_preprocess, load_model, load_preprocess
from lavis.datasets.builders import load_dataset
import Utils as ut
from coattention import CoAttention

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
    
    
    
if __name__ == '__main__':
    itr_dataset = load_dataset('flickr30k',vis_path='/mnt/data/itr_dataset/dataset/flickr30k_images/')
    print(type(itr_dataset))
    print(itr_dataset)
        

    subset = {
       'train':itr_dataset['train'] ,
       'test':itr_dataset['test'], 
       'val':itr_dataset['val'], 
    }

    print(subset)
    
    config_path = 'Config/flickr30k/C1.yml' 
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
    val_dataset = ut.Retrieval_Dataset(config=config, dataset=itr_dataset, 
                                        model_1_dict=model_1_dict, model_2_dict=model_2_dict, 
                                        type_dataset='val')
    model = CoAttention(768, 768)
    # img_rep1 = train_dataset[0]['cap']['ft_1']
    # img_rep_proj_1= train_dataset[0]['cap']['ft_proj_2']
    # img_rep2= train_dataset[0]['cap']['ft_2']
    # img_rep_proj_2= train_dataset[0]['cap']['ft_proj_2']
    # print(train_dataset[0]['img'].keys())

    # print(img_rep1.shape)
    # print(img_rep2.shape)
    # print(img_rep_proj_1.shape)
    # print(img_rep_proj_2.shape)

    # model(img_rep1, img_rep2)