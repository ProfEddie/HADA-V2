import torch.optim as optim
import time
from Lorentz_Models import *
from Utils import *
from Retrieval_Utils import i2t, t2i, evaluate_recall
from pathlib import Path
import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.nn as nn
from hyptorch.math import dist_matrix
import wandb
from tqdm import tqdm
from hyptorch.lorentz.manifold import Lorentz
from geoopt.optim import RiemannianAdam 
from geoopt import ManifoldParameter



def get_param(model):
    no_decay = ['bias', 'scale']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters() if p.requires_grad and not any(
                nd in n
                for nd in no_decay) and not isinstance(p, ManifoldParameter)
        ],
        'weight_decay': 0.0, 
    }, {
        'params': [
            p for n, p in model.named_parameters() if p.requires_grad and any(
                nd in n
                for nd in no_decay) or isinstance(p, ManifoldParameter)
        ],
        'weight_decay': 0.0
    }]

    return optimizer_grouped_parameters

class Controller(nn.Module):
    def __init__(self, config):
        super(Controller, self).__init__()
        # CONFIG
        self.config = config
        self.f1_in = config['f1_in']
        self.f2_in = config['f2_in']
        self.f1_out = config['f1_out']
        self.f2_out = config['f2_out']
        self.use_weighted_retrieval = config['use_weighted_retrieval']
        
        self.ft_gcn = config['ft_gcn']
        self.ft_com = config['ft_com']
        self.ft_itm = config['ft_itm']
        self.ft_trans = config['ft_trans']
        self.type_gcn = config['type_gcn']
        self.skip = config['skip']
        self.act_func = config['act_func']
        self.batch_norm = config['batch_norm']
        self.dropout = config['dropout']
        self.n_heads = config['n_heads']
        self.l2_norm = config['l2_norm']
        
        self.alpha = config['alpha']
        self.distill = config['distill']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.manifold = Lorentz(k=config['curvature'], learnable=True)
        
        self.optimizer_choice = config['optimizer_choice']
        self.weight_decay = config['weight_decay']
        self.early_stop = config['early_stop']
        self.thres_loss = config['thres_loss']
        self.learning_rate = config['learning_rate']
        self.device = config['device']
        self.min_lr = config['min_lr']
        self.factor = config['factor']
        self.patience = config['patience']
        self.T0 = config['T0']
        self.Tmul = config['Tmul']
        self.Tmax = config['Tmax']
        weight_nll_loss = config['weight_nll_loss']
        weight_itm_loss = config['weight_itm_loss']
        total_weight = weight_nll_loss + weight_itm_loss
        total_weight = 1
        self.weight_nll_loss = weight_nll_loss / total_weight
        self.weight_itm_loss = weight_itm_loss / total_weight
        # current_folder = os.path.dirname(os.path.abspath(__file__)) 
        self.out_dir = config['out_dir']
        self.pretrained_path = config['pretrained_path']
        self.temp = config['temp']
        self.dist_f = lambda x, y: -self.manifold.dist_batch(x, y.T)
        if self.temp > 0:
            self.temp_para = nn.Parameter(torch.ones([]) * self.temp)
            # self.temp_para = self.temp_para.to(self.device)
        else:
            self.temp_para = self.temp
        
        if self.use_weighted_retrieval:
            self.weight_1 = nn.Parameter(torch.ones([]) * 0.5)   
        else:
            self.weight_1 = 0
            
        # MODEL
        self.img_encoder = Lorentz_MH(self.manifold, f1_in=self.f1_in, f2_in=self.f2_in, f1_out=self.f1_out, f2_out=self.f2_out,
                              ft_trans=self.ft_trans, ft_gcn=self.ft_gcn, ft_com=self.ft_com,
                              type_gcn=self.type_gcn, skip=self.skip,
                              batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        self.cap_encoder = Lorentz_MH(self.manifold, f1_in=self.f1_in, f2_in=self.f2_in, f1_out=self.f1_out, f2_out=self.f2_out,
                              ft_trans=self.ft_trans, ft_gcn=self.ft_gcn, ft_com=self.ft_com,
                              type_gcn=self.type_gcn, skip=self.skip,
                              batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        self.discriminator = Lorentz_Discriminator(self.manifold, ft_in=self.ft_com[-1], ft_out=self.ft_itm,
                                           batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        self.img_encoder_m = Lorentz_MH(self.manifold, f1_in=self.f1_in, f2_in=self.f2_in, f1_out=self.f1_out, f2_out=self.f2_out,
                                ft_trans=self.ft_trans, ft_gcn=self.ft_gcn, ft_com=self.ft_com,
                                type_gcn=self.type_gcn, skip=self.skip,
                                batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        self.cap_encoder_m = Lorentz_MH(self.manifold, f1_in=self.f1_in, f2_in=self.f2_in, f1_out=self.f1_out, f2_out=self.f2_out,
                                ft_trans=self.ft_trans, ft_gcn=self.ft_gcn, ft_com=self.ft_com,
                                type_gcn=self.type_gcn, skip=self.skip,
                                batch_norm=self.batch_norm, dropout=self.dropout, act_func=self.act_func)
        self.img_encoder = self.img_encoder.to(self.device)
        self.cap_encoder = self.cap_encoder.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.img_encoder_m = self.img_encoder_m.to(self.device)
        self.cap_encoder_m = self.cap_encoder_m.to(self.device)
        
        self.model_pairs = [[self.img_encoder,self.img_encoder_m],
                            [self.cap_encoder,self.cap_encoder_m]]
        self.copy_params()
        
        ## PARAMS & OPTIMIZER
        self.params = []
        self.params += get_param(self.cap_encoder) 
        self.params += get_param(self.img_encoder) 
        self.params += get_param(self.discriminator) 
        self.params += get_param(self.img_encoder_m) 
        self.params += get_param(self.cap_encoder_m) 
        if self.use_weighted_retrieval:
            self.params += [{'params':self.weight_1}] 
        
        if self.temp > 0:
            self.params += [{'params':self.temp_para}]

        self.optimizer = RiemannianAdam(params=self.params, lr=self.learning_rate, stabilize=10)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=int(3625),
                                                   gamma=float(0.5)) 
            
        # create the queue
        self.register_buffer("image_queue", torch.randn(self.ft_com[-1], self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.ft_com[-1], self.queue_size))
        self.register_buffer("idx_queue", torch.full((1,self.queue_size),-100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
        self.register_buffer("image_queue_ori_1", torch.randn(self.f1_out, self.queue_size))
        self.register_buffer("text_queue_ori_1", torch.randn(self.f1_out, self.queue_size))
        
        self.image_queue = F.normalize(self.image_queue, dim=0).to(self.device)
        self.text_queue = F.normalize(self.text_queue, dim=0).to(self.device)
        self.image_queue_ori_1 = F.normalize(self.image_queue_ori_1, dim=0).to(self.device)
        self.text_queue_ori_1 = F.normalize(self.text_queue_ori_1, dim=0).to(self.device)
        
        ## Add wandb 
        self.dataset_name = config['dataset_name']
        self.config_path = config['config_path']
        self.config_name = self.config_path.split('/')[-1]
        self.log_enable = True 
        if self.log_enable:
            wandb.init(
                name='Lorentz',
                project='HADA-V2',  
                config={
                    "dataset": self.dataset_name,
                    "manifold": 'Lorentz'
                }
            )

    def log(self, stat:dict):
        if self.log_enable:
            wandb.log(stat)
            
    def train_mode(self):
        self.img_encoder.train()
        self.cap_encoder.train()
        self.img_encoder_m.train()
        self.cap_encoder_m.train()
        self.discriminator.train()
        
    def eval_mode(self):
        self.img_encoder.eval()
        self.cap_encoder.eval()
        self.img_encoder_m.eval()
        self.cap_encoder_m.eval()
        self.discriminator.eval()
        
    # ---------- LOAD TRAINED MODEL ---------
    def load_model(self, save_path=None, weight_only=False):
        #---- Load checkpoint 
        if save_path is not None:
            modelCheckpoint = torch.load(save_path)
            self.img_encoder.load_state_dict(modelCheckpoint['img_encoder'])
            self.cap_encoder.load_state_dict(modelCheckpoint['cap_encoder'])
            self.discriminator.load_state_dict(modelCheckpoint['discriminator'])
            if not weight_only:
                self.temp_para = modelCheckpoint['temp_para']
                if self.use_weighted_retrieval:
                    self.weight_1 = modelCheckpoint['weight_1']
            if 'optimizer' in list(modelCheckpoint.keys()):
                self.optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print(f"LOADED PRETRAINED MODEL AT {save_path}")
        else:
            print("TRAIN FROM SCRATCH")
            
    def save_model(self, loss, epochID=0, save_path='', optimizer=True):
        if optimizer:
            torch.save({'epoch': epochID,
                        'img_encoder': self.img_encoder.state_dict(),
                        'cap_encoder': self.cap_encoder.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'temp_para': self.temp_para,
                        'weight_1': self.weight_1,
                        'optimizer': self.optimizer.state_dict(),
                        'best_loss': loss}, f"{save_path}")
        else:
            torch.save({'epoch': epochID,
                        'img_encoder': self.img_encoder.state_dict(),
                        'cap_encoder': self.cap_encoder.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'temp_para': self.temp_para,
                        'weight_1': self.weight_1,
                        'best_loss': loss}, f"{save_path}")
    
    def itm_loss(self, imgs, cap, sim_i2t, sim_t2i, idx):
        # Find negative
        with torch.no_grad():
            bs = imgs.size(0)
            weights_i2t = F.softmax(sim_i2t[:,:bs]+1e-4,dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs]+1e-4,dim=1)
            mask = torch.eq(idx, idx.T)
            mask = mask.to(self.device)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 
        # select a negative image for each text
        img_enc_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            img_enc_neg.append(imgs[neg_idx])
        img_enc_neg = torch.stack(img_enc_neg,dim=0) 

        # select a negative text for each image
        cap_enc_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            cap_enc_neg.append(cap[neg_idx])
        cap_enc_neg = torch.stack(cap_enc_neg,dim=0)   

        cap_enc_all = torch.cat([cap, cap, cap_enc_neg],dim=0)     
        img_enc_all = torch.cat([imgs, img_enc_neg, imgs],dim=0)
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.float),torch.zeros(2*bs,dtype=torch.float)],
                               dim=0).view(-1,1).to(imgs.device)

        disc = self.discriminator(img_enc_all, cap_enc_all)
        loss_itm = F.binary_cross_entropy(disc, itm_labels)
        return loss_itm
    
    def forward_batch(self, batch):
        with torch.no_grad():
            self.temp_para.clamp_(0.001, 0.5)
            
        img_dict, cap_dict, img_id, instance_id = batch
        for key, value in img_dict.items():
            img_dict[key] = value.to(self.device)
        for key, value in cap_dict.items():
            cap_dict[key] = value.to(self.device) 
        
        idx = img_id.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()     
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
        sim_targets = sim_targets.to(self.device)
        
        img_enc = self.img_encoder(img_dict)
        cap_enc = self.cap_encoder(cap_dict)
        
        # Extend Original ALBEF
        img_enc_ori_1 = img_dict['ft_proj_ori_1']
        cap_enc_ori_1 = cap_dict['ft_proj_ori_1']

        # Normalize ???
        if self.l2_norm:
            img_enc = F.normalize(img_enc, dim=1)
            cap_enc = F.normalize(cap_enc, dim=1)
            
        with torch.no_grad():
            self._momentum_update()
            img_enc_m = self.img_encoder_m(img_dict) 
            cap_enc_m = self.cap_encoder_m(cap_dict) 
            if self.l2_norm:
                img_enc_m = F.normalize(img_enc_m, dim=1)
                cap_enc_m = F.normalize(cap_enc_m, dim=1)
            
            img_enc_all = torch.cat([img_enc_m.t(),self.image_queue.clone().detach()],dim=1) 
            cap_enc_all = torch.cat([cap_enc_m.t(),self.text_queue.clone().detach()],dim=1)
            
            img_enc_ori_1_all = torch.cat([img_enc_ori_1.t(),self.image_queue_ori_1.clone().detach()],dim=1) 
            cap_enc_ori_1_all = torch.cat([cap_enc_ori_1.t(),self.text_queue_ori_1.clone().detach()],dim=1)
            # dist_f = lambda x, y: -dist_matrix(x, y, c=0.1)
            # dist_f = lambda x, y: x @ y 

            if self.distill:               

                sim_i2t_m = self.dist_f(img_enc_m, cap_enc_all) / self.temp_para
                sim_t2i_m = self.dist_f(cap_enc_m, img_enc_all) / self.temp_para   

                sim_i2t_targets = self.alpha*F.softmax(sim_i2t_m, dim=1)+(1-self.alpha)*sim_targets
                sim_t2i_targets = self.alpha*F.softmax(sim_t2i_m, dim=1)+(1-self.alpha)*sim_targets 
        
        sim_i2t = self.dist_f(img_enc, cap_enc_all) / self.temp_para
        sim_t2i = self.dist_f(cap_enc, img_enc_all) / self.temp_para 
        sim_i2t_ori_1 = self.dist_f(img_enc_ori_1, cap_enc_ori_1_all) / self.temp_para
        sim_t2i_ori_1 = self.dist_f(cap_enc_ori_1, img_enc_ori_1_all) / self.temp_para 
        
        sim_i2t_combine = (1-self.weight_1)*sim_i2t + self.weight_1*sim_i2t_ori_1
        sim_t2i_combine = (1-self.weight_1)*sim_t2i + self.weight_1*sim_t2i_ori_1

        if self.distill:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t_combine, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i_combine, dim=1)*sim_t2i_targets,dim=1).mean() 
        else:
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t_combine, dim=1)*sim_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i_combine, dim=1)*sim_targets,dim=1).mean()   

        loss_nll = 0.5*loss_i2t+0.5*loss_t2i
        
        self._dequeue_and_enqueue(img_enc_m, cap_enc_m, img_enc_ori_1, cap_enc_ori_1, idx)
        
        loss_itm = self.itm_loss(img_enc, cap_enc, sim_i2t, sim_t2i, idx)
        result = {'img_enc':img_enc, 'cap_enc': cap_enc, 'img_id': img_id, 
                  'loss_nll': loss_nll, 'loss_itm': loss_itm}
        return result
    
    def train_epoch(self, dataloader, epochID=0):
        self.train_mode()
        loss_all_report = 0
        loss_nll_report = 0
        loss_itm_report = 0
        numb_iter = len(dataloader)
        progress_bar = tqdm(range(numb_iter))
        for idx, batch in enumerate(dataloader):
            print(idx/numb_iter)
            # if idx > 10: # DEBUG
            #     break
            # Forward
            if epochID > 0:
                self.alpha = self.config['alpha']
            else:
                self.alpha = self.config['alpha']*min(1,idx/len(dataloader))
                
            # Clamp weight based on epoch
            with torch.no_grad():
                # self.weight_1.clamp_(0.1, min(0.9, 0.7 + epochID*0.05))
                if self.use_weighted_retrieval:
                    self.weight_1.clamp_(min=0.1, max=0.9)
                
            br = self.forward_batch(batch)
            loss_nll, loss_itm = br['loss_nll'], br['loss_itm']
            all_loss = self.weight_nll_loss*loss_nll + self.weight_itm_loss*loss_itm
            self.log({'current loss': all_loss.item()})
            print(f'current loss: {all_loss.item()}')
            
            # Update
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            loss_all_report += all_loss.item()
            loss_nll_report += loss_nll.item()
            loss_itm_report += loss_itm.item()
            progress_bar.update(1)

        loss_all_report = round(loss_all_report/(idx+1), 6)
        loss_nll_report = round(loss_nll_report/(idx+1), 6)
        loss_itm_report = round(loss_itm_report/(idx+1), 6)
        loss_dict = {'all': loss_all_report, 'nll': loss_nll_report, 'itm': loss_itm_report}
        return loss_dict
    
    def evaluate_multimodal(self, dataset, apply_temp=True, return_sim=False):
        dataset.set_branch(branch='img')
        dataloader_img = make_dataloader(dataset, branch='img', batch_size=int(self.config['batch_size']/2), shuffle=False)
        img_enc, img_enc_ori_1, img_enc_ori_2 = self.eval_encode(dataloader_img, branch='img')
        
        dataset.set_branch(branch='txt')
        dataloader_txt = make_dataloader(dataset, branch='txt', batch_size=int(self.config['batch_size']/2), shuffle=False)
        cap_enc, cap_enc_ori_1, cap_enc_ori_2 = self.eval_encode(dataloader_txt, branch='txt')
       
        with torch.no_grad():
            if apply_temp:
                sims = self.dist_f(img_enc , cap_enc.T) / self.temp_para
                sims_ori_1 = self.dist_f(img_enc_ori_1 , cap_enc_ori_1.T) / self.temp_para
            else:
                sims = self.dist_f(img_enc , cap_enc.T) 
                sims_ori_1 = self.dist_f(img_enc_ori_1 , cap_enc_ori_1.T) 
            sims = (1-self.weight_1)*sims + self.weight_1 * sims_ori_1
            # sims = img_enc_ori_1 @ cap_enc_ori_1.T # for BLIP
            # sims = img_enc_ori_2 @ cap_enc_ori_2.T # for CLIP
            sims = sims.cpu().numpy()
        r1i, r5i, r10i, r1t, r5t, r10t = evaluate_recall(sims, mode='both')
        rall = r1i + r5i + r10i + r1t + r5t + r10t
        loss = 6 - rall
        if return_sim:
            return (r1i, r5i, r10i, r1t, r5t, r10t), loss, sims
        else:
            return (r1i, r5i, r10i, r1t, r5t, r10t), loss
            
    def train(self, dataset_train,  dataset_val, num_epoch=10, model_name='best'):
        if 'coco' in self.dataset_name:
            drop_last = True
        else:
            drop_last = False
        dataloader_train = make_dataloader(dataset_train, branch='both', batch_size=self.config['batch_size'], 
                                           shuffle=True, drop_last=drop_last)
        self.n_iters = len(dataloader_train)
        
        save_dir = f"{self.out_dir}/{model_name}"
        Path(f"{save_dir}").mkdir(parents=True, exist_ok=True)
        
        self.load_model(save_path=self.pretrained_path, weight_only=True)
        
         ## REPORT ##
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampLaunch = timestampDate + '-' + timestampTime
        
        count_change_loss = 0
        
        loss_best = 100000
    
        
        # with mlflow.start_run(experiment_id = self.experiment_id, 
                            #   run_name = self.config_name):
        if self.log_enable:
            wandb.log_artifact(self.config_path)
        params = {'Numb_Para': self.count_parameters()}
        
        self.log(params)
        best_epoch = 0
        
        for idx_epoch in tqdm(range(num_epoch)):
            loss_tr_dict = self.train_epoch(dataloader_train, idx_epoch)
            self.log(loss_tr_dict)
            loss_tr_all, loss_tr_nll, loss_tr_itm = loss_tr_dict['all'], loss_tr_dict['nll'], loss_tr_dict['itm']

            with torch.no_grad():
                timestampTime = time.strftime("%H%M%S")
                timestampDate = time.strftime("%d%m%Y")
                timestampEND = timestampDate + '-' + timestampTime
                apply_temp = True if self.temp > 0 else False

                if dataset_val is not None: 
                    r, loss_rall = self.evaluate_multimodal(dataset_val, apply_temp, return_sim=False)
                    r1i, r5i, r10i, r1t, r5t, r10t = r
                    loss_update = loss_rall
                else:
                    loss_update = loss_tr_all

            if self.T0 <= 0 and self.Tmax <= 0:
                self.scheduler.step(loss_update)

            if loss_update <= (loss_best-self.thres_loss):
                best_epoch = idx_epoch
                count_change_loss = 0
                print(f"[SAVE MODEL]")
                self.save_model(loss=loss_rall, epochID=idx_epoch, save_path=f"{save_dir}/best.pth.tar", optimizer=False)
                loss_best = loss_update
                info_txt = f"Epoch {idx_epoch}/{num_epoch-1} [{timestampEND} --- SAVE MODEL]\n"
                metrics = {'temp_para': self.temp_para.item(),
                            'best_epoch':best_epoch, 
                            'current_epoch':idx_epoch, 
                            'Ri': np.round(r1i+r5i+r10i, 6),
                            'Rt': np.round(r1t+r5t+r10t, 6),
                            'Rall': np.round(r1i+r5i+r10i+r1t+r5t+r10t, 6),
                            'R1i': np.round(r1i, 6),
                            'R5i': np.round(r5i, 6),
                            'R10i': np.round(r10i,6),
                            'R1t': np.round(r1t, 6),
                            'R5t': np.round(r5t, 6),
                            'R10t': np.round(r10t,6),
                            'Ctemp_para': self.temp_para.item(),
                            'eval_loss': loss_rall 
                            }
                if self.use_weighted_retrieval:
                    metrics['weight_1'] = self.weight_1.item()
                else:
                    metrics['weight_1'] = self.weight_1
                self.log(metrics)
            else:
                self.save_model(loss=loss_update, epochID=idx_epoch, save_path=f"{save_dir}/current.pth.tar", optimizer=False)
                metrics = {'Ctemp_para': self.temp_para.item(),
                            'current_epoch':idx_epoch,
                            }
                if self.use_weighted_retrieval:
                    metrics['Cweight_1'] = self.weight_1.item()
                else:
                    metrics['Cweight_1'] = self.weight_1
                self.log(metrics)
                count_change_loss += 1
                info_txt = f"Epoch {idx_epoch}/{num_epoch-1} [{timestampEND}]\n"

            info_txt += f"Loss Train: {loss_tr_all}\nLoss Train Nll: {loss_tr_nll}\nLoss Train Itm: {loss_tr_itm}\n"
            
            info_txt += f"R1i: {np.round(r1i,6)}\nR5i: {np.round(r5i,6)}\nR10i: {np.round(r10i,6)}\n"
            info_txt += f"R1t: {np.round(r1t,6)}\nR5t: {np.round(r5t,6)}\nR10t: {np.round(r10t,6)}\n"
            info_txt += f"Ri: {np.round(r1i+r5i+r10i,6)}\nRt: {np.round(r1t+r5t+r10t,6)}\n"
            info_txt += f"Rall: {np.round(r1i+r5i+r10i+r1t+r5t+r10t,6)}\n"
  

            if count_change_loss >= self.early_stop:
                print(f'Early stopping: {count_change_loss} epoch not decrease the loss')
                info_txt += "[EARLY STOPPING]\n"
                break
            print(info_txt)
            
        
    def count_parameters(self, trainable=True):
        total = 0
        if trainable:
            total += sum(p.numel() for p in self.img_encoder.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.cap_encoder.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.img_encoder_m.parameters() if p.requires_grad)
            total += sum(p.numel() for p in self.cap_encoder_m.parameters() if p.requires_grad)
        else:
            total += sum(p.numel() for p in self.img_encoder.parameters())
            total += sum(p.numel() for p in self.cap_encoder.parameters())
            total += sum(p.numel() for p in self.discriminator.parameters())
            total += sum(p.numel() for p in self.img_encoder_m.parameters())
            total += sum(p.numel() for p in self.cap_encoder_m.parameters())
        return total
    
    def eval_encode(self, dataloader, branch='img'):
        self.eval_mode()
        list_enc = torch.tensor(()).to(self.device)
        list_enc_ori_1 = torch.tensor(()).to(self.device)
        list_enc_ori_2 = torch.tensor(()).to(self.device)
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                x_dict,_ = batch
                for key, value in x_dict.items():
                    x_dict[key] = value.to(self.device)
                if branch == 'img':
                    x_enc = self.img_encoder(x_dict)
                else:
                    x_enc = self.cap_encoder(x_dict)
                x_enc_ori_1 = x_dict['ft_proj_ori_1']
                x_enc_ori_2 = x_dict['ft_proj_ori_2']
                if self.l2_norm:
                    x_enc = F.normalize(x_enc, dim=1)
                list_enc = torch.cat((list_enc, x_enc), 0)
                list_enc_ori_1 = torch.cat((list_enc_ori_1, x_enc_ori_1), 0)
                list_enc_ori_2 = torch.cat((list_enc_ori_2, x_enc_ori_2), 0)
        return list_enc, list_enc_ori_1, list_enc_ori_2
    
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, image_feat_1, text_feat_1, idx):
        batch_size = image_feat.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        self.image_queue_ori_1[:, ptr:ptr + batch_size] = image_feat_1.T
        self.text_queue_ori_1[:, ptr:ptr + batch_size] = text_feat_1.T
        self.idx_queue[:, ptr:ptr + batch_size] = idx.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr  