import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from data_transforms import TransformSZ
from datasets import EvalDataset
import networks
import json
import os
import argparse
import torch.nn.functional as F



root = '/nfs/turbo/lsa-jbregman/campratt/UDA/'
sz_norms = np.genfromtxt('/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/sz_norms.txt')
sz_mean = sz_norms[0]
sz_std = sz_norms[1]

freq_norms = np.genfromtxt('/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/freq_norms.txt').T
freq_mean = freq_norms[0][:, np.newaxis, np.newaxis]
freq_std = freq_norms[1][:, np.newaxis, np.newaxis]

print(freq_mean)
print(freq_std)

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Hyperparameter Tuning")
    parser.add_argument('--fn_config', type=str, required=False, help='Path to the config file', default=None)
    parser.add_argument('--model', type=str, required=False, help='Network architecture', default='NestedUNet3D')
    parser.add_argument('--model_size', type=str, required=False, help='Network architecture size', default='vsmall')
    parser.add_argument('--N_sim', type=int, required=False, help='Number of all-sky simulations maps', default=100)
    parser.add_argument('--N_dim', type=float, required=False, help='Height/Width dimension of images', default=128)
    parser.add_argument('--N_pred', type=float, required=False, help='Number of cutout images used for evaluation', default=10)
    parser.add_argument('--cross_val_id', type=int, required=False, help='Number ID defining how to split the data for cross-validation', default=0)
    parser.add_argument('--sweep_id', type=int, required=False, help='Number ID defining the hyperparameter sweep', default=0)
    parser.add_argument('--job_id', type=int, required=False, help='Number ID defining the job in the hyperparameter sweep', default=5)
    parser.add_argument('--deep_supervision', type=bool, required=False, help='Weather or not to use deep supervision of the outputs', default=True)
    parser.add_argument('--sz_only', type=bool, required=False, help='Whether or not to use self-supervison or supervised only', default=True)
    parser.add_argument('--train_or_val', type=str, required=False, help='Evaluate the training or validation set', default='val')
    parser.add_argument('--train_domain', type=str, required=False, choices=['sim','simreal','mix'], help='Domain of data to perform training', default='simreal')
    parser.add_argument('--val_domain', type=str, required=False, choices=['sim','simreal','mix'], help='Domain of data to perform evaluation', default='simreal')
    parser.add_argument('--infer_domain', type=str, required=False, choices=['sim','simreal','mix','real'], help='Domain of data to perform inference', default='simreal')


    args = parser.parse_args()
    return args

def load_config(fname):
    with open(fname, 'r') as json_file:
        config = json.load(json_file)
    return config

def custom_collate_fn(batch):
    """
    Custom collate function to handle cases where 'z' might be None.
    """
    # Separate x, y, and z from the batch
    ID_batch = torch.tensor([item[0] for item in batch])
    x_batch = [item[1] for item in batch]
    y_batch = [item[2] for item in batch]
    z_batch = [item[3] for item in batch]
    
    # Stack x and y into tensors
    
    x_batch = torch.stack(x_batch)
    y_batch = torch.stack(y_batch)
    
    # Handle z
    if any(z is not None for z in z_batch):
        # Replace None with a placeholder (e.g., a zero tensor or any appropriate default)
        #z_batch = [torch.zeros_like(x_batch[0]) if z is None else z for z in z_batch]
        z_batch = torch.stack(z_batch)
    else:
        # If all z are None, return None
        z_batch = None
    
    #print(x_batch.size(),y_batch.size(),z_batch.size())
    return ID_batch, x_batch, y_batch, z_batch

#def upscale_tensors(tensor_list):
#    # Make sure the list is not empty
#    if not tensor_list:
#        raise ValueError("The input list is empty.")
#    
#    # Get the target shape from the first tensor and ensure it has at least 3 dimensions (C, H, W)
#    target_shape = tensor_list[0].shape
#    
#    # Ensure tensors have the same number of dimensions
#    # Here we consider 2D and 3D tensors, if tensors are 4D with batch sizes 
#    # (batch_size, channels, height, width)
#    if len(target_shape) < 3:
#        raise ValueError("Target tensor must have at least 3 dimensions.")
#    
#    # Rescale each tensor to have the same shape as the first tensor
#    # Add batch dimension if necessary
#    scaled_tensors = []
#    for tensor in tensor_list:
#        if len(tensor.shape) == len(target_shape):
#            scaled_tensors.append(F.interpolate(tensor.unsqueeze(0), size=target_shape[1:], mode='nearest').squeeze(0))
#        else:  # If tensor lacks the batch dimension
#            scaled_tensors.append(F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=target_shape).squeeze(0).squeeze(0))
#    
#    return scaled_tensors

def upscale_tensors(tensor_list):
    """
    Upscale a list of tensors to match the shape of the first tensor in the list.
    Handles tensors of shape BxCxTxWxH.
    
    Args:
        tensor_list (list of torch.Tensor): List of tensors to be upscaled.
    
    Returns:
        list of torch.Tensor: List of tensors upscaled to the shape of the first tensor.
    """
    # Make sure the list is not empty
    if not tensor_list:
        raise ValueError("The input list is empty.")
    
    # Get the target shape from the first tensor
    target_tensor = tensor_list[0]
    if len(target_tensor.shape) != 5:
        raise ValueError("All tensors must have shape BxCxTxWxH.")
    
    target_shape = target_tensor.shape
    
    # Rescale each tensor to match the target shape
    scaled_tensors = []
    for tensor in tensor_list:
        if len(tensor.shape) != 5:
            raise ValueError(f"Tensor shape {tensor.shape} does not match the expected BxCxTxWxH format.")
        
        # Match the target shape using interpolation
        scaled_tensor = F.interpolate(
            tensor, 
            size=target_shape[2:],  # Resize time, width, and height
            mode='nearest',  # Use nearest neighbor interpolation
            align_corners=None if 'linear' not in 'nearest' else True  # Avoid align_corners for non-linear modes
        )
        scaled_tensors.append(scaled_tensor)
    
    return scaled_tensors

def eval_map(Net, Data, load_model, device, output_dir, infer_domain='sim', idx=0):
    idx = idx.cpu().numpy()[0]
    model = Net
    state_dict = torch.load(load_model, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # Remove "module." if present
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    if infer_domain == 'sim':
        x, y, _ = Data
        x = x.to(device).float()
        y = y.to(device).float()
        with torch.no_grad():
            #pred = Net.evaluate_map(x)[0]
            pred = Net(x)
            pred = pred.detach().cpu().numpy()
            
            
            
            #np.save(root + 'y_eval_ss_cmb/' + f'y_{idx}.npy', pred)
            np.save(output_dir + f'y_{idx}_sim_pred.npy', pred)
            #np.save(output_dir + f'logvar_{idx}_sim_pred.npy', log_sigma2)
            np.save(output_dir + f'y_{idx}_sim_true.npy', y.detach().cpu().numpy())
            np.save(output_dir + f'x_{idx}_sim.npy', x[0].detach().cpu().numpy())
            #np.save(output_dir + f'x_{idx}_sim_pred.npy',ss[0].detach().cpu().numpy())


    elif infer_domain == 'real':
        x, _, _ = Data
        x = x.to(device).float()
        with torch.no_grad():
            #pred = Net.evaluate_map(x)[0]
            pred = Net(x)
            pred = pred.detach().cpu().numpy()
            #log_sigma2 = log_sigma2.detach().cpu().numpy()
            np.save(output_dir + f'y_{idx}_real_pred.npy', pred)
            #np.save(output_dir + f'logvar_{idx}_real_pred.npy', log_sigma2)
            np.save(output_dir + f'x_{idx}_real.npy', x[0].detach().cpu().numpy())
            #np.save(output_dir + f'x_{idx}_real_pred.npy', ss[0].detach().cpu().numpy())


    elif infer_domain == 'simreal':
        x, y, mask = Data
        x = x.to(device).float()
        y = y.to(device).float()
        mask = mask.to(device).float()
        with torch.no_grad():
            #pred = Net.evaluate_map(x)[0]
            pred = Net(x)
            pred = pred.detach().cpu().numpy()
            #log_sigma2 = log_sigma2.detach().cpu().numpy()
            np.save(output_dir + f'y_{idx}_simreal_pred.npy', pred)
            np.save(output_dir + f'y_{idx}_simreal_true.npy', y.detach().cpu().numpy())
            np.save(output_dir + f'mask_{idx}_simreal.npy', mask.detach().cpu().numpy())
            np.save(output_dir + f'x_{idx}_simreal.npy', x[0].detach().cpu().numpy())
            #np.save(output_dir + f'x_{idx}_real_pred.npy', ss[0].detach().cpu().numpy())


            


def eval_features(Net, Data, load_model, device, output_dir, idx=0, infer_domain='sim'):
    model = Net
    state_dict = torch.load(load_model, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # Remove "module." if present
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()

    x = Data.to(device).float()
        
    with torch.no_grad():
        #pred, psis = Net.evaluate_features(x,return_attn=True)
        pred, psis = Net(x,return_attn=True)
        pred = pred.detach().cpu().numpy()
        psi = psis[-1].detach().cpu().numpy()
        psis = upscale_tensors(psis)
        psis = torch.stack(psis).detach().cpu().numpy()
        idx = idx.cpu().numpy()[0]
        #np.save(root + 'y_pred_ss/' + f'f_{idx}.npy', pred)
        np.save(output_dir + f'feat_{infer_domain}_{idx}.npy', pred)
        np.save(output_dir + f'psis_{infer_domain}_{idx}.npy', psis)


def get_ranks(arr,return_indices=False):
    if return_indices:
        order = arr.argsort()
        ranks = order.argsort()
        return ranks, order
    else:
        order = arr.argsort()
        ranks = order.argsort()
        return ranks
    

def main():
    args = parse_args()

    if args.fn_config is not None:
        config = load_config(args.fn_config)

    else:
        config = {}

    for key, value in vars(args).items():
        if key not in config.keys():
            config[key] = value

    print(config)

    # Miscellanous
    #output_dir = config['output_dir']
    
    model = config['model']
    N_sim = config['N_sim']
    N_dim = config['N_dim']
    N_pred = config['N_pred']
    cross_val_id = config['cross_val_id']
    cross_val_id = int(cross_val_id)
    job_id = config['job_id']
    sweep_id = config['sweep_id']
    deep_supervision = config['deep_supervision']
    sz_only = config['sz_only']
    train_or_val = config['train_or_val']
    model_size = config['model_size']
    #sim_or_real = config['sim_or_real']
    train_domain = config['train_domain']
    val_domain = config['val_domain']
    infer_domain = config['infer_domain']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #output_dir = root + 'y_eval_ss_sz/' + f'model={model}_crossval={cross_val_id}/'
    #root_scratch = '/scratch/jbregman_root/jbregman0/campratt/UDA/'
    root_scratch = '/nfs/turbolsa-jbregman/campratt/UDA/'
    
    output_dir = root_scratch + f'y_eval_sz_train={train_domain}_val={val_domain}_infer={infer_domain}/' + f'model={model}_sweepid={sweep_id}_crossval={cross_val_id}_jobid={job_id}_{train_or_val}/'
    print('output_dir: %s'%output_dir)
    
    models_dir = root_scratch + f'models_sz_train={train_domain}_val={val_domain}/' + f'model={model}_sweepid={sweep_id}_crossval={cross_val_id}_jobid={job_id}/'
    os.makedirs(output_dir, exist_ok=True)

    # Defined by the setup of the experiment
    N_IDs_real_max = 700
    N_IDs_sim_max = 700*N_sim

    IDs_real_all = np.arange(0, int(round(N_IDs_real_max,1)))
    IDs_sim_all = np.arange(0, int(round(N_IDs_sim_max,1)))

    # Split into 70/30 train/val
    n_parts = 3
    def get_indices(data, n_parts):

        inds_all = np.arange(len(data))
        split_size = len(data) // n_parts
        indices = [i * split_size for i in range(0, n_parts)]
        indices.append(len(data))
        inds_train, inds_val = [],[]
        for i in range(len(indices)-1):
            i_start = indices[i]
            i_end = indices[i+1]
            inds_val.append(np.arange(i_start,i_end))
            inds_train.append(inds_all[~np.isin(inds_all,inds_val[i])])
        return [np.array(inds_train), np.array(inds_val)]

    inds_train_real, inds_val_real = get_indices(IDs_real_all, n_parts)
    inds_train_sim, inds_val_sim = get_indices(IDs_sim_all, n_parts)

    print('train: ',inds_train_sim)
    print('val: ', inds_val_sim) 

    

    
        

    IDs_real_train, IDs_real_val = inds_train_real[cross_val_id], inds_val_real[cross_val_id]
    IDs_sim_train, IDs_sim_val = inds_train_sim[cross_val_id], inds_val_sim[cross_val_id]

    Net = getattr(networks, model)(model_size=model_size).to(device)

    sim_x_dir = '/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/x_sim/'
    sim_y_dir = '/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/y_sim_sz/'
    real_x_dir = '/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/x_real/'
    #simreal_x_dir = '/scratch/jbregman_root/jbregman0/campratt/DANN_SZ/x_simreal/'
    #simreal_y_dir = '/scratch/jbregman_root/jbregman0/campratt/DANN_SZ/y_simreal_sz/'
    #simreal_mask_dir = '/scratch/jbregman_root/jbregman0/campratt/DANN_SZ/mask_simreal/'
    simreal_x_dir = '/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/x_simreal/'
    simreal_y_dir = '/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/y_simreal/'
    simreal_mask_dir = '/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/mask_simreal/'
    

    if infer_domain == 'mix':
        signal_loss_sim = np.genfromtxt(models_dir + 'err_signal_sim_val.txt')
        signal_loss_simreal = np.genfromtxt(models_dir + 'err_signal_simreal_val.txt')

        mae_loss_sim = np.genfromtxt(models_dir + 'mae_sim_val.txt')

        var_loss_sim = np.genfromtxt(models_dir + 'var_sim_val.txt')
        var_loss_simreal = np.genfromtxt(models_dir + 'var_simreal_val.txt')

        #signal_rank_loss = (get_ranks(signal_loss_sim) + get_ranks(signal_loss_simreal))/2
        signal_rank_loss = get_ranks(signal_loss_simreal)
        mae_rank_loss = get_ranks(mae_loss_sim)
        #var_rank_loss = (get_ranks(var_loss_sim) + get_ranks(var_loss_simreal))/2
        var_rank_loss = get_ranks(abs(var_loss_sim - var_loss_simreal))

        rank_loss = signal_rank_loss + mae_rank_loss + var_rank_loss

    elif infer_domain == 'simreal':
        strong_bias = np.genfromtxt(models_dir + 'bias_strong_simreal_val.txt')
        strong_loss = np.genfromtxt(models_dir + 'err_strong_simreal_val.txt')
        int_bias = np.genfromtxt(models_dir + 'bias_int_simreal_val.txt')
        int_loss = np.genfromtxt(models_dir + 'err_int_simreal_val.txt')
        weak_bias = np.genfromtxt(models_dir + 'bias_weak_simreal_val.txt')
        weak_loss = np.genfromtxt(models_dir + 'err_weak_simreal_val.txt')
        #cont_loss = np.genfromtxt(models_dir + 'err_cont_simreal_val.txt')
        #var_loss = np.genfromtxt(models_dir + 'var_simreal_val.txt')
        #rank_loss = signal_loss + mae_loss + var_loss
        #where_bias = np.where(abs(signal_bias) < 0.1)[0]
        #rank_loss = get_ranks(signal_loss[where_bias]) + get_ranks(var_loss[where_bias])
        rank_loss = get_ranks(strong_loss) + get_ranks(int_loss) + get_ranks(weak_loss) + get_ranks(abs(strong_bias)) + get_ranks(abs(int_bias)) + get_ranks(abs(weak_bias))

    else:
        bias_loss = np.genfromtxt(models_dir + 'bias_signal_val.txt')
        map_loss = np.genfromtxt(models_dir + 'map_mse_val.txt')

        
        bias_order = bias_loss.argsort()
        bias_ranks = bias_order.argsort()

        map_order = map_loss.argsort()
        map_ranks = map_order.argsort()

        #args_choose = np.where(abs(source_loss-target_loss)/np.mean([source_loss,target_loss],axis=1) < 0.1)[0]
        #arg_best = int(args_choose[int(np.argmin(map_loss[args_choose]))] + 1)

    
    arg_best = int(np.argmin(rank_loss) + 1)
    #arg_best = int(where_bias[np.argmin(rank_loss)] + 1)


    #arg_best = 109
    print('argbest = %i'%arg_best)
    #model_best = models_dir + f'model={model}_crossval={cross_val_id}_epoch={arg_best}.pth'
    model_best = models_dir + f'epoch={arg_best}.pth'


    # Transoform and load the data
    transform = TransformSZ(N_dim, eval=True)

    # Source data
    
    if train_or_val == 'train':
        IDs_sim = IDs_sim_train
        IDs_real = IDs_real_train
    elif train_or_val == 'val':
        IDs_sim = IDs_sim_val
        IDs_real = IDs_real_val
    
    if infer_domain=='sim':
        infer_loader = torch.utils.data.DataLoader(EvalDataset(IDs = IDs_sim,
                                                                x_dir = sim_x_dir,
                                                                y_dir = sim_y_dir,
                                                                transform = transform),
                                                                batch_size=1, drop_last=False,shuffle=False,pin_memory=True, collate_fn=custom_collate_fn)
    elif infer_domain=='simreal':
        infer_loader = torch.utils.data.DataLoader(EvalDataset(IDs = IDs_sim,
                                                                x_dir = simreal_x_dir,
                                                                y_dir = simreal_y_dir,
                                                                mask_dir = simreal_mask_dir,
                                                                transform = transform),
                                                                batch_size=1, drop_last=False,shuffle=False,pin_memory=True, collate_fn=custom_collate_fn)
        
    elif infer_domain=='real':
        infer_loader = torch.utils.data.DataLoader(EvalDataset(IDs = IDs_real,
                                                                x_dir = real_x_dir,
                                                                transform = transform),
                                                                batch_size=1, drop_last=False,shuffle=False,pin_memory=True, collate_fn=custom_collate_fn)
    

    if infer_domain == 'mix':
        # Sim
        infer_loader = torch.utils.data.DataLoader(EvalDataset(IDs = IDs_sim,
                                                                x_dir = sim_x_dir,
                                                                y_dir = sim_y_dir,
                                                                transform = transform),
                                                                batch_size=1, drop_last=False,shuffle=False,pin_memory=True, collate_fn=custom_collate_fn)
        for idx, (ID, x, y, _) in enumerate(infer_loader):
            if idx == N_pred:
                break
            eval_map(Net, (x, y, _), model_best, device, output_dir, infer_domain='sim', idx=ID)
            eval_features(Net, x, model_best, device, output_dir, infer_domain='sim', idx=ID) 

        # Simreal
        infer_loader = torch.utils.data.DataLoader(EvalDataset(IDs = IDs_sim,
                                                                x_dir = simreal_x_dir,
                                                                y_dir = simreal_y_dir,
                                                                mask_dir = simreal_mask_dir,
                                                                transform = transform),
                                                                batch_size=1, drop_last=False,shuffle=False,pin_memory=True, collate_fn=custom_collate_fn)
        for idx, (ID, x, y, mask) in enumerate(infer_loader):
            if idx == N_pred:
                break
            eval_map(Net, (x, y, mask), model_best, device, output_dir, infer_domain='simreal', idx=ID)
            eval_features(Net, x, model_best, device, output_dir, infer_domain='simreal', idx=ID) 

    elif infer_domain == 'simreal':
        # Simreal
        infer_loader = torch.utils.data.DataLoader(EvalDataset(IDs = IDs_sim,
                                                                x_dir = simreal_x_dir,
                                                                y_dir = simreal_y_dir,
                                                                mask_dir = simreal_mask_dir,
                                                                transform = transform),
                                                                batch_size=1, drop_last=False,shuffle=False,pin_memory=True, collate_fn=custom_collate_fn)
        for idx, (ID, x, y, mask) in enumerate(infer_loader):
            if idx == N_pred:
                break
            eval_map(Net, (x, y, mask), model_best, device, output_dir, infer_domain='simreal', idx=ID)
            eval_features(Net, x, model_best, device, output_dir, infer_domain='simreal', idx=ID)  

    
    

        


if __name__ == '__main__':
    main()

