import torch
import os
from torch import nn
import numpy as np
import argparse
from funcs import train_simreal_eval_simreal, load_config, custom_collate_fn
from datasets import TrainDataset
from data_transforms import TransformSZ
import wandb
import json
import networks

#ca54b3d2066f30ac84f4859fb15a845d015e8c82
os.environ['WANDB_API_KEY'] = 'ca54b3d2066f30ac84f4859fb15a845d015e8c82'
os.environ['WANDB_DIR'] = '/scratch/jbregman_root/jbregman0/campratt'





def parse_args():
    parser = argparse.ArgumentParser(description='Train a network on SZ data')
    parser.add_argument('--fn_config', type=str, required=False, help='Path to the config file', default=None)
    parser.add_argument('--data_dir', type=str, required=False, help='Directory to data', default='/nfs/turbo/lsa-jbregman/campratt/DANN_SZ')
    parser.add_argument('--model', type=str, required=False, help='Network architecture', default='NestedUNet3D')
    parser.add_argument('--model_size', type=str, required=False, help='Network architecture size', default='small')
    parser.add_argument('--N_sim', type=int, required=False, help='Number of all-sky simulations maps', default=100)
    parser.add_argument('--N_dim', type=float, required=False, help='Height/Width dimension of images', default=128)
    parser.add_argument('--max_epochs', type=float, required=False, help='Maximum number of epochs to train', default=30)
    parser.add_argument('--batch_size', type=float, required=False, help='Batch size', default=128)
    parser.add_argument('--lr', type=float, required=False, help='Learning rate', default=1e-3)
    parser.add_argument('--wd', type=float, required=False, help='Weight decay', default=0.0)
    parser.add_argument('--betas', type=float, required=False, help='Momentum parameters', default=(0.99,0.999))
    parser.add_argument('--clip_norm', type=float, required=False, help='Gradient clipping value', default=None)
    parser.add_argument('--alpha', type=float, required=False, help='Focal loss alpha', default=2)
    parser.add_argument('--y_sig_start', type=float, required=False, help='Starting lower bound for signals included in curriculum loss', default=60)
    parser.add_argument('--y_strong_low', type=float, required=False, help='Lower bound for strong signals', default=60)
    parser.add_argument('--y_weak_low', type=float, required=False, help='Lower bound for weak signals', default=10)
    parser.add_argument('--y_weak_high', type=float, required=False, help='Upper bound for weak signals', default=30)
    parser.add_argument('--y_int_low', type=float, required=False, help='Lower bound for intermediate signals', default=30)
    parser.add_argument('--y_int_high', type=float, required=False, help='Upper bound for intermediate signals', default=60)
    parser.add_argument('--N_per_epoch', type=float, required=False, help='Number of samples used for training one epoch', default=None)
    parser.add_argument('--cross_val_id', type=int, required=False, help='Number ID defining how to split the data for cross-validation', default=None)
    parser.add_argument('--models_dir', type=str, required=False, help='Directory to save models', default='/scratch/jbregman_root/jbregman0/campratt/SZU/model_example/')
    

    args = parser.parse_args()
    return args


    

def main():
    args = parse_args()

    if args.fn_config is not None:
        config = load_config(args.fn_config)

    else:
        config = {}

    # Update config with command line arguments if they don't already exist in the config file or if a config file is not provided
    for key, value in vars(args).items():
        if key not in config.keys():
            config[key] = value


    
    ########################################################
    # Main configuration parameters
    model = config['model']
    model_size = config['model_size']
    N_sim = config['N_sim']
    N_dim = config['N_dim']
    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    lr = config['lr']
    wandb_on = True
    N_per_epoch = config['N_per_epoch']
    cross_val_id = config['cross_val_id']
    wd = config['wd']
    betas = config['betas']
    clip_norm = config['clip_norm']
    models_dir = config['models_dir']
    alpha = config['alpha']
    y_sig_start = config['y_sig_start']
    y_strong_low = config['y_strong_low']
    y_weak_low = config['y_weak_low']
    y_weak_high = config['y_weak_high']
    y_int_low = config['y_int_low']
    y_int_high = config['y_int_high']
    data_dir = config['data_dir']
    


    

    ########################################################
    # Define the setup of the cross-validation experiments
    N_IDs_sim_max = 700*N_sim
    IDs_sim_all = np.arange(0, int(round(N_IDs_sim_max,1)))

    # Split into 67/33 train/val
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

    inds_train_sim, inds_val_sim = get_indices(IDs_sim_all, n_parts)

    

    if cross_val_id is None:
        cross_val_id = int(0)
    else:
        cross_val_id = int(cross_val_id)
        
    IDs_sim_train, IDs_sim_val = inds_train_sim[cross_val_id], inds_val_sim[cross_val_id]

    
    
    ########################################################
    # Initialize the network, optimizer, and model directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Net_train = getattr(networks, model)(model_size=model_size).to(device)


    if torch.cuda.device_count() > 1:
        Net_train = nn.DataParallel(Net_train)

    optimizer = torch.optim.Adam(Net_train.parameters(),lr=lr, weight_decay=wd, betas=betas)

    os.makedirs(models_dir, exist_ok=True)
    

    ########################################################
    # Set up Weights and Biases
    # First Training
    print('First training: ',model)
    if wandb_on:
        train_run = wandb.init(project='SZU', entity='campratt')
        train_run.config.update(config)
        train_run.watch(Net_train, log="all")
    
    ########################################################
    # Set up the data loaders
    simreal_x_dir = os.path.join(data_dir,'x_simreal/')
    simreal_y_dir = os.path.join(data_dir,'y_simreal/')
    simreal_mask_dir = os.path.join(data_dir,'mask_simreal/')

    transform_train = TransformSZ(N_dim)

    train_loader_simreal = torch.utils.data.DataLoader(TrainDataset(IDs = IDs_sim_train, 
                                                                x_dir = simreal_x_dir,
                                                                y_dir = simreal_y_dir,
                                                                mask_dir = simreal_mask_dir,
                                                                N_per_epoch=N_per_epoch,
                                                                transform = transform_train
                                                                ),
                                                                batch_size=batch_size, 
                                                                drop_last=True,
                                                                shuffle=True, 
                                                                collate_fn=custom_collate_fn)
    

    transform_val = TransformSZ(N_dim,eval=True)
    val_loader_simreal = torch.utils.data.DataLoader(TrainDataset(IDs = IDs_sim_val, 
                                                                    x_dir = simreal_x_dir,
                                                                    y_dir = simreal_y_dir,
                                                                    mask_dir = simreal_mask_dir,
                                                                    N_per_epoch=None,
                                                                    transform = transform_val,
                                                                    ),
                                                                    batch_size=batch_size, 
                                                                    drop_last=False,
                                                                    shuffle=True, 
                                                                    collate_fn=custom_collate_fn)
        
   

    
    ########################################################
    # Train the network and evaluate on the validation set
    losses = []
    errs_signal_simreal = []
    biases_signal_simreal = []
    errs_frac_signal_simreal = []
    biases_frac_signal_simreal = []
    errs_strong_simreal = []
    biases_strong_simreal = []
    errs_frac_strong_simreal = []
    biases_frac_strong_simreal = []
    errs_weak_simreal = []
    biases_weak_simreal = []
    errs_frac_weak_simreal = []
    biases_frac_weak_simreal = []
    errs_int_simreal = []
    biases_int_simreal = []
    errs_frac_int_simreal = []
    biases_frac_int_simreal = []
    vars_simreal = []


    for epoch in range(1, max_epochs+1):
        print('')
        print("Epoch: "+str(epoch))

        
            
        
        print('train=simreal val=simreal')
        #loss,err_signal_simreal, bias_signal_simreal, err_frac_signal_simreal, bias_frac_signal_simreal,\
        #    err_weak_simreal, bias_weak_simreal, err_frac_weak_simreal, bias_frac_weak_simreal,\
        #    err_int_simreal, bias_int_simreal, err_frac_int_simreal, bias_frac_int_simreal,\
        #            err_strong_simreal, bias_strong_simreal, err_frac_strong_simreal, bias_frac_strong_simreal, \
        #            var_simreal = \
        #        train_simreal_eval_simreal(Net_train, train_loader_simreal, val_loader_simreal, \
        #                            optimizer, epoch, device, alpha, y_sig_start, y_weak_low, y_weak_high, y_int_low, y_int_high, y_strong_low, alpha, clip_norm, WB=train_run)
        
        results = train_simreal_eval_simreal(Net_train, 
                                              train_loader_simreal, 
                                              val_loader_simreal, 
                                              optimizer, 
                                              epoch, 
                                              device, 
                                              alpha, 
                                              y_sig_start, 
                                              y_weak_low, 
                                              y_weak_high, 
                                              y_int_low, 
                                              y_int_high, 
                                              y_strong_low, 
                                              clip_norm, 
                                              WB=train_run)
        
        loss,err_signal_simreal, bias_signal_simreal, err_frac_signal_simreal, bias_frac_signal_simreal,\
        err_weak_simreal, bias_weak_simreal, err_frac_weak_simreal, bias_frac_weak_simreal,\
        err_int_simreal, bias_int_simreal, err_frac_int_simreal, bias_frac_int_simreal,\
        err_strong_simreal, bias_strong_simreal, err_frac_strong_simreal, bias_frac_strong_simreal, \
        var_simreal = results


        losses.append(loss)
        errs_signal_simreal.append(err_signal_simreal)
        biases_signal_simreal.append(bias_signal_simreal)
        errs_frac_signal_simreal.append(err_frac_signal_simreal)
        biases_frac_signal_simreal.append(bias_frac_signal_simreal)
        errs_strong_simreal.append(err_strong_simreal)
        biases_strong_simreal.append(bias_strong_simreal)
        errs_frac_strong_simreal.append(err_frac_strong_simreal)
        biases_frac_strong_simreal.append(bias_frac_strong_simreal)
        errs_weak_simreal.append(err_weak_simreal)
        biases_weak_simreal.append(bias_weak_simreal)
        errs_frac_weak_simreal.append(err_frac_weak_simreal)
        biases_frac_weak_simreal.append(bias_frac_weak_simreal)
        errs_int_simreal.append(err_int_simreal)
        biases_int_simreal.append(bias_int_simreal)
        errs_frac_int_simreal.append(err_frac_int_simreal)
        biases_frac_int_simreal.append(bias_frac_int_simreal)
        vars_simreal.append(var_simreal)


        # Update validation metrics every epoch
        np.savetxt(f"{models_dir}loss_val.txt", losses)
        np.savetxt(f"{models_dir}err_signal_simreal_val.txt", errs_signal_simreal)
        np.savetxt(f"{models_dir}bias_signal_simreal_val.txt", biases_signal_simreal)
        np.savetxt(f"{models_dir}err_frac_signal_simreal_val.txt", errs_frac_signal_simreal)
        np.savetxt(f"{models_dir}bias_frac_signal_simreal_val.txt", biases_frac_signal_simreal)
        np.savetxt(f"{models_dir}err_strong_simreal_val.txt", errs_strong_simreal)
        np.savetxt(f"{models_dir}bias_strong_simreal_val.txt", biases_strong_simreal)
        np.savetxt(f"{models_dir}err_frac_strong_simreal_val.txt", errs_frac_strong_simreal)
        np.savetxt(f"{models_dir}bias_frac_strong_simreal_val.txt", biases_frac_strong_simreal)
        np.savetxt(f"{models_dir}err_weak_simreal_val.txt", errs_weak_simreal)
        np.savetxt(f"{models_dir}bias_weak_simreal_val.txt", biases_weak_simreal)
        np.savetxt(f"{models_dir}err_frac_weak_simreal_val.txt", errs_frac_weak_simreal)
        np.savetxt(f"{models_dir}bias_frac_weak_simreal_val.txt", biases_frac_weak_simreal)
        np.savetxt(f"{models_dir}err_int_simreal_val.txt", errs_int_simreal)
        np.savetxt(f"{models_dir}bias_int_simreal_val.txt", biases_int_simreal)
        np.savetxt(f"{models_dir}err_frac_int_simreal_val.txt", errs_frac_int_simreal)
        np.savetxt(f"{models_dir}bias_frac_int_simreal_val.txt", biases_frac_int_simreal)
        np.savetxt(f"{models_dir}var_simreal_val.txt", vars_simreal)
            
        
        
        # Save the model every epoch
        torch.save(Net_train.state_dict(), f"{models_dir}epoch={epoch}.pth")

        


if __name__ == '__main__':
    main()
