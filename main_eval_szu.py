import torch
import numpy as np
from utils.data_transforms import TransformSZ
from utils.datasets import EvalDataset
import networks
import os
import argparse
from utils.funcs import collate_fn



def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Hyperparameter Tuning")
    parser.add_argument('--x_data_dir', type=str, required=False, help='Directory to input data', default='/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/x_simreal')
    parser.add_argument('--y_data_dir', type=str, required=False, help='Directory to input data', default='/nfs/turbo/lsa-jbregman/campratt/DANN_SZ/y_simreal')
    parser.add_argument('--model', type=str, required=False, help='Network architecture', default='NestedUNet3D')
    parser.add_argument('--model_size', type=str, required=False, help='Network architecture size', default='small')
    parser.add_argument('--model_dir', type=str, required=False, help='Directory to saved models', default='/scratch/jbregman_root/jbregman0/campratt/SZU/model_example/')
    parser.add_argument('--model_path', type=int, required=False, help='Path to specific model instead of using model_dir', default=None)
    parser.add_argument('--IDs', type=int, required=False, help='IDs of samples', nargs='+', default=[0,1])
    parser.add_argument('--N_dim', type=float, required=False, help='Height/Width dimension of images', default=128)
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save outputs', default='/nfs/turbo/lsa-jbregman/campratt/SZU/outputs_evaluate/')
    
    

    args = parser.parse_args()
    return args



def eval_map(Net, Data, load_model, device, output_dir, ID=0):
    ID = ID.cpu().numpy()[0]
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

    
    x, y, mask = Data
    x = x.to(device).float()
    if y is not None:
        y = y.to(device).float()
    if mask is not None:
        mask = mask.to(device).float
    
    with torch.no_grad():
        pred = Net(x)
        pred = pred.detach().cpu().numpy()
        np.save(os.path.join(output_dir, f'y_{ID}_pred.npy'), pred)
        if y is not None:
            y_true = y.detach().cpu().numpy()
            np.save(os.path.join(output_dir, f'y_{ID}_true.npy'), y_true)
        if mask is not None:
            mask = mask.detach().cpu().numpy()
            np.save(os.path.join(output_dir, f'mask_{ID}.npy'), mask)
        
        np.save(os.path.join(output_dir, f'x_{ID}.npy'), x[0].detach().cpu().numpy())
        





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

    config = {}
    for key, value in vars(args).items():
        #if key not in config.keys():
        config[key] = value

    print(config)

    # Miscellanous
    
    model = config['model']
    model_size = config['model_size']
    N_dim = config['N_dim']

    output_dir = config['output_dir']
    model_dir = config['model_dir']
    IDs = config['IDs']
    x_data_dir = config['x_data_dir']
    y_data_dir = config['y_data_dir']
    model_path = config['model_path']

    print('output_dir: %s'%output_dir)
    os.makedirs(output_dir, exist_ok=True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Net = getattr(networks, model)(model_size=model_size).to(device)

    
    # Find the best model
    if model_path is None and model_dir is not None:
        strong_bias = np.genfromtxt(os.path.join(model_dir, 'bias_strong_simreal_val.txt'))
        strong_loss = np.genfromtxt(os.path.join(model_dir, 'err_strong_simreal_val.txt'))
        int_bias = np.genfromtxt(os.path.join(model_dir, 'bias_int_simreal_val.txt'))
        int_loss = np.genfromtxt(os.path.join(model_dir, 'err_int_simreal_val.txt'))
        weak_bias = np.genfromtxt(os.path.join(model_dir, 'bias_weak_simreal_val.txt'))
        weak_loss = np.genfromtxt(os.path.join(model_dir, 'err_weak_simreal_val.txt'))

        rank_loss = get_ranks(strong_loss) + get_ranks(int_loss) + get_ranks(weak_loss) + get_ranks(abs(strong_bias)) + get_ranks(abs(int_bias)) + get_ranks(abs(weak_bias))

        epoch_best = int(np.argmin(rank_loss) + 1)
        print('epoch best = %i'%epoch_best)
        model_path = os.path.join(model_dir,f'epoch={epoch_best}.pth')
    else:
        model_path = model_path



    # Transoform and load the data
    transform = TransformSZ(N_dim, eval=True)

    infer_loader = torch.utils.data.DataLoader(EvalDataset(IDs = IDs,
                                                                x_dir = x_data_dir,
                                                                y_dir = y_data_dir,
                                                                transform = transform),
                                                                batch_size=1, 
                                                                drop_last=False,
                                                                shuffle=False,
                                                                pin_memory=True, 
                                                                collate_fn=collate_fn)
    

    

    
    
    for _, (ID, x, y, mask) in enumerate(infer_loader):
        eval_map(Net, (x, y, mask), model_path, device, output_dir, ID=ID)
        


if __name__ == '__main__':
    main()

