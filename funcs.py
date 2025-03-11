import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from losses import Signal_Loss


# Train and evaluate the network for a single epoch

def train_simreal_eval_simreal(Net,
                               DataLoader_Train_SimReal, 
                               DataLoader_Val_SimReal,
                               optimizer, 
                               epoch, 
                               device, 
                               alpha=2,
                               y_sig_start=60,
                               y_weak_low=10,
                               y_weak_high=30,
                               y_int_low=30,
                               y_int_high=60, 
                               y_strong_low=60, 
                               clip_norm=None, 
                               WB=None):
    
    # Training
    Net.train()


    for batch_idx, (x_simreal, y_simreal, mask_simreal) in enumerate(DataLoader_Train_SimReal):
        x_simreal = x_simreal.to(device).float()
        y_simreal = y_simreal.to(device).float()
        mask_simreal = mask_simreal.to(device).float()

        optimizer.zero_grad()

        


        # SimReal
        mask = None
        if mask_simreal is not None:
            mask = mask_simreal.squeeze(1)
            mask = torch.tensor(mask, dtype=torch.bool)

        y_simreal = y_simreal.squeeze(1)
        y_simreal_medians = torch.median(y_simreal.view(y_simreal.size()[0], -1), dim=1).values
        y_simreal_medians = y_simreal_medians.view(-1, 1, 1)
        y_simreal = y_simreal - y_simreal_medians

        y_pred_simreal = Net(x_simreal)
        y_pred_simreal_medians = torch.median(y_pred_simreal.view(y_pred_simreal.size()[0], -1), dim=1).values
        y_pred_simreal_medians = y_pred_simreal_medians.view(-1, 1, 1)
        y_pred_simreal = y_pred_simreal - y_pred_simreal_medians
        


        if mask is not None:
            y_pred_simreal = torch.flatten(y_pred_simreal)
            y_simreal = torch.flatten(y_simreal)
            mask = torch.flatten(mask)
            y_simreal = y_simreal[mask]
            y_pred_simreal = y_pred_simreal[mask]

        
        # Strong Signal
        criterion_strong_simreal = Signal_Loss(y_low=y_strong_low)
        loss_strong_simreal, err_strong_simreal, bias_strong_simreal, err_frac_strong_simreal, bias_frac_strong_simreal = criterion_strong_simreal(y_pred_simreal, y_simreal)

        # Intermediate Signal
        criterion_int_simreal = Signal_Loss(y_low=y_int_low,y_high=y_int_high)
        loss_int_simreal, err_int_simreal, bias_int_simreal, err_frac_int_simreal, bias_frac_int_simreal = criterion_int_simreal(y_pred_simreal, y_simreal)

    
        # Weak Signal
        criterion_weak_simreal = Signal_Loss(y_low=y_weak_low,y_high=y_weak_high)
        loss_weak_simreal, err_weak_simreal, bias_weak_simreal, err_frac_weak_simreal, bias_frac_weak_simreal = criterion_weak_simreal(y_pred_simreal, y_simreal)

        
        # Signal
        y_sig_new = y_sig_start - (alpha*(epoch-1))
        if y_sig_new<10:
            y_sig_new = 10
        criterion_signal_simreal = Signal_Loss(y_low=y_sig_new)
        loss_signal_simreal, err_signal_simreal, bias_signal_simreal, err_frac_signal_simreal, bias_frac_signal_simreal = criterion_signal_simreal(y_pred_simreal, y_simreal)
        

        loss = loss_signal_simreal
        loss.backward()
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(Net.parameters(), clip_norm)
        optimizer.step()


        
        if WB is not None:
            WB.log({'epoch': epoch, 
                    'batch': batch_idx, 
                    'loss': loss.item(),
                    'err_signal_simreal': err_signal_simreal.item(), 'bias_signal_simreal': bias_signal_simreal.item(),
                    'err_frac_signal_simreal': err_frac_signal_simreal.item(), 'bias_frac_signal_simreal': bias_frac_signal_simreal.item(),
                    'err_weak_simreal': err_weak_simreal.item(), 'bias_weak_simreal': bias_weak_simreal.item(),
                    'err_frac_weak_simreal': err_frac_weak_simreal.item(), 'bias_frac_weak_simreal': bias_frac_weak_simreal.item(),
                    'err_int_simreal': err_int_simreal.item(), 'bias_int_simreal': bias_int_simreal.item(),
                    'err_frac_int_simreal': err_frac_int_simreal.item(), 'bias_frac_int_simreal': bias_frac_int_simreal.item(),
                    'err_strong_simreal': err_strong_simreal.item(), 'bias_strong_simreal': bias_strong_simreal.item(),
                    'err_frac_strong_simreal': err_frac_strong_simreal.item(), 'bias_frac_strong_simreal': bias_frac_strong_simreal.item(),
                    'var_simreal': torch.mean(torch.abs((y_pred_simreal.flatten() - torch.mean(y_pred_simreal.flatten())))).item()
                    })



    # Evaluation
    with torch.no_grad():
        torch.cuda.empty_cache()
        Net.eval()
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



        for batch_idx, ((x_simreal, y_simreal, mask_simreal)) in enumerate(DataLoader_Val_SimReal):
            x_simreal = x_simreal.to(device).float()
            y_simreal = y_simreal.to(device).float()
            mask_simreal = mask_simreal.to(device).float()



            if mask_simreal is not None:
                mask = mask_simreal.squeeze(1)
                mask = torch.tensor(mask, dtype=torch.bool)

            y_simreal = y_simreal.squeeze(1)
            y_simreal_medians = torch.median(y_simreal.view(y_simreal.size()[0], -1), dim=1).values
            y_simreal_medians = y_simreal_medians.view(-1, 1, 1)
            y_simreal = y_simreal - y_simreal_medians

            y_pred_simreal = Net(x_simreal)
            y_pred_simreal_medians = torch.median(y_pred_simreal.view(y_pred_simreal.size()[0], -1), dim=1).values
            y_pred_simreal_medians = y_pred_simreal_medians.view(-1, 1, 1)
            y_pred_simreal = y_pred_simreal - y_pred_simreal_medians
            


            if mask is not None:
                y_pred_simreal = torch.flatten(y_pred_simreal)
                y_simreal = torch.flatten(y_simreal)
                mask = torch.flatten(mask)
                y_simreal = y_simreal[mask]
                y_pred_simreal = y_pred_simreal[mask]

            # Strong Signal
            criterion_strong_simreal = Signal_Loss(y_low=y_strong_low)
            loss_strong_simreal, err_strong_simreal, bias_strong_simreal, err_frac_strong_simreal, bias_frac_strong_simreal = criterion_strong_simreal(y_pred_simreal, y_simreal)
        
            # Weak Signal
            criterion_weak_simreal = Signal_Loss(y_low=y_weak_low,y_high=y_weak_high)
            loss_weak_simreal, err_weak_simreal, bias_weak_simreal, err_frac_weak_simreal, bias_frac_weak_simreal = criterion_weak_simreal(y_pred_simreal, y_simreal)

            # Intermediate Signals
            criterion_int_simreal = Signal_Loss(y_low=y_int_low,y_high=y_int_high)
            loss_int_simreal, err_int_simreal, bias_int_simreal, err_frac_int_simreal, bias_frac_int_simreal = criterion_int_simreal(y_pred_simreal, y_simreal)

            # Signal
            y_sig_new = y_sig_start - (alpha*(epoch-1))
            if y_sig_new<10:
                y_sig_new = 10
            criterion_signal_simreal = Signal_Loss(y_low=y_sig_new)
            loss_signal_simreal, err_signal_simreal, bias_signal_simreal, err_frac_signal_simreal, bias_frac_signal_simreal = criterion_signal_simreal(y_pred_simreal, y_simreal)
            
            loss = loss_signal_simreal 
            
            losses.append(loss.cpu().numpy())
            errs_signal_simreal.append(err_signal_simreal.cpu().numpy())
            biases_signal_simreal.append(bias_signal_simreal.cpu().numpy())
            errs_frac_signal_simreal.append(err_frac_signal_simreal.cpu().numpy())
            biases_frac_signal_simreal.append(bias_frac_signal_simreal.cpu().numpy())
            errs_strong_simreal.append(err_strong_simreal.cpu().numpy())
            biases_strong_simreal.append(bias_strong_simreal.cpu().numpy())
            errs_frac_strong_simreal.append(err_frac_strong_simreal.cpu().numpy())
            biases_frac_strong_simreal.append(bias_frac_strong_simreal.cpu().numpy())
            errs_weak_simreal.append(err_weak_simreal.cpu().numpy())
            biases_weak_simreal.append(bias_weak_simreal.cpu().numpy())
            errs_frac_weak_simreal.append(err_frac_weak_simreal.cpu().numpy())
            biases_frac_weak_simreal.append(bias_frac_weak_simreal.cpu().numpy())
            errs_int_simreal.append(err_int_simreal.cpu().numpy())
            biases_int_simreal.append(bias_int_simreal.cpu().numpy())
            errs_frac_int_simreal.append(err_frac_int_simreal.cpu().numpy())
            biases_frac_int_simreal.append(bias_frac_int_simreal.cpu().numpy())
            vars_simreal.append(torch.mean(torch.abs((y_pred_simreal.flatten() - torch.mean(y_pred_simreal.flatten())))).cpu().numpy())

               


        

        print('Memory Allocated: ',torch.cuda.memory_allocated()/1e9)

        
        
        
        if WB is not None:
            WB.log({'epoch': epoch, 
                    'loss_val': np.mean(losses),
                    'err_signal_simreal_val': np.mean(errs_signal_simreal), 'bias_signal_simreal_val': np.mean(biases_signal_simreal),
                    'err_frac_signal_simreal_val': np.mean(errs_frac_signal_simreal), 'bias_frac_signal_simreal_val': np.mean(biases_frac_signal_simreal),
                    'err_weak_simreal_val': np.mean(errs_weak_simreal), 'bias_weak_simreal_val': np.mean(biases_weak_simreal),
                    'err_frac_weak_simreal_val': np.mean(errs_frac_weak_simreal), 'bias_frac_weak_simreal_val': np.mean(biases_frac_weak_simreal),
                    'err_int_simreal_val': np.mean(errs_int_simreal), 'bias_int_simreal_val': np.mean(biases_int_simreal),
                    'err_frac_int_simreal_val': np.mean(errs_frac_int_simreal), 'bias_frac_int_simreal_val': np.mean(biases_frac_int_simreal),
                    'err_strong_simreal_val': np.mean(errs_strong_simreal), 'bias_strong_simreal_val': np.mean(biases_strong_simreal),
                    'err_frac_strong_simreal_val': np.mean(errs_frac_strong_simreal), 'bias_frac_strong_simreal_val': np.mean(biases_frac_strong_simreal),
                    'var_simreal_val': np.mean(vars_simreal),
                    })
  

    return [np.mean(losses), 
    np.mean(errs_signal_simreal), 
    np.mean(biases_signal_simreal), 
    np.mean(errs_frac_signal_simreal), 
    np.mean(biases_frac_signal_simreal), 
    np.mean(errs_weak_simreal), 
    np.mean(biases_weak_simreal), 
    np.mean(errs_frac_weak_simreal), 
    np.mean(biases_frac_weak_simreal),
    np.mean(errs_int_simreal), 
    np.mean(biases_int_simreal), 
    np.mean(errs_frac_int_simreal), 
    np.mean(biases_frac_int_simreal),
    np.mean(errs_strong_simreal), 
    np.mean(biases_strong_simreal), 
    np.mean(errs_frac_strong_simreal), 
    np.mean(biases_frac_strong_simreal),
    np.mean(vars_simreal)]

    


