import torch
from torch import nn

class Signal_Loss(nn.Module):

    def __init__(self, y_low=10,y_high=None):
        super(Signal_Loss, self).__init__()
        self.y_low = y_low
        self.y_high = y_high

    def forward(self, y_pred, y_true):
        
        
        y_pred = torch.flatten(y_pred,start_dim=0)
        y_true = torch.flatten(y_true,start_dim=0)

        
        
        if self.y_high is None:
            where_signal = (y_true > self.y_low)
        else:
            where_signal = (y_true > self.y_low) & (y_true<self.y_high)

        y_pred = y_pred[where_signal]
        y_true = y_true[where_signal]


        errs_init = y_pred - y_true

        
        biases_signal = errs_init.mean()
        errs_signal = torch.abs(errs_init).mean()

        biases_frac_signal = (errs_init/y_true).mean()
        errs_frac_signal = torch.abs(errs_init/y_true).mean()

        loss = errs_signal
        

        return loss, errs_signal, biases_signal, errs_frac_signal, biases_frac_signal
    
    
