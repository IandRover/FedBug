import torch, copy
import numpy as np
from torch.utils import data
from utils_dataset import *

def get_acc_loss(args, data_x, data_y, model, w_decay = None):
    acc_overall = 0; loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=args.task), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(args.device)

    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            y_pred = model(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            y_pred = np.argmax(y_pred.cpu().numpy(), axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            acc_overall += np.sum(y_pred == batch_y)
        
    model.train()
    return loss_overall, acc_overall / n_tst

# --- Helper functions

def set_client_from_params(args, mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(args.device))
        idx += length
    mdl.load_state_dict(dict_param)
    del dict_param
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)