from utils_libs import *
from utils_dataset import *
from utils_0301_models import *
 
# Global parameters
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10
# --- Evaluate a NN model

def get_acc_loss(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0; loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)

    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = np.argmax(y_pred.cpu().numpy(), axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            acc_overall += np.sum(y_pred == batch_y)
    
    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc_overall / n_tst

# --- Helper functions

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    mdl.load_state_dict(dict_param)    
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

# --- Train functions
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

import time
def NormGrad(args, model):
    with torch.no_grad():
        for p1 , p2 in model.named_parameters():
            if p2.grad == None: continue
            if args.use_mean == 1:
                if len(p2.grad.data.shape) == 2: p2.grad.data -= torch.mean(p2.grad.data, dim=(1), keepdim=True)
                elif len(p2.grad.data.shape) == 4: p2.grad.data -= torch.mean(p2.grad.data, dim=(1,2,3), keepdim=True)
            elif args.use_mean == 2:
                if len(p2.grad.data.shape) == 2: p2.grad.data -= torch.mean(p2.grad.data, dim=(0), keepdim=True)
                elif len(p2.grad.data.shape) == 4: p2.grad.data -= torch.mean(p2.grad.data, dim=(0,2,3), keepdim=True)
            elif args.use_mean == 5:
                if len(p2.grad.data.shape) == 2: 
                    p2.grad.data -= torch.mean(p2.grad.data, dim=(0), keepdim=True)
                    p2.grad.data -= torch.mean(p2.grad.data, dim=(1), keepdim=True)
                elif len(p2.grad.data.shape) == 4: 
                    p2.grad.data -= torch.mean(p2.grad.data, dim=(0,2,3), keepdim=True)
                    p2.grad.data -= torch.mean(p2.grad.data, dim=(1,2,3), keepdim=True)
            if args.gs == 4122:
                if "fc3" not in p1 and "fc2" not in p1: p2.grad.data *= args.gs_enc
                else: p2.grad.data *= args.gs_clf
            elif args.gs == 4123:
                if "fc3" not in p1 and "fc2" not in p1 and "fc1" not in p1: p2.grad.data *= args.gs_enc
                else: p2.grad.data *= args.gs_clf
            elif args.gs == 4125:
                if "fc3" not in p1 and "fc2" not in p1 and "fc1" not in p1 and "conv3" not in p1 and "conv2" not in p1: p2.grad.data *= args.gs_enc
                else: p2.grad.data *= args.gs_clf
            elif args.gs == 4126:
                if "fc3" not in p1 and "fc2" not in p1 and "fc1" not in p1 and "conv3" not in p1 and "conv2" not in p1 and "conv1" not in p1: p2.grad.data *= args.gs_enc
                else: p2.grad.data *= args.gs_clf
            elif args.gs in [4441, 4442, 4443, 4445, 4448, 44410]:
                args.gs_clf = 0.001
                args.gs_enc = 1
                if   args.gs == 4441: factor = 1
                elif args.gs == 4442: factor = 2
                elif args.gs == 4443: factor = 3
                elif args.gs == 4445: factor = 5
                elif args.gs == 4448: factor = 8
                elif args.gs == 44410: factor = 10
                if   "conv2" in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 0: p2.grad.data *= args.gs_clf
                elif "fc1"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 1: p2.grad.data *= args.gs_clf
                elif "fc2"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 2: p2.grad.data *= args.gs_clf
                elif "fc3"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 3: p2.grad.data *= args.gs_clf
                else: p2.grad.data *= args.gs_enc
            elif args.gs in [4451, 4452, 4453, 4455, 4458, 44510]:
                args.gs_clf = 0.001
                args.gs_enc = 1
                if   args.gs == 4451: factor = 1
                elif args.gs == 4452: factor = 2
                elif args.gs == 4453: factor = 3
                elif args.gs == 4455: factor = 5
                elif args.gs == 4458: factor = 8
                elif args.gs == 44510: factor = 10
                if   "conv1" in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 0: p2.grad.data *= args.gs_clf
                elif "conv2" in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 1: p2.grad.data *= args.gs_clf
                elif "fc1"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 2: p2.grad.data *= args.gs_clf
                elif "fc2"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 3: p2.grad.data *= args.gs_clf
                elif "fc3"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 4: p2.grad.data *= args.gs_clf
                else: p2.grad.data *= args.gs_enc
            elif args.gs in [4461, 4462, 4463, 4465, 4468, 44610]:
                args.gs_clf = 0.01
                args.gs_enc = 1
                if   args.gs == 4461: factor = 1
                elif args.gs == 4462: factor = 2
                elif args.gs == 4463: factor = 3
                elif args.gs == 4465: factor = 5
                elif args.gs == 4468: factor = 8
                elif args.gs == 44610: factor = 10
                if   "conv2" in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 0: p2.grad.data *= args.gs_clf
                elif "fc1"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 1: p2.grad.data *= args.gs_clf
                elif "fc2"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 2: p2.grad.data *= args.gs_clf
                elif "fc3"   in p1 and (args.local_epoch * 10 + args.local_iter) // factor <= 3: p2.grad.data *= args.gs_clf
                else: p2.grad.data *= args.gs_enc
            elif args.gs in [44712, 44713, 44715, 44722, 44723, 44725]:
                args.gs_clf = 0.001
                args.gs_enc = 1
                if   args.gs == 44712: factor, exp = 2, 0.5
                elif args.gs == 44713: factor, exp = 3, 0.5
                elif args.gs == 44715: factor, exp = 5, 0.5
                elif args.gs == 44722: factor, exp = 2, 1
                elif args.gs == 44723: factor, exp = 3, 1
                elif args.gs == 44725: factor, exp = 5, 1
                if   "conv2" in p1:
                    if (args.local_epoch * 10 + args.local_iter) // factor <= 0: p2.grad.data *= args.gs_clf
                    else: p2.grad.data *= 1 / 2 ** exp
                elif   "fc1" in p1:
                    if (args.local_epoch * 10 + args.local_iter) // factor <= 1: p2.grad.data *= args.gs_clf
                    else: p2.grad.data *= 1 / 3 ** exp
                elif   "fc2" in p1:
                    if (args.local_epoch * 10 + args.local_iter) // factor <= 2: p2.grad.data *= args.gs_clf
                    else: p2.grad.data *= 1 / 4 ** exp
                elif   "fc3" in p1:
                    if (args.local_epoch * 10 + args.local_iter) // factor <= 3: p2.grad.data *= args.gs_clf
                    else: p2.grad.data *= 1 / 5 ** exp      
            elif args.gs in [44812, 44813, 44815, 44822, 44823, 44825]:
                args.gs_clf = 0.001
                args.gs_enc = 1
                if   args.gs == 44812: factor, exp = 2, 0.5
                elif args.gs == 44813: factor, exp = 3, 0.5
                elif args.gs == 44815: factor, exp = 5, 0.5
                elif args.gs == 44822: factor, exp = 2, 1
                elif args.gs == 44823: factor, exp = 3, 1
                elif args.gs == 44825: factor, exp = 5, 1
                if   "conv2" in p1:
                    if (args.local_epoch * 10 + args.local_iter) // factor <= 0: p2.grad.data *= args.gs_clf
                    else: p2.grad.data *= 1 / 5 ** exp
                elif   "fc1" in p1:
                    if (args.local_epoch * 10 + args.local_iter) // factor <= 1: p2.grad.data *= args.gs_clf
                    else: p2.grad.data *= 1 / 4 ** exp
                elif   "fc2" in p1:
                    if (args.local_epoch * 10 + args.local_iter) // factor <= 2: p2.grad.data *= args.gs_clf
                    else: p2.grad.data *= 1 / 3 ** exp
                elif   "fc3" in p1:
                    if (args.local_epoch * 10 + args.local_iter) // factor <= 3: p2.grad.data *= args.gs_clf
                    else: p2.grad.data *= 1 / 2 ** exp 

def train_model(args, model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):

    # print("Hi1")
    if "CIFAR100" in args.task: 
        args.feat_dim = 100
    elif "CIFAR10" in args.task: args.feat_dim = 10
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    for e in range(epoch):

        args.local_epoch = e

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            args.local_iter = i
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]

            optimizer.zero_grad()
            # lr_scheduler(args, i)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            NormGrad(args, model)
            optimizer.step()
           
        if (e+1) % print_per == 0: model.train()
    
    for params in model.parameters():
        params.requires_grad = False
    model.eval()            
    return model

def train_scaffold_mdl(args, model, model_func, state_params_diff, trn_x, trn_y, learning_rate, batch_size, n_minibatch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    n_iter_per_epoch = int(np.ceil(n_trn/batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)
    count_step = 0
    is_done = False
    
    step_loss = 0; n_data_step = 0
    for e in range(epoch):

        args.local_epoch = e
        # Training
        if is_done:
            break
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            
            args.local_iter = i
            # lr_scheduler(args, i)
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = torch.sum(local_par_list * state_params_diff)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            NormGrad(args, model)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0]; n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay)/2 * np.sum(params * params)
                # print("Step %3d, Training Loss: %.4f" %(count_step, step_loss))
                step_loss = 0; n_data_step = 0
                model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_feddyn_mdl(args, model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    if args.gs in [711, 712, 713, 714] and args.epoch == 0:
        args.last_correct = 1
        args.running_correct = 0

    for e in range(epoch):
        # Training
        args.local_epoch = e
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            args.local_iter = i
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            # lr_scheduler(args, i)
            optimizer.zero_grad()
            loss.backward()
            NormGrad(args, model)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            # epoch_loss /= n_trn
            # if weight_decay != None:
            #     # Add L2 loss to complete f_i
            #     params = get_mdl_params([model], n_par)
            #     epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
            # print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

###
def train_fedprox_mdl(args, model, avg_model_param_, mu, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = len(avg_model_param_)
    
    for e in range(epoch):
        # Training
        args.local_epoch = e
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            # lr_scheduler(args, i)
            args.local_iter = i
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = mu/2 * torch.sum(local_par_list * local_par_list)
            loss_algo += -mu * torch.sum(local_par_list * avg_model_param_)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            NormGrad(args, model)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += weight_decay/2 * np.sum(params * params)
            
            # print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model