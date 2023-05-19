from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_gu import *
from utils_misc import *
max_norm = 10

def train_model(args, model, trn_x, trn_y):

    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=args.task), batch_size=args.bs, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    args.local_iter_count = 0
<<<<<<< HEAD
    # print(n_trn/args.bs)
    args.total_local_iter = int(np.ceil(n_trn/args.bs)) * args.epoch
    # print(args.total_local_iter)
    for e in range(args.epoch):
        args.current_epoch = e
=======

    for e in range(args.epoch):
>>>>>>> c997a3d207ad4b9ac3f6a4e4d69c977badba0d0e
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/args.bs))):
            # print(int(np.ceil(n_trn/args.bs)))
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            StopGradScheduler(args, model)
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()
            args.local_iter_count += 1
           
    model.eval()            
    return model


class FedDecorrLoss(nn.Module):
    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8
    def _off_diagonal(self, mat):
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    def forward(self, x):
        N, C = x.shape
        if N == 1: return 0.0
        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))
        corr_mat = torch.matmul(x.t(), x)
        loss = (self._off_diagonal(corr_mat).pow(2)).mean() / N
        return loss

def train_feddecorr_model(args, model, trn_x, trn_y):

    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=args.task), batch_size=args.bs, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_fn2 = FedDecorrLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    args.local_iter_count = 0

    for e in range(args.epoch):
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/args.bs))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            StopGradScheduler(args, model)
            y_pred, z = model.forward_feat(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]

<<<<<<< HEAD
            loss += 0.01 * loss_fn2(z)
=======
            loss += 0.001 * loss_fn2(z)
>>>>>>> c997a3d207ad4b9ac3f6a4e4d69c977badba0d0e
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()
            args.local_iter_count += 1
        
    model.eval()            
    return model

def train_feddyn_mdl(args, model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=args.task), batch_size=args.bs, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=alpha_coef+args.weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    
    args.local_iter_count = 0
    for e in range(args.epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/args.bs))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            StopGradScheduler(args, model)
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor): local_par_list = param.reshape(-1)
                else: local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

            args.local_iter_count += 1

    model.eval()
            
    return model

###
def train_fedprox_mdl(args, model, avg_model_param_, mu, trn_x, trn_y):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=args.task), batch_size=args.bs, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    
    args.local_iter_count = 0
    
    for e in range(args.epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/args.bs))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            StopGradScheduler(args, model)
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
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

            args.local_iter_count += 1

    model.eval()
            
    return model

def train_scaffold_mdl(args, model, model_func, state_params_diff, trn_x, trn_y, learning_rate, batch_size, n_minibatch):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=args.task), batch_size=args.bs, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    
    n_iter_per_epoch = int(np.ceil(n_trn/args.bs))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)
    count_step = 0
    is_done = False
    args.local_iter_count = 0
    
    step_loss = 0; n_data_step = 0
    for e in range(args.epoch):
        # Training
        if is_done:
            break
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/args.bs))):
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            StopGradScheduler(args, model)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor): local_par_list = param.reshape(-1)
                else: local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
        
            loss_algo = torch.sum(local_par_list * state_params_diff)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0]; n_data_step += list(batch_y.size())[0]
   
    model.eval()
            
    return model
