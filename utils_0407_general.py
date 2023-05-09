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
        # print(name, length)
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
import time

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def __SetGrad(tensor, sign):
    tensor.weight.requires_grad = sign
    tensor.bias.requires_grad = sign

def GlobalGradScheduler(args, new_model, avg_model, new_avg_model_param):
    # Vanilla
    if int(args.GGU) == 0: return set_client_from_params(new_model, new_avg_model_param)

    elif int(args.GGU) == 901: return set_client_from_params(new_model, new_avg_model_param)
    
    
    # with args.seed * 10000
    elif int(args.GGU) == 902: return set_client_from_params(new_model, new_avg_model_param)
    
    elif int(args.GGU) == 111:
        factor = (args.GGU*100)%100/100
        A = avg_model_param = get_mdl_params([avg_model], args.n_par)[0]
        B = new_avg_model_param
        dict_param = copy.deepcopy(dict(new_model.named_parameters()))
        idx = 0
        for name, param in new_model.named_parameters():
            # print(name)
            if   "conv1" in name: coefficient = factor ** 0
            elif "conv2" in name: coefficient = factor ** 1
            elif "fc1" in name: coefficient = factor ** 2
            elif "fc2" in name: coefficient = factor ** 3
            elif "fc3" in name: coefficient = factor ** 4
            weights = param.data
            length = len(weights.reshape(-1))
            new_params = (B[idx:idx+length]-A[idx:idx+length]) * coefficient + A[idx:idx+length]
            # print(args.GGU, factor, coefficient)
            new_params = torch.tensor(new_params.reshape(weights.shape)).to(device)
            dict_param[name].data.copy_(new_params)
            idx += length    
        new_model.load_state_dict(dict_param)    
        return new_model

    elif int(args.GGU) == 121:
        factor = (args.GGU*100)%100/100 + 1
        A = avg_model_param = get_mdl_params([avg_model], args.n_par)[0]
        B = new_avg_model_param
        dict_param = copy.deepcopy(dict(new_model.named_parameters()))
        idx = 0
        for name, param in new_model.named_parameters():
            # print(name)
            if   "conv1" in name: coefficient = factor ** 4
            elif "conv2" in name: coefficient = factor ** 3
            elif "fc1" in name: coefficient = factor ** 2
            elif "fc2" in name: coefficient = factor ** 1
            elif "fc3" in name: coefficient = factor ** 0
            weights = param.data
            length = len(weights.reshape(-1))
            new_params = (B[idx:idx+length]-A[idx:idx+length]) * coefficient + A[idx:idx+length]
            # print(args.GGU, factor, coefficient)
            new_params = torch.tensor(new_params.reshape(weights.shape)).to(device)
            dict_param[name].data.copy_(new_params)
            idx += length
        new_model.load_state_dict(dict_param)
        return new_model

    elif int(args.GGU) == 122:
        B = new_avg_model_param
        dict_param = copy.deepcopy(dict(new_model.named_parameters()))
        idx = 0
        for name, param in new_model.named_parameters():
            weights = param.data
            length = len(weights.reshape(-1))
            new_params = B[idx:idx+length]
            new_params = torch.tensor(new_params.reshape(weights.shape)).to(device)
            dict_param[name].data.copy_(new_params)
            idx += length
        new_model.load_state_dict(dict_param)    
        return new_model

    elif int(args.GGU) == 131:
        factor = (args.GGU*100)%100/100
        A = avg_model_param = get_mdl_params([avg_model], args.n_par)[0]
        B = new_avg_model_param
        dict_param = copy.deepcopy(dict(new_model.named_parameters()))
        idx = 0
        for name, param in new_model.named_parameters():
            if   "conv1" in name: coefficient = factor ** 4
            elif "conv2" in name: coefficient = factor ** 3
            elif "fc1" in name: coefficient = factor ** 2
            elif "fc2" in name: coefficient = factor ** 1
            elif "fc3" in name: coefficient = factor ** 0
            weights = param.data
            length = len(weights.reshape(-1))
            new_params = (B[idx:idx+length]-A[idx:idx+length]) * coefficient + A[idx:idx+length]
            # print(args.GGU, factor, coefficient)
            new_params = torch.tensor(new_params.reshape(weights.shape)).to(device)
            dict_param[name].data.copy_(new_params)
            idx += length
        new_model.load_state_dict(dict_param)    
        return new_model

def StopGradScheduler(args, model):
    with torch.no_grad():
        for p1 , p2 in model.named_parameters():
            p2.requires_grad = True
        # Bottom up
        if int(args.GU) == 111:
            if args.task in ["emnist26"]:
                factor = (args.GU*100)%100
                if  args.local_iter_count // factor <= 0: __SetGrad(model.fc2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(model.fc3, False)
            else:
                factor = (args.GU*100)%100
                if  args.local_iter_count // factor <= 0: __SetGrad(model.conv2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(model.fc1, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(model.fc2, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(model.fc3, False)

                if    args.lradp == 1 and args.local_iter_count // factor <= 3: 
                    args.optimizer.param_groups[0]["lr"] = args.learning_rate * 2
                elif  args.lradp == 2: 
                    X = factor * 4 / 50
                    args.optimizer.param_groups[0]["lr"] = args.learning_rate / (1 - 0.5 * X)
                elif  args.lradp == 3 and args.local_iter_count // factor > 3: 
                    X = factor * 4 / 50
                    args.optimizer.param_groups[0]["lr"] = args.learning_rate * (2 - X) / 2 / (1-X) 
                else: args.optimizer.param_groups[0]["lr"] = args.learning_rate
                # print(args.local_iter_count, args.local_iter_count // factor, args.optimizer.param_groups[0]["lr"])
                # time.sleep(0.2)

        # Top down
        elif int(args.GU) == 112:
            factor = int(args.GU*100)%100
            if  args.local_iter_count // factor <= 3: __SetGrad(model.conv1, False)
            if  args.local_iter_count // factor <= 2: __SetGrad(model.conv2, False)
            if  args.local_iter_count // factor <= 1: __SetGrad(model.fc1, False)
            if  args.local_iter_count // factor <= 0: __SetGrad(model.fc2, False)

        elif int(args.GU) == 311:
            __SetGrad(model.fc3, False)
        
        elif int(args.GU) == 312:
            __SetGrad(model.fc3, False)
            __SetGrad(model.fc2, False)

def GradModulator(args, model):
    with torch.no_grad():
        if int(args.GU) == 211:
            factor = int(args.GU*10000)%10000 / 10000
            model.fc3.weight.grad.data *= factor
            model.fc3.bias.grad.data *= factor

def train_model(args, model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):

    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    args.optimizer, args.learning_rate = optimizer, learning_rate
    model.train(); model = model.to(device)
    args.local_iter_count = 0

    for e in range(epoch):
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            StopGradScheduler(args, model)
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            GradModulator(args, model)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()
            args.local_iter_count += 1
           
        if (e+1) % print_per == 0: model.train()
    
    for params in model.parameters(): params.requires_grad = False
    model.eval()            
    return model


class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss
    
def train_feddecorr_model(args, model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):

    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_fn2 = FedDecorrLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    args.optimizer, args.learning_rate = optimizer, learning_rate
    model.train(); model = model.to(device)
    args.local_iter_count = 0

    for e in range(epoch):
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            StopGradScheduler(args, model)
            y_pred, z = model.forward_feat(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]

            loss += 0.05 * loss_fn2(z)

            optimizer.zero_grad()
            loss.backward()
            GradModulator(args, model)

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()
            args.local_iter_count += 1
           
        if (e+1) % print_per == 0: model.train()
    
    for params in model.parameters(): params.requires_grad = False
    model.eval()            
    return model

def train_fedcm_model(args, model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):

    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    args.local_iter_count = 0

    for e in range(epoch):
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            StopGradScheduler(args, model)
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            GradModulator(args, model)

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)

            idx = 0
            for name, param in model.named_parameters():
                length = len(param.reshape(-1))
                args.alpha * torch.tensor(args.delta[idx:idx+length].reshape(param.shape))
                param.grad.data.multiply_(args.alpha)
                param.grad.data.add_((1-args.alpha) * torch.tensor(args.delta[idx:idx+length].reshape(param.shape)).to(device))
                idx += length

            optimizer.step()
            args.local_iter_count += 1
           
        if (e+1) % print_per == 0: model.train()
    
    for params in model.parameters(): params.requires_grad = False
    model.eval()            
    return model

def train_feddyn_mdl(args, model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    args.optimizer, args.learning_rate = optimizer, learning_rate
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]

    args.local_iter_count = 0
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
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
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

            args.local_iter_count += 1

        if (e+1) % print_per == 0: model.train()
    
    for params in model.parameters(): params.requires_grad = False
    model.eval()
            
    return model

###
def train_fedprox_mdl(args, model, avg_model_param_, mu, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    args.optimizer, args.learning_rate = optimizer, learning_rate
    model.train(); model = model.to(device)
    
    n_par = len(avg_model_param_)
    args.local_iter_count = 0
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
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
    args.local_iter_count = 0
    
    step_loss = 0; n_data_step = 0
    for e in range(epoch):
        # Training
        if is_done:
            break
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
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