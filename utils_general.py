from utils_libs import *
from utils_dataset import *
from utils_models import *

max_norm = 10

def get_acc_loss(args, data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0; loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
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

def set_client_from_params(args, mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        # print(name, length)
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(args.device))
        idx += length
    
    if args.model_name in ["mobilenetv2"]: mdl.load_state_dict(dict_param, strict=False)
    else: mdl.load_state_dict(dict_param)    
                
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

def __SetGrad(args, tensor, sign):
    if args.model_name in ["cnn"] or args.task in ["emnist"]:
        tensor.weight.requires_grad = sign
        tensor.bias.requires_grad = sign
    elif args.model_name in ["resnet", "resnet34", "mobilenetv2"]:
        for p1, p2 in tensor.named_parameters():
            p2.requires_grad = sign

def GlobalGradScheduler(args, new_model, avg_model, new_avg_model_param):
    if int(args.GGU) == 0: return set_client_from_params(new_model, new_avg_model_param)

def StopGradScheduler(args, model):
    with torch.no_grad():
        for p1 , p2 in model.named_parameters(): p2.requires_grad = True
        # Bottom up
        
        if args.model_name == "mobilenetv2":
            if int(args.GU) == 111:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.features[2], False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.features[3], False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.features[4], False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.model.features[5], False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.model.features[6], False)
                if  args.local_iter_count // factor <= 5: __SetGrad(args, model.model.features[7], False)
                if  args.local_iter_count // factor <= 6: __SetGrad(args, model.model.features[8], False)
                if  args.local_iter_count // factor <= 7: __SetGrad(args, model.model.features[9], False)
                if  args.local_iter_count // factor <= 8: __SetGrad(args, model.model.features[10], False)
                if  args.local_iter_count // factor <= 9: __SetGrad(args, model.model.features[11], False)
                if  args.local_iter_count // factor <= 10: __SetGrad(args, model.model.features[12], False)
                if  args.local_iter_count // factor <= 11: __SetGrad(args, model.model.features[13], False)
                if  args.local_iter_count // factor <= 12: __SetGrad(args, model.model.features[14], False)
                if  args.local_iter_count // factor <= 13: __SetGrad(args, model.model.features[15], False)
                if  args.local_iter_count // factor <= 14: __SetGrad(args, model.model.features[16], False)
                if  args.local_iter_count // factor <= 15: 
                    __SetGrad(args, model.model.features[17], False)
                    __SetGrad(args, model.model.features[18], False)
                    __SetGrad(args, model.model.classifier, False)

        if args.model_name == "resnet34":
            if int(args.GU) == 104:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.layer2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.layer3, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.layer4, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.fc, False)
            elif int(args.GU) == 116:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.layer1[1], False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.layer1[2], False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.layer2[0], False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.model.layer2[1], False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.model.layer2[2], False)
                if  args.local_iter_count // factor <= 5: __SetGrad(args, model.model.layer2[3], False)
                if  args.local_iter_count // factor <= 6: __SetGrad(args, model.model.layer3[0], False)
                if  args.local_iter_count // factor <= 7: __SetGrad(args, model.model.layer3[1], False)
                if  args.local_iter_count // factor <= 8: __SetGrad(args, model.model.layer3[2], False)
                if  args.local_iter_count // factor <= 9: __SetGrad(args, model.model.layer3[3], False)
                if  args.local_iter_count // factor <= 10: __SetGrad(args, model.model.layer3[4], False)
                if  args.local_iter_count // factor <= 11: __SetGrad(args, model.model.layer3[5], False)
                if  args.local_iter_count // factor <= 12: __SetGrad(args, model.model.layer4[0], False)
                if  args.local_iter_count // factor <= 13: __SetGrad(args, model.model.layer4[1], False)
                if  args.local_iter_count // factor <= 14: __SetGrad(args, model.model.layer4[2], False)
                if  args.local_iter_count // factor <= 15: __SetGrad(args, model.model.fc, False)

            return 

        if args.model_name == "resnet":
            if int(args.GU) == 104:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.layer2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.layer3, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.layer4, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.fc, False)
            elif int(args.GU) == 108:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.layer1[1], False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.layer2[0], False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.layer2[1], False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.model.layer3[0], False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.model.layer3[1], False)
                if  args.local_iter_count // factor <= 5: __SetGrad(args, model.model.layer4[0], False)
                if  args.local_iter_count // factor <= 6: __SetGrad(args, model.model.layer4[1], False)
                if  args.local_iter_count // factor <= 6: __SetGrad(args, model.model.fc, False)

            return 

        if int(args.GU) == 111:
            if args.model_name == "resnet":
                factor = (args.GU*100)%100
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.layer1, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.layer2, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.layer3, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.model.layer4, False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.model.fc, False)
            
            elif args.model_name == "cnn" and args.task in ["CIFAR10", "CIFAR100"]:
                factor = (args.GU*100)%100
                if  args.local_iter_count // factor <= 0: __SetGrad(model.conv2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(model.fc1, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(model.fc2, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(model.fc3, False)
            elif args.model_name == "cnn" and args.task in ["TinyImageNet"]:
                factor = (args.GU*100)%100
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.conv2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.conv3, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.fc1, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.fc2, False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.fc3, False)

        if int(args.GU) == 110:
            if args.model_name == "cnn" and args.task in ["TinyImageNet"]:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.conv2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.conv3, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.fc1, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.fc2, False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.fc3, False)
            else:
                assert 0 == 1

        if int(args.GU) == 112:           
            if args.model_name == "cnn" and args.task in ["TinyImageNet"]:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.fc2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.fc1, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.conv3, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.conv2, False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.conv1, False)

        if int(args.GU) == 311:
            if args.model_name == "cnn" and args.task in ["TinyImageNet"]:
                __SetGrad(args, model.fc3, False)

        if int(args.GU) == 312:
            if args.model_name == "cnn" and args.task in ["TinyImageNet"]:
                __SetGrad(args, model.fc3, False)
                __SetGrad(args, model.fc2, False)

        if int(args.GU) == 113:
            if args.model_name == "resnet":
                factor = (args.GU*100)%100
                # print("Hi")
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.layer2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.layer3, False)
                if  args.local_iter_count // factor <= 2: 
                    __SetGrad(args, model.model.layer4, False)
                    __SetGrad(args, model.model.fc, False)

def train_model(args, model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):

    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    args.local_iter_count = 0
    for e in range(epoch):
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            # print(int(np.ceil(n_trn/batch_size)))
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
        if (e+1) % print_per == 0: model.train()
    for params in model.parameters(): params.requires_grad = False
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

def train_feddecorr_model(args, model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):

    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_fn2 = FedDecorrLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    args.local_iter_count = 0

    for e in range(epoch):
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            StopGradScheduler(args, model)
            y_pred, z = model.forward_feat(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]

            loss2 = loss_fn2(z)
            loss += 0.001 * loss2

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()
            args.local_iter_count += 1
        
        if (e+1) % print_per == 0: model.train()

        # break
    
    for params in model.parameters(): params.requires_grad = False
    model.eval()            
    return model

def train_feddyn_mdl(args, model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    
    n_par = get_mdl_params([model_func()]).shape[1]

    args.local_iter_count = 0
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
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
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    
    n_par = len(avg_model_param_)
    args.local_iter_count = 0
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
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
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=args.momentum)
    model.train(); model = model.to(args.device)
    
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
        
            print(state_params_diff.sum())
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
