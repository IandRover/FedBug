from utils_libs import *
from utils_dataset import *
from utils_0301_models import *
from utils_0407_general import *

import pandas as pd
import datetime, time
 

def df_append(df, data): df.loc[len(df)] = data

def current_lr(args, learning_rate, lr_decay_per_round, i):
    if args.lr_update_mode == "exp": return learning_rate * (lr_decay_per_round ** i)
    if args.lr_update_mode == "lin": 
        lr = learning_rate * (1 - i/args.com_amount)
        # print(lr)
        return lr

### Methods
def train(args, data_obj, learning_rate, batch_size, epoch, print_per, weight_decay, model_func, init_model, lr_decay_per_round, rand_seed=0):
    
    memory = pd.DataFrame(columns=['task', 'mode', "balance", "distribution", "n_clients", "act_prob", 
                                   "seed", "epoch", "a1", "a2", "a3", "a4", "l1", "l2", "l3", "l4"])

    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    cent_x, cent_y = np.concatenate(clnt_x, axis=0), np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    if args.mode in ["fedavg", "fedexp", "fedcm", "feddecorr", "fedavg1", "fedavg2", "fedavg3", "fedavg4", "fedavg5"]:
        weight_list = weight_list.reshape((n_clnt, 1))
    elif args.mode in ["scaffold", "feddyn", "fedadam", "fedadagrad", "fedavgm"]:
        weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it
    elif args.mode in ["fedprox"]:
        weight_list = weight_list.reshape((n_clnt, 1))

    if args.task == "CIFAR10":
        if args.momentum > 0 or args.epoch > 5:
            savepath = 'Output_GU_C10_M9/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.distribution, args.n_clients, args.act_prob, args.extension)
        else:
            savepath = 'Output_GU_C10/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.distribution, args.n_clients, args.act_prob, args.extension)
    elif args.task == "CIFAR100":
        savepath = 'Output_GU_C100/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.distribution, args.n_clients, args.act_prob, args.extension)
    elif args.task == "emnist26":
        savepath = 'Output_GU_emnist26/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.distribution, args.n_clients, args.act_prob, args.extension)
    elif args.task == "emnist":
        savepath = 'Output_GU_emnist/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.distribution, args.n_clients, args.act_prob, args.extension)
    else:
        savepath = 'Output_GU/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.distribution, args.n_clients, args.act_prob, args.extension)
    
    if not os.path.exists(savepath): os.mkdir(savepath)
    savename = savepath+'/s{}.csv'.format(args.seed)
    print(savename), print()
    
    if os.path.exists(savename): 
        print("The seed already exists.")
        print("The path is ", savename)
        print()
        time.sleep(0.1)
        import sys
        sys.exit()
        
    args.n_par = n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if args.mode == "scaffold":
        state_param_list = np.zeros((n_clnt+1, n_par)).astype('float32')
    elif args.mode in ["fedadam", "fedadagrad", "fedavgm", "fedcm"]:
        momentum = np.zeros((n_par)).astype('float32')
        d = np.zeros((n_par)).astype('float32')
        v = np.zeros((n_par)).astype('float32')
        args.delta = np.zeros((n_par)).astype('float32')
    elif args.mode == "feddyn":
        local_param_list = np.zeros((n_clnt, n_par)).astype('float32')
        cld_model = model_func().to(device)
        cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        cld_mdl_param = get_mdl_params([cld_model], n_par)[0]

    starttime = time.time()
    for i in range(args.com_amount):
        
        args.global_epoch = i

        inc_seed = 0
        while(True):
            np.random.seed(i +  inc_seed* 1000 + args.seed * 10000000)
            act_list    = np.random.uniform(size=n_clnt)
            act_clients = act_list <= args.act_prob
            selected_clnts = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clnts) != 0: break

        if args.mode in ["fedavg", "fedprox", "feddecorr", "fedavg1", "fedavg2", "fedavg3", "fedavg4", "fedavg5"]:
            if args.mode == "fedprox":
                avg_model_param = get_mdl_params([avg_model], n_par)[0]
                avg_model_param_tensor = torch.tensor(avg_model_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters(): params.requires_grad = True
                if args.mode in ["fedavg", "fedavg1", "fedavg2", "fedavg3", "fedavg4", "fedavg5"]: clnt_models[clnt] = train_model(args, clnt_models[clnt], trn_x, trn_y, 
                                                                        current_lr(args, learning_rate, lr_decay_per_round, i), 
                                                                        batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                if args.mode == "feddecorr": clnt_models[clnt] = train_feddecorr_model(args, clnt_models[clnt], trn_x, trn_y, 
                                                                        current_lr(args, learning_rate, lr_decay_per_round, i), 
                                                                        batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                if args.mode == "fedprox": clnt_models[clnt] = train_fedprox_mdl(args, clnt_models[clnt], avg_model_param_tensor, args.mu, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # new_avg_model_param = np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0)
            # avg_model = GlobalGradScheduler(args, model_func(), avg_model, new_avg_model_param)      

            avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func(), avg_mdl_param) 
        
        if args.mode in ["fedexp"]:

            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters(): params.requires_grad = True

                clnt_models[clnt] = train_model(args, clnt_models[clnt], trn_x, trn_y, 
                                                current_lr(args, learning_rate, lr_decay_per_round, i), 
                                                batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            avg_model_param = get_mdl_params([avg_model], n_par)
            norm_avg_params = torch.norm(torch.mean((torch.Tensor(clnt_params_list[selected_clnts])-avg_model_param), axis=0)).item()
            avg_norm_params = torch.mean(torch.norm(torch.Tensor(clnt_params_list[selected_clnts])-avg_model_param, dim=1)**2).item()
            lr = max(1, avg_norm_params/(norm_avg_params**2 + len(selected_clnts*0.001)))
            A, B = avg_model_param[0], np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func(), (B - A) *  lr  + A)

        if args.mode in ["fedadagrad", "fedadam", "fedavgm"]:
            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters(): params.requires_grad = True
                clnt_models[clnt] = train_model(args, clnt_models[clnt], trn_x, trn_y, 
                                                current_lr(args, learning_rate, lr_decay_per_round, i), 
                                                batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            avg_model_param = get_mdl_params([avg_model], n_par)[0]
            delta = (np.mean(clnt_params_list[selected_clnts], axis = 0) - avg_model_param)
            if args.mode == "fedadagrad":
                momentum = args.b1 * momentum + (1-args.b1) * delta
                v = v + delta ** 2
                avg_model = set_client_from_params(model_func(), avg_model_param + 0.1 * momentum / (np.sqrt(v)+0.01))
            if args.mode == "fedavgm":
                momentum = args.b1 * momentum + (1-args.b1) * delta
                avg_model = set_client_from_params(model_func(), avg_model_param + momentum)

        elif args.mode == "scaffold":

            global_learning_rate = 1
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters(): params.requires_grad = True
                # Scale down c
                state_params_diff_curr = torch.tensor(state_param_list[-1] - state_param_list[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_scaffold_mdl(args, clnt_models[clnt], model_func, state_params_diff_curr, trn_x, trn_y, 
                                                       current_lr(args, learning_rate, lr_decay_per_round, i), batch_size, args.n_minibatch, print_per, weight_decay, data_obj.dataset)
                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_param_list[clnt] - state_param_list[-1] + 1/args.n_minibatch/learning_rate * (prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_param_list[clnt])
                state_param_list[clnt] = new_c
                clnt_params_list[clnt] = curr_model_param

            avg_model_params = np.mean(clnt_params_list[selected_clnts], axis = 0)
            state_param_list[-1] += 1 / n_clnt * delta_c_sum
            avg_model = set_client_from_params(model_func().to(device), avg_model_params)
            # all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))

        elif args.mode == "feddyn":

            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
                for params in model.parameters(): params.requires_grad = True

                # Scale down
                alpha_coef_adpt = args.alpha_coef / weight_list[clnt] # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_feddyn_mdl(args, model, model_func, alpha_coef_adpt, cld_mdl_param_tensor, 
                                                     local_param_list_curr, trn_x, trn_y, 
                                                     current_lr(args, learning_rate, lr_decay_per_round, i), batch_size, epoch, 
                                                     print_per, weight_decay, data_obj.dataset)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

            avg_model = set_client_from_params(model_func(), avg_mdl_param)
            
            cld_model = set_client_from_params(model_func().to(device), cld_mdl_param) 

        if (i+1) % 10 == 0:

            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))

            l1, a1 = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset) 
            # print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            l2, a2 = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            # print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            l3, a3 = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            # print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            l4, a4 = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            # print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            print("Round {:<4}. Elapsed time: {:.0f}".format(i+1, time.time()-starttime))
            print("\t \t Train: {:.2f} \t Test: {:.2f}.".format(a4*100, a3*100))
            print("\t \t Train: {:.2f} \t Test: {:.2f}.".format(a2*100, a1*100))
            df_append(memory, [args.task, args.mode, args.balance, args.distribution, args.n_clients, 
                               args.act_prob, args.seed, i, a1, a2, a3, a4, l1, l2, l3, l4])
            memory.to_csv(savepath+'/s{}.csv'.format(args.seed))
            starttime = time.time()

    ts = datetime.datetime.now()
    memory.to_csv(savepath+'/s{}-t{}.csv'.format(args.seed, ts.strftime("%H%M")))