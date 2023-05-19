from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *

import pandas as pd
import datetime, time


def df_append(df, data): df.loc[len(df)] = data

### Methods
def train(args, data_obj, model_func, init_model):
    
    memory = pd.DataFrame(columns=['task', 'mode', "balance", "distribution", "n_clients", "act_prob", "seed", "GUP1", "GUP2",
                                   "n_epochs", "epoch", "time", "a1", "a2", "a3", "a4", "l1", "l2", "l3", "l4"])

    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    cent_x, cent_y = np.concatenate(clnt_x, axis=0), np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    if args.mode in ["fedavg", "fedexp", "fedcm", "feddecorr"]: weight_list = weight_list.reshape((n_clnt, 1))
    elif args.mode in ["scaffold", "feddyn", "fedadam", "fedadagrad", "fedavgm"]: weight_list = weight_list / np.sum(weight_list) * n_clnt
    elif args.mode in ["fedprox"]: weight_list = weight_list.reshape((n_clnt, 1))
        
    n_par = len(get_mdl_params([model_func()])[0])
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    avg_model = model_func().to(args.device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    all_model = model_func().to(args.device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        
    if args.mode == "scaffold":
        state_param_list = np.zeros((n_clnt+1, n_par)).astype('float32')
    elif args.mode in ["fedadam", "fedadagrad", "fedavgm", "fedcm"]:
        momentum = np.zeros((n_par)).astype('float32')
        d = np.zeros((n_par)).astype('float32')
        v = np.zeros((n_par)).astype('float32')
        args.delta = np.zeros((n_par)).astype('float32')
    elif args.mode in ["feddyn", "feddyn2"]:
        local_param_list = np.zeros((n_clnt, n_par)).astype('float32')
        cld_model = model_func().to(args.device)
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
            # if args.mode == "fedprox":
            #     avg_model_param = get_mdl_params([avg_model], n_par)[0]
            #     avg_model_param_tensor = torch.tensor(avg_model_param, dtype=torch.float32, device=args.device)

            for clnt in selected_clnts:
                # trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                # clnt_models[clnt] = model_func().to(args.device)
                # clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                
                # for params in clnt_models[clnt].parameters(): params.requires_grad = True
                # if args.mode in ["fedavg"]: clnt_models[clnt] = train_model(args, clnt_models[clnt], trn_x, trn_y)
                # # if args.mode == "feddecorr": clnt_models[clnt] = train_feddecorr_model(args, clnt_models[clnt], trn_x, trn_y)
                # # if args.mode == "fedprox": clnt_models[clnt] = train_fedprox_mdl(args, clnt_models[clnt], avg_model_param_tensor, args.mu, trn_x, trn_y)
                # clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # clnt_models[clnt] 

                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_model = model_func().to(args.device)
                clnt_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                
                for params in clnt_model.parameters(): params.requires_grad = True
                if args.mode in ["fedavg"]: clnt_model = train_model(args, clnt_model, trn_x, trn_y)
                # if args.mode == "feddecorr": clnt_models[clnt] = train_feddecorr_model(args, clnt_models[clnt], trn_x, trn_y)
                # if args.mode == "fedprox": clnt_models[clnt] = train_fedprox_mdl(args, clnt_models[clnt], avg_model_param_tensor, args.mu, trn_x, trn_y)
                clnt_params_list[clnt] = get_mdl_params([clnt_model], n_par)[0]
                del clnt_model


            avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(args, model_func(), avg_mdl_param) 
        
        if args.mode in ["fedexp"]:

            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(args.device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters(): params.requires_grad = True

                clnt_models[clnt] = train_model(args, clnt_models[clnt], trn_x, trn_y)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            avg_model_param = get_mdl_params([avg_model], n_par)
            norm_avg_params = torch.norm(torch.mean((torch.Tensor(clnt_params_list[selected_clnts])-torch.Tensor(avg_model_param)), axis=0)).item()
            avg_norm_params = torch.mean(torch.norm(torch.Tensor(clnt_params_list[selected_clnts])-torch.Tensor(avg_model_param), dim=1)**2).item()
            lr = max(1, avg_norm_params/(norm_avg_params**2 + len(selected_clnts*0.001)))
            A, B = avg_model_param[0], np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(args, model_func(), (B - A) *  lr  + A)

        if args.mode in ["fedadagrad", "fedadam", "fedavgm"]:
            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(args.device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters(): params.requires_grad = True
                clnt_models[clnt] = train_model(args, clnt_models[clnt], trn_x, trn_y)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            avg_model_param = get_mdl_params([avg_model], n_par)[0]
            delta = (np.mean(clnt_params_list[selected_clnts], axis = 0) - avg_model_param)
            if args.mode == "fedadagrad":
                momentum = args.b1 * momentum + (1-args.b1) * delta
                v = v + delta ** 2
                avg_model = set_client_from_params(args, model_func(), avg_model_param + 0.1 * momentum / (np.sqrt(v)+0.01))
            if args.mode == "fedavgm":
                momentum = args.b1 * momentum + (1-args.b1) * delta
                avg_model = set_client_from_params(args, model_func(), avg_model_param + momentum)

        elif args.mode == "scaffold":

            global_learning_rate = 1
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(args.device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters(): params.requires_grad = True
                # Scale down c
                state_params_diff_curr = torch.tensor(state_param_list[-1] - state_param_list[clnt], dtype=torch.float32, device=args.device)
                clnt_models[clnt] = train_scaffold_mdl(args, clnt_models[clnt], model_func, state_params_diff_curr, trn_x, trn_y, 
                                                       args.n_minibatch)
                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_param_list[clnt] - state_param_list[-1] + 1/args.n_minibatch/args.lr * (prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_param_list[clnt])
                state_param_list[clnt] = new_c
                clnt_params_list[clnt] = curr_model_param

            avg_model_params = np.mean(clnt_params_list[selected_clnts], axis = 0)
            state_param_list[-1] += 1 / n_clnt * delta_c_sum
            avg_model = set_client_from_params(args, model_func().to(args.device), avg_model_params)
            # all_model = set_client_from_params(args, model_func(), np.mean(clnt_params_list, axis = 0))

        elif args.mode in ["feddyn"]:

            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=args.device)
            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(args.device)
                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
                for params in model.parameters(): params.requires_grad = True

                # Scale down
                alpha_coef_adpt = args.alpha_coef / weight_list[clnt] # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device=args.device)
                clnt_models[clnt] = train_feddyn_mdl(args, model, model_func, alpha_coef_adpt, cld_mdl_param_tensor, 
                                                     local_param_list_curr, trn_x, trn_y)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

            avg_model = set_client_from_params(args, model_func(), avg_mdl_param)
            cld_model = set_client_from_params(args, model_func().to(args.device), cld_mdl_param) 

        if (i+1) % 10 == 0:
            all_model = set_client_from_params(args, model_func(), np.mean(clnt_params_list, axis = 0))
            for params in all_model.parameters(): params.requires_grad = True
            l1, a1 = get_acc_loss(args, data_obj.tst_x, data_obj.tst_y, avg_model) 
            l2, a2 = get_acc_loss(args, cent_x, cent_y, avg_model)
            l3, a3 = get_acc_loss(args, data_obj.tst_x, data_obj.tst_y, all_model)
            l4, a4 = get_acc_loss(args, cent_x, cent_y, all_model)
            print("Round {:<4}. Elapsed time: {:.0f}".format(i+1, time.time()-starttime))
            print("\t \t Train: {:.2f} \t Test: {:.2f}.".format(a4*100, a3*100))
            print("\t \t Train: {:.2f} \t Test: {:.2f}.".format(a2*100, a1*100))

    
    # memory = pd.DataFrame(columns=['task', 'mode', "balance", "distribution", "n_clients", "act_prob", "seed", 
                                     # "GUP1", "GUP2",
    #                                "n_epochs", "epoch","time", "a1", "a2", "a3", "a4", "l1", "l2", "l3", "l4"])

            df_append(memory, [args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.seed, 
                               args.GUP1, args.GUP2, args.epoch, int(time.time()-starttime), i, a1, a2, a3, a4, int(l1), int(l2), int(l3), int(l4)])
            memory.to_csv(args.savename)
            starttime = time.time()

    ts = datetime.datetime.now()
    memory.to_csv(args.savepath+'/s{}-t{}.csv'.format(args.seed, ts.strftime("%H%M")))
