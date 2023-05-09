from utils_libs import *
from utils_dataset import *
from utils_0301_models import *
from utils_0305_general import *

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
    
    memory = pd.DataFrame(columns=['task', 'mode', "gn", "balance", "distribution", "n_clients", "act_prob", 
                                   "seed", "epoch", "a1", "a2", "a3", "a4", "l1", "l2", "l3", "l4"])

    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    cent_x, cent_y = np.concatenate(clnt_x, axis=0), np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    if args.mode == "fedavg":
        weight_list = weight_list.reshape((n_clnt, 1))
    elif args.mode in ["scaffold", "feddyn"]:
        weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it
    elif args.mode in ["fedprox"]:
        weight_list = weight_list.reshape((n_clnt, 1))

    savepath = 'Output/{}-{}-G{}{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.use_gn, args.use_mean, args.balance, args.distribution, args.n_clients, args.act_prob, args.extension)
    if args.lr_update_mode == "lin":
        savepath += "-Lin"
    if args.use_gn >= 1 or args.use_mean >= 1:
        savepath += "-LR{}".format(args.lr)
    # if args.SWA == 0: pass
    # else: savepath += "-SWA{}".format(args.SWA)
    print(savepath), print()
    if not os.path.exists(savepath): os.mkdir(savepath)
        
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if args.mode == "scaffold":
        state_param_list = np.zeros((n_clnt+1, n_par)).astype('float32')

    elif args.mode == "feddyn":
        local_param_list = np.zeros((n_clnt, n_par)).astype('float32')
        cld_model = model_func().to(device)
        cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        cld_mdl_param = get_mdl_params([cld_model], n_par)[0]

    starttime = time.time()
    for i in range(args.com_amount):
        
        args.epoch = i
        
        # if (args.gn_stop_epoch!=0 and args.epoch >= args.gn_stop_epoch):
        #     learning_rate = 0.1

        inc_seed = 0
        while(True):
            # Fix randomness in client selection
            np.random.seed(i + inc_seed + args.seed)
            act_list    = np.random.uniform(size=n_clnt)
            act_clients = act_list <= args.act_prob
            selected_clnts = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clnts) != 0: break

        if args.hs in [21]:
            args.freeze_name = np.random.choice(["conv2", "fc1", "fc2", "fc3"])

        if args.mode in ["fedavg", "fedprox"]:
            if args.mode == "fedprox":
                avg_model_param = get_mdl_params([avg_model], n_par)[0]
                avg_model_param_tensor = torch.tensor(avg_model_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                trn_x, trn_y = clnt_x[clnt], clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                for params in clnt_models[clnt].parameters(): params.requires_grad = True
                args.clnt = clnt
                if args.mode == "fedavg": clnt_models[clnt] = train_model(args, clnt_models[clnt], trn_x, trn_y, 
                                                                        current_lr(args, learning_rate, lr_decay_per_round, i), 
                                                                        batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                if args.mode == "fedprox": clnt_models[clnt] = train_fedprox_mdl(args, clnt_models[clnt], avg_model_param_tensor, args.mu, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            # if args.PCOS == 1:
            #     avg_model_param = get_mdl_params([avg_model], n_par)[0][None,:]
            #     # print(np.shape(clnt_params_list[selected_clnts]))
            #     # print(np.shape(avg_model_param))
            #     norm_avg_params = torch.norm(torch.sum((torch.Tensor(clnt_params_list[selected_clnts])-avg_model_param)*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis=0)).item()
            #     avg_norm_params = torch.mean(torch.norm(torch.Tensor(clnt_params_list[selected_clnts])-avg_model_param, dim=1)).item()
            #     # print(len(selected_clnts))
            #     # print("Epoch: {}. N1: {:.4f}. N2: {:.4f}. Ratio: {:.4f}.".format(i, avg_norm_params, norm_avg_params, avg_norm_params/norm_avg_params))
            #     # X = torch.Tensor(clnt_params_list[selected_clnts])-avg_model_param
            #     # CS = torch.nn.functional.cosine_similarity(X,torch.roll(X, 1, 0), dim=1)
            #     # print("Epoch: {}. CS: {:.4f}.".format(i, torch.mean(CS)))
                
            #     A = avg_model_param = get_mdl_params([avg_model], n_par)[0]
            #     B = np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0)
            #     # print(avg_norm_params/norm_avg_params)
            #     avg_model = set_client_from_params(model_func(), (B - A) *  avg_norm_params/norm_avg_params  + A)
            #     all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))
            # if args.PCOS == 0:
            #     # A = avg_model_param = get_mdl_params([avg_model], n_par)[0]
            B = np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0)


            
            avg_model = set_client_from_params(model_func(), B)
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))        

            # if args.svm == 1:
            #     args.svmodel = copy.deepcopy(avg_model).to(device)

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
                state_params_diff_curr = torch.tensor(-state_param_list[clnt] + state_param_list[-1]/weight_list[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_scaffold_mdl(args, clnt_models[clnt], model_func, state_params_diff_curr, trn_x, trn_y, 
                                                       current_lr(args, learning_rate, lr_decay_per_round, i), batch_size, args.n_minibatch, print_per, weight_decay, data_obj.dataset)
                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_param_list[clnt] - state_param_list[-1] + 1/args.n_minibatch/learning_rate * (prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_param_list[clnt])*weight_list[clnt]
                state_param_list[clnt] = new_c
                clnt_params_list[clnt] = curr_model_param

            avg_model_params = global_learning_rate*np.mean(clnt_params_list[selected_clnts], axis = 0) + (1-global_learning_rate)*prev_params
            state_param_list[-1] += 1 / n_clnt * delta_c_sum

            avg_model = set_client_from_params(model_func().to(device), avg_model_params)
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))

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
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))
            cld_model = set_client_from_params(model_func().to(device), cld_mdl_param) 

        if (i+1) % 10 == 0:
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
            df_append(memory, [args.task, args.mode, args.use_gn, args.balance, args.distribution, args.n_clients, 
                               args.act_prob, args.seed, i, a1, a2, a3, a4, l1, l2, l3, l4])
            memory.to_csv(savepath+'/s{}.csv'.format(args.seed))
            starttime = time.time()

    ts = datetime.datetime.now()
    memory.to_csv(savepath+'/s{}-t{}.csv'.format(args.seed, ts.strftime("%H%M")))