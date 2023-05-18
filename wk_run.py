from utils_general import *
from utils_methods import *  
import argparse, datetime, copy, time


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = "fedavg", 
                    choices=["feddyn", "scaffold", "fedavg", "fedprox",
                             "fedexp", "fedcm", "feddecorr",
                             "fedadam", "fedadagrad", "fedavgm"])
parser.add_argument('--epoch', type = int, default = 5)
parser.add_argument('--model_name', type = str, default = "resnet", choices=["cnn", "resnet18", "resnet34"])
parser.add_argument('--norm', type = str, default = "no")


parser.add_argument('--task', type = str, default = "CIFAR100", 
                    choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
parser.add_argument('--balance', type = float, default = 0.,
                    help="0: balanced | 0-1: sampling rate",
                    choices=["CIFAR10", "CIFAR100", "mnist", "emnist"])
parser.add_argument('--dist', type = float, default = 0.0,
                    help="0: iid | 0-1: non-iid with unbalanced_sgm")
parser.add_argument('--n_clients', type = int, default = 10)
parser.add_argument('--act_prob', type = float, default = .1)
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--lr_update_mode', type = str, default = "exp",
                    choices=["exp", "lin"])
parser.add_argument('--lr_decay_per_round', type = float, default = 1,)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--eps', type = float, default = 1e-3)

parser.add_argument('--weight_decay', type = float, default = 1e-3)
parser.add_argument('--momentum', type = float, default = 0)
parser.add_argument('--GU', type = float, default = 0.)

parser.add_argument('--extension', type = str, default = "")

parser.add_argument('--device_idx', type = int, default = 0)
args, _ = parser.parse_known_args()
args.extension += "_EP" + str(args.epoch)

# ==============================================
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
if torch.cuda.device_count() == 2:
    if args.device_idx == 0: args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device_idx == 1: args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using GPU of Device ", args.device, " of ", torch.cuda.device_count())
elif torch.cuda.device_count() == 1:
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using GPU of Device ", args.device, " of ", torch.cuda.device_count())
max_norm = 10

# ==============================================
if args.weight_decay != 1e-3: args.extension += "_WD" + format(args.weight_decay, '.0e')[-1]
if args.model_name == "resnet": args.extension += "_resnet"
if args.model_name == "mobilenet": args.extension += "_mobilenet"
if args.GU == 0: args.GU = 0
if args.GU != 0:
    if int(args.GU) == args.GU: args.GU = int(args.GU)
    args.extension += "_GU" + str(args.GU)
if args.momentum != 0:
    args.extension += "_Mom" + str(int(args.momentum * 10))

# ==============================================
weight_decay       = args.weight_decay
batch_size         = 50
lr_decay_per_round = args.lr_decay_per_round
epoch              = args.epoch
print_per          = 5
args.eps = 1e-5
args.batch_size = 50

if args.task == "TinyImageNet":
    if args.act_prob == 0.01:
        args.com_amount = 400
    if args.act_prob == 0.1:
        args.com_amount = 300
    if args.act_prob == 1.:
        args.com_amount = 500
    args.batch_size = 50

    if args.n_clients == 10 and args.epoch == 5: 
        args.com_amount = 100
    elif args.n_clients == 10 and args.epoch == 3: 
        args.com_amount = 100

if args.task == "CIFAR100":
    if args.act_prob == 0.01:
        args.com_amount = 500
    if args.act_prob == 0.1:
        args.com_amount = 500
    if args.act_prob == 1.:
        args.com_amount = 500
    args.batch_size = 50

    if args.n_clients == 10 and args.epoch == 5: 
        args.com_amount = 100

if args.task == "CIFAR10":
    if args.act_prob == 0.01:
        args.com_amount = 500
    if args.act_prob == 0.1:
        args.com_amount = 500
    if args.act_prob == 1.:
        args.com_amount = 500
    args.batch_size = 50

    if args.n_clients == 10:
        args.com_amount = 200

if args.task == "emnist":
    if args.act_prob == 1:
        args.com_amount = 500
        args.batch_size = 50
            
    if args.act_prob == 0.01:
        args.com_amount = 500
        args.batch_size = 50

args.extension += "__LR" + str(args.lr)
if args.balance == 0.: args.balance = 0
if args.dist == 0.: 
    args.dist = 0

# ==============================================
if args.task == "CIFAR100" and args.model_name == "resnet":
    savepath = 'Output_GU_C100_Resnet/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)
elif args.task == "TinyImageNet" and args.model_name == "resnet":
    savepath = 'Output_GU_TIN_Resnet/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)
elif args.task == "CIFAR100" and args.model_name == "mobilenetv2":
    savepath = 'Output_GU_C100_Mobilenetv2/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)
elif args.task == "TinyImageNet" and args.model_name == "mobilenetv2":
    savepath = 'Output_GU_TIN_Mobilenetv2/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)
elif args.task == "CIFAR10":
    savepath = 'Output_GU_C10/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)
elif args.task == "CIFAR100":
    savepath = 'Output_GU_C100/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)
elif args.task == "TinyImageNet":
    savepath = 'Output_GU_TIN/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)
else:
    savepath = 'Output_GU/{}-{}-B{}-D{}-N{}-P{}_{}'.format(args.task, args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)


if not os.path.exists(savepath): os.mkdir(savepath)
args.savename = savepath+'/s{}.csv'.format(args.seed)
args.savepath = savepath
print("Save Logs at: ", args.savename), print()

if os.path.exists(args.savename): 
    print("The seed already exists.")
    print("The path is ", args.savename)
    print()
    time.sleep(0.1)
    import sys
    sys.exit()

# ==============================================
model_func = lambda : client_model(args.task.lower(), args)
init_model = model_func()


if args.balance == 0.: args.balance = 0
if args.dist == 0.: 
    args.dist = 0
    data_obj = DatasetObject(dataset=args.task, n_client=args.n_clients, rule="iid", unbalanced_sgm=0)
else: 
    data_obj = DatasetObject(dataset=args.task, n_client=args.n_clients, unbalanced_sgm=0, 
                             rule='Dirichlet', rule_arg=args.dist)
# ===========================

if args.mode == "feddyn":
    args.alpha_coef = 1e-2
elif args.mode == "scaffold":
    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / args.n_clients
    n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
    args.n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
elif args.mode == "fedprox":
    args.mu = 1e-4
elif args.mode in ["fedadam", "fedadagrad", "fedadagrad1", "fedadagrad2", "fedavgm"]:
    # print("Hi")
    args.b1 = 0.9
    args.b2 = 0.99

train(args=args, data_obj=data_obj, learning_rate=args.lr, batch_size=batch_size,
             epoch=epoch, print_per=print_per, weight_decay=weight_decay,
             model_func=model_func, init_model=init_model, lr_decay_per_round=lr_decay_per_round)
