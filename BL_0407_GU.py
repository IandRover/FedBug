from utils_0407_general import *
from utils_0407_methods import *  
import argparse, datetime, copy, time


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = "fedavg", 
                    choices=["feddyn", "scaffold", "fedavg", "fedprox", "fedavg1", "fedavg2", "fedavg3", "fedavg4", "fedavg5",
                             "fedexp", "fedcm", "feddecorr",
                             "fedadam", "fedadagrad", "fedavgm"])
parser.add_argument('--epoch', type = int, default = 5)
parser.add_argument('--norm', type = str, default = "no")

parser.add_argument('--task', type = str, default = "CIFAR100", 
                    choices=["CIFAR10", "CIFAR100", "mnist", "emnist", "emnist26", "TinyImageNet"])
parser.add_argument('--balance', type = float, default = 0.,
                    help="0: balanced | 0-1: sampling rate",
                    choices=["CIFAR10", "CIFAR100", "mnist", "emnist"])
parser.add_argument('--distribution', type = float, default = 0.0,
                    help="0: iid | 0-1: non-iid with unbalanced_sgm")
parser.add_argument('--n_clients', type = int, default = 100)
parser.add_argument('--act_prob', type = float, default = .01)
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--lr_update_mode', type = str, default = "exp",
                    choices=["exp", "lin"])
parser.add_argument('--lr_decay_per_round', type = float, default = 1,)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--eps', type = float, default = 1e-3)

parser.add_argument('--lradp', type = int, default = 0, choices=[0,1,2,3])


parser.add_argument('--momentum', type = float, default = 0, choices=[0,0.9])
# parser.add_argument('--FConly', type = int, default = -1)
# parser.add_argument('--E', type = int, default = 0)

parser.add_argument('--weight_decay', type = float, default = 1e-3)
parser.add_argument('--GU', type = float, default = 0.)
parser.add_argument('--GGU', type = float, default = 0.)

parser.add_argument('--extension', type = str, default = "")
args, _ = parser.parse_known_args()

if args.epoch != 5: 
    args.extension += "_EP" + str(args.epoch)

# if args.weight_decay != 1e-3: 
#     args.extension += "_WD" + format(args.weight_decay, '.0e')[-1]


if args.GU == 0: args.GU = 0
if args.GGU == 0: args.GGU = 0
if args.GU != 0:
    if int(args.GU) == args.GU: args.GU = int(args.GU)
    args.extension += "_GU" + str(args.GU)
if args.GGU != 0:
    if int(args.GGU) == args.GGU: args.GGU = int(args.GGU)
    args.extension += "_GGU" + str(args.GGU)

if args.lr != 0.1: args.extension += "_LR" + str(args.lr)

if args.lradp in [1,2,3]: args.extension += "_LRadp"+ str(args.lradp)
if args.momentum in [0.9]: args.extension += "_M9"

##################### My hyperparams

if args.balance == 0.: args.balance = 0
if args.distribution == 0.: 
    args.distribution = 0
    data_obj = DatasetObject(dataset=args.task, n_client=args.n_clients, rule="iid", unbalanced_sgm=0)
else: 
    data_obj = DatasetObject(dataset=args.task, n_client=args.n_clients, unbalanced_sgm=0, 
                             rule='Dirichlet', rule_arg=args.distribution)

###
# com_amount         = 1000
weight_decay       = args.weight_decay
batch_size         = 50
lr_decay_per_round = args.lr_decay_per_round
epoch              = args.epoch
print_per          = 5
args.eps = 1e-5

##################### My hyperparams
args.batch_size = 50

if args.task == "TinyImageNet":
    if args.act_prob == 0.01:
        args.com_amount = 700
    if args.act_prob == 0.1:
        args.com_amount = 500
    if args.act_prob == 1.:
        args.com_amount = 600
    args.batch_size = 50

if args.task == "CIFAR100":
    if args.act_prob == 0.01:
        args.com_amount = 600
    if args.act_prob == 0.1:
        args.com_amount = 500
        lr_decay_per_round = 0.998
    if args.act_prob == 1.:
        args.com_amount = 600
    args.batch_size = 50

if args.task == "CIFAR10":
    if args.act_prob == 0.01:
        args.com_amount = 600
        if args.mode == "feddyn": args.com_amount = 1000
    if args.act_prob == 0.1:
        args.com_amount = 300
    # if args.act_prob == 1.:
    #     args.com_amount = 600
    args.batch_size = 50

if args.task == "emnist":
    if args.act_prob == 1:
        args.com_amount = 100
        args.batch_size = 50
            
    if args.act_prob == 0.1:
        args.com_amount = 100
        args.batch_size = 50

    if args.act_prob == 0.01:
        args.com_amount = 100
        args.batch_size = 50

if args.task == "emnist26":
    if args.act_prob == 1:
        args.com_amount = 100
        args.batch_size = 50
            
    if args.act_prob == 0.1:
        args.com_amount = 100
        args.batch_size = 50

    if args.act_prob == 0.01:
        args.com_amount = 300
        args.batch_size = 50

#####################

# Model function
model_func = lambda : client_model(args.task.lower(), args)
# model_func = lambda : client_model("Resnet18", args)

init_model = model_func()

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