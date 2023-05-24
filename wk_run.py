from utils_general import *
from utils_methods import *  
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = "fedavg", 
                    choices=["feddyn", "scaffold", "fedavg", "fedprox",
                             "fedexp", "fedcm", "feddecorr",
                             "fedadam", "fedadagrad", "fedavgm"])
parser.add_argument('--epoch', type = int, default = 5)
parser.add_argument('--model_name', type = str, default = "cnn", choices=["cnn", "resnet18", "resnet34"])

parser.add_argument('--task', type = str, default = "CIFAR100", choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
parser.add_argument('--balance', type = float, default = 0., help="0: balanced | 0-1: sampling rate")
parser.add_argument('--dist', type = float, default = 0.0, help="0: iid | 0-1: non-iid with unbalanced_sgm")

parser.add_argument('--n_clients', type = int, default = 100)
parser.add_argument('--act_prob', type = float, default = .01)

parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--momentum', type = float, default = 0)
parser.add_argument('--gu_ratio', type = float, default = 0.)
parser.add_argument('--gu_unit', type = str, default = "L")

parser.add_argument('--device_idx', type = int, default = 0)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--extension', type = str, default = "")
args, _ = parser.parse_known_args()

# ==============================================
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
if torch.cuda.device_count() == 1: args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else: args.device = torch.device("cuda:{}".format(args.device_idx) if torch.cuda.is_available() else "cpu")
print("Using GPU of Device ", args.device, " of ", torch.cuda.device_count())
# ==============================================
if args.task == "TinyImageNet":
    args.epoch = 3
    if args.n_clients == 10: args.com_amount = 100
if args.task == "CIFAR100":
    args.epoch = 5
    if args.n_clients == 10: args.com_amount = 100
    if args.n_clients == 100: args.com_amount = 300
if args.task == "CIFAR10":
    args.epoch = 10
    if args.n_clients == 10: args.com_amount = 100
    if args.n_clients == 100: args.com_amount = 500
# ==============================================
args.bs = 50
args.weight_decay = 1e-3
args.extension += "_EP" + str(args.epoch)
if args.balance == 0.: args.balance = 0
if args.dist == 0.: args.dist = 0
args.extension += "_LR" + str(args.lr)
if args.momentum != 0:
    args.extension += "_Mom" + str(int(args.momentum * 10))
if args.gu_ratio == 0: args.gu_ratio = 0
if args.gu_ratio != 0:
    if int(args.gu_ratio) == args.gu_ratio: args.gu_ratio = int(args.gu_ratio)
    args.extension += "_gu" + str(args.gu_unit) + str(args.gu_ratio)
# ==============================================
dir_folder = args.task + "-" + args.model_name
if not os.path.exists(dir_folder): os.mkdir(dir_folder)
args.savepath = dir_folder + '/{}-B{}-D{}-N{}-P{}_{}'.format(args.mode, args.balance, args.dist, args.n_clients, args.act_prob, args.extension)
if not os.path.exists(args.savepath): os.mkdir(args.savepath)
args.savename = args.savepath+'/s{}.csv'.format(args.seed)
print("Save Logs at: ", args.savename), print()

if os.path.exists(args.savename): 
    print("The seed already exists.")
    print("The path is ", args.savename)
    print()
    # time.sleep(0.1)
    # import sys
    # sys.exit()

# ==============================================
model_func = lambda : client_model(args.task.lower(), args)
init_model = model_func()
if args.dist == 0.: data_obj = DatasetObject(dataset=args.task, n_client=args.n_clients, rule="iid", unbalanced_sgm=0)
else: data_obj = DatasetObject(dataset=args.task, n_client=args.n_clients, unbalanced_sgm=0, 
                             rule='Dirichlet', rule_arg=args.dist)
# ==============================================
if args.mode == "feddyn":
    args.alpha_coef = 1e-2
elif args.mode == "scaffold":
    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / args.n_clients
    n_iter_per_epoch  = np.ceil(n_data_per_client/args.bs)
    args.n_minibatch = (args.epoch*n_iter_per_epoch).astype(np.int64)
elif args.mode == "fedprox":
    args.mu = 1e-4
elif args.mode in ["fedadam", "fedadagrad", "fedadagrad1", "fedadagrad2", "fedavgm"]:
    args.b1 = 0.9
    args.b2 = 0.99
# ==============================================
train(args=args, data_obj=data_obj, model_func=model_func, init_model=init_model)
