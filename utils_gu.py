import torch

def __SetGrad(args, tensor, sign):
    if args.model_name in ["cnn"]:
        tensor.weight.requires_grad = sign
        tensor.bias.requires_grad = sign
    elif args.model_name in ["resnet18", "resnet34"]:
        for p1, p2 in tensor.named_parameters():
            p2.requires_grad = sign

def StopGradScheduler(args, model):
    for p1 , p2 in model.named_parameters(): p2.requires_grad = True

    if args.GU == 0: return 

    if args.model_name == "resnet34":
        if int(args.GU) == 104:
            factor = (args.GU*1000)%1000
            if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.layer2, False)
            if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.layer3, False)
            if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.layer4, False)
            if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.fc, False)
            return
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

    if args.model_name == "resnet18":
        if int(args.GU) == 104:
            print("GU104")
            factor = (args.GU*1000)%1000
            if  args.local_iter_count // factor <= 0: __SetGrad(args, model.model.layer2, False)
            if  args.local_iter_count // factor <= 1: __SetGrad(args, model.model.layer3, False)
            if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.layer4, False)
            if  args.local_iter_count // factor <= 2: __SetGrad(args, model.model.fc, False)
            return 
        elif int(args.GU) == 108:
            print("GU108")
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

    if args.model_name == "cnn":
        if int(args.GU) == 100:
            if args.model_name == "cnn" and args.task in ["CIFAR10", "CIFAR100"]:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.conv2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.fc1, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.fc2, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.fc3, False)
                return 
            if args.model_name == "cnn" and args.task in ["TinyImageNet"]:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.conv2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.conv3, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.fc1, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.fc2, False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.fc3, False)
                return 
        if int(args.GU) == 200:     
            if args.model_name == "cnn" and args.task in ["TinyImageNet"]:
                factor = (args.GU*1000)%1000
                if  args.local_iter_count // factor <= 0: __SetGrad(args, model.fc2, False)
                if  args.local_iter_count // factor <= 1: __SetGrad(args, model.fc1, False)
                if  args.local_iter_count // factor <= 2: __SetGrad(args, model.conv3, False)
                if  args.local_iter_count // factor <= 3: __SetGrad(args, model.conv2, False)
                if  args.local_iter_count // factor <= 4: __SetGrad(args, model.conv1, False)
                return 
        if int(args.GU) == 301:
            if args.model_name == "cnn":
                __SetGrad(args, model.fc3, False)
                return 
        if int(args.GU) == 302:
            if args.model_name == "cnn":
                __SetGrad(args, model.fc3, False)
                __SetGrad(args, model.fc2, False)
                return 
            
    assert 0 ==1, "Mismatch between model type, dataset, and GU strategy."
