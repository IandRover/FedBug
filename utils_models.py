from utils_libs import *
import torchvision.models as models

class client_model(nn.Module):
    def __init__(self, name, args):
        super(client_model, self).__init__()
        self.name = name
        self.model_name = args.model_name

        if self.name == "cifar10": self.n_cls = 10
        if self.name == "cifar100": self.n_cls = 100
        if self.name == "tinyimagenet": self.n_cls = 200   
        
        if self.model_name in ["cnn"] and self.name in ['cifar10', 'cifar100']:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.model_name in ["cnn"] and self.name in ['tinyimagenet']:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*4*4, 256) 
            self.fc2 = nn.Linear(256, 128) 
            self.fc3 = nn.Linear(128, self.n_cls)

        if self.model_name == "resnet": self.model = resnet18(self.n_cls)

        if self.model_name == "resnet34": self.model = resnet34(self.n_cls)

        elif self.model_name == "mobilenetv2": self.model = mobilenetv2(self.n_cls)
              
    def forward(self, x):
        
        if self.model_name in ["resnet", "mobilenetv2", "resnet34"]:
            x = self.model(x)

        elif self.model_name in ["cnn"] and self.name in ['cifar10','cifar100']:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.model_name in ["cnn"] and self.name in ['tinyimagenet']:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64*4*4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        return x

    def forward_feat(self, x):
        
        if self.name in ['tinyimagenet']:

            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64*4*4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

        return self.fc3(x), x


# +
def mobilenetv2(n_cls):
    def replace_batchnorm(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                model._modules[name] = replace_batchnorm(module)
            if isinstance(module, torch.nn.BatchNorm2d):
                model._modules[name] = torch.nn.GroupNorm(2, module.num_features)
        return model 
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(1280, n_cls)
    model = replace_batchnorm(model)
    return model

def resnet34(n_cls):
    def replace_batchnorm(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                model._modules[name] = replace_batchnorm(module)
            if isinstance(module, torch.nn.BatchNorm2d):
                model._modules[name] = torch.nn.GroupNorm(2, module.num_features)
        return model 
    model = models.resnet34()
    model.fc = nn.Linear(512, n_cls)
    model = replace_batchnorm(model)
    return model

def resnet18(n_cls):
    resnet18 = models.resnet18()
    resnet18.fc = nn.Linear(512, n_cls)

    # Change BN to GN 
    resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

    resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
    resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
    resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
    resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

    resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
    resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
    resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
    resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
    resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

    resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
    resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
    resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
    resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
    resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

    resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
    resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
    resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
    resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
    resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

    assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'

    return resnet18
