# Introduction

## What is **FedBug**?
**FedBug**, standing for **Fed**erated Learning with **B**ottom-**U**p **G**radual Unfreezing, a novel FL framework designed to effectively mitigate client drift.
FedBug adaptively leverages the client model parameters, distributed by the server at each global round, as the reference points for cross-client alignment. 

## What does FedBug work?

![alt text](/assets/alg_model.png)

FedBug works on the client side. It begins by freezing the entire model, then gradually unfreezes the layers, from the input layer to the output layer.
This bottom-up approach allows models to train the newly thawed layers to project data into a latent space, wherein the separating hyperplanes remain consistent across all clients. 

Take `FedBug (40\%)` for example, the first 40\% of training iterations perform gradual unfreezing (GU), while the remaining 60\% perform vanilla training. With the same number of training iterations, FedBug has fewer parameters to update and thus exhibits improved learning efficiency.

## What is the intuition behind FedBug?

Presumming some basic knowledge about fedeerated learnig and deep learning,  recall that 
1. At the start of each global round, all clients receive an identical model from the server;
2. Each intermediate layer parameterizes a set of hyperplanes that separate latent features, which is outputed by the previous layer.

Taken together, these insights suggest a strategy: By freezing the models received from the server, every client actually shares sets of hyperplans, parameterized by the frozen layers. By exploiting the frozen layers, cliens share common intermediate feature spaces. 

Below, we provide an example considering a `four-layer` model trained using `FedBug (40%)`.

Suppose we are in the `second GU period`, where all clients have just `unfrozen their second module`. During this period, the clients adapt their first and second modules and project the data into a feature space. Notably, the separating hyperplanes within this feature space are parameterized by the yet-to-be-unfrozen modules (the third and fourth modules in this case). These modules remain consistent during this period, serving as a shared anchor among clients. 
Similarly, as we progress to the subsequent third period, this process continues, with clients mapping their data into decision regions defined by the still-frozen fourth module. By leveraging the shared reference, FedBug ensures ongoing alignment among the clients.

## How does FedBug really work?

It is embarassingly simple. In terms of Pytorch Implementation, FedBug only changes the `requires_grad` attribute of a Tensor. 

# Experimental Setup
In this code, we assess the effectiveness of the FedBug algorithms withing three datasets (`CIFAR-10`, `CIFAR-100`, `Tiny-ImageNet`), five FL algorithms (`FedAvg`, `FedProx`, `FedDyn`, `FedExp`, `FedDecorr`), and various training conditions.

## Dataset Setup

For `CIFAR-10`, `CIFAR-100`, the data are downloaded automatically.

For `Tiny-ImageNet`, please follow the below steps [[1](https://github.com/bytedance/FedDecorr)]:
- Download the dataset to "data" directory from this link: 
http://cs231n.stanford.edu/tiny-imagenet-200.zip
- Unzip the downloaded file under "data" directory.
- Lastly, to reformat the validation set, under the folder "data/tiny-imagenet-200", run `python preprocess_tiny_imagenet.py`.

## Run Experiment

**Baseline**:
    
     python wk_run.py --mode 'fedavg' --task 'TinyImageNet'

**FedBug (40%)**:

     python wk_run.py --mode 'fedavg' --task 'TinyImageNet' --GUP1 0.4 --GUP2 "M"

`GUP1` sets the GU stage percentage; `GUP2` sets the use of ResNet Module `M` or residual block `B` as basic unit.