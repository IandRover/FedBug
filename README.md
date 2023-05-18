# FedBug

FedBUG: Federated Learning with Bottom-Up Gradual Unfreezing

## Intuition and Basic Ideas

FedBug, standing for Federated Learning with Bottom-Up Gradual Unfreezing, a novel FL framework designed to effectively mitigate client drift.
FedBug adaptively leverages the client model parameters, distributed by the server at each global round, as the reference points for cross-client alignment. 
Specifically, on the client side, FedBug begins by freezing the entire model, then gradually unfreezes the layers, from the input layer to the output layer.
This bottom-up approach allows models to train the newly thawed layers to project data into a latent space, wherein the separating hyperplanes remain consistent across all clients. 

## Experimental Setup
In this code, we assess the effectiveness of the FedBug algorithms withing three datasets (`CIFAR-10`, `CIFAR-100`, `TIny-ImageNet`), five FL algorithms (`FedAvg`, `FedProx`, `FedDyn`, `FedExp`, `FedDecorr`), and various training conditions.

## Dataset Setup

For `CIFAR-10`, `CIFAR-100`, the data are downloaded automatically.
For `Tiny-Imagenet`, please follow the below steps [[1](https://github.com/bytedance/FedDecorr)]:
- Download the dataset to "data" directory from this link: 
http://cs231n.stanford.edu/tiny-imagenet-200.zip
- Unzip the downloaded file under "data" directory.
- Lastly, to reformat the validation set, under the folder "data/tiny-imagenet-200", run `python3 preprocess_tiny_imagenet.py`.

## Run Experiment


