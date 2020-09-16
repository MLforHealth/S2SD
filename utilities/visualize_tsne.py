from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
# import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
import random

import parameters    as par
import datasampler   as dsamplers
import datasets      as datasets
import architectures as archs


def tSNE(opt, dataloader, model, n_samples=1000, perplex=40.0):

    #Compute features
    _ = model.eval()
    with torch.no_grad():
        collect_features = list()
        collect_labels = list()
        final_iter  = tqdm(dataloader, desc='Computing Embeddings')
        for i,out in enumerate(final_iter):
            class_labels, input, input_indices = out

            features = model(input.to(opt.device))
            if isinstance(features, tuple): features = features[0]

            collect_features.extend(features.cpu().detach().numpy().tolist())
            collect_labels.extend(class_labels.cpu().numpy().tolist())

        features = np.vstack(collect_features).astype('float32')
        labels = np.asarray(collect_labels)

    # choose subset
    idx2use = np.random.choice(features.shape[0], size=n_samples, replace=False)
    features = features[idx2use, :]
    labels = labels[idx2use]

    # create save dir
    save_path_base = opt.save_path + f'/tSNE/test'
    if not os.path.isdir(save_path_base):
        os.makedirs(save_path_base)

    ### SCATTER PLOT
    save_path = save_path_base + f'/scatter_p{perplex}.svg'
    visualize_scatter(features, labels, perplex, save_path=save_path)

def tSNE_test_train(opt, dataloaders, model, n_samples=1000, perplex=40.0):

    #Compute features
    _ = model.eval()
    collect_features = list()
    collect_labels = list()
    label = 0
    with torch.no_grad():
        for name, dl in dataloaders.items():
            dl_iter  = tqdm(dl, desc='Computing Embeddings')
            for i,out in enumerate(dl_iter):
                class_labels, input, input_indices = out

                features = model(input.to(opt.device))
                if isinstance(features, tuple): features = features[0]

                collect_features.extend(features.cpu().detach().numpy().tolist())

                labels_tmp = np.ones(features.shape[0]) * label
                collect_labels.extend(labels_tmp.tolist())

            features = np.vstack(collect_features).astype('float32')
            labels = np.asarray(collect_labels)

            label += 1

    # choose subset
    idx2use = np.random.choice(features.shape[0], size=n_samples, replace=False)
    features = features[idx2use, :]
    labels = labels[idx2use]

    # create save dir
    save_path_base = opt.save_path + f'/tSNE/test'
    if not os.path.isdir(save_path_base):
        os.makedirs(save_path_base)

    ### SCATTER PLOT
    save_path = save_path_base + f'/scatter_p{perplex}.svg'
    visualize_scatter(features, labels, perplex, save_path=save_path)

def visualize_scatter(features, labels, perplex, figsize=(10,10), save_path=None):

    # compute tSNE
    tsne = TSNE(n_components=2, perplexity=perplex)
    tsne_result = tsne.fit_transform(features)

    # plot
    cmap = plt.cm.get_cmap('jet', len(np.unique(labels)))
    fig, ax = plt.subplots(figsize=figsize)
    for id in range(len(labels)):
        plt.scatter(tsne_result[id, 0],  tsne_result[id, 1],
                    marker='o',
                    color=cmap(labels[id] % (len(np.unique(labels)) + 1)),
                    linewidth='1',
                    alpha=0.8)
    ax.update_datalim(tsne_result)
    ax.autoscale()
    plt.show()

    # save plot
    # fig.savefig(save_path, dpi=1000)
    plt.close()


if __name__ == "__main__":
    ################### INPUT ARGUMENTS ###################
    parser = argparse.ArgumentParser()

    parser = par.basic_training_parameters(parser)
    parser = par.batch_creation_parameters(parser)
    parser = par.batchmining_specific_parameters(parser)
    parser = par.loss_specific_parameters(parser)
    parser = par.wandb_parameters(parser)

    ##### Read in parameters
    opt = parser.parse_args()

    """==================================================================================================="""
    opt.source_path += '/' + opt.dataset
    opt.save_path += '/' + opt.dataset

    # Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
    assert not opt.bs % opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

    opt.pretrained = not opt.not_pretrained

    """==================================================================================================="""
    ################### GPU SETTINGS ###########################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # if not opt.use_data_parallel:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])

    """==================================================================================================="""
    #################### SEEDS FOR REPROD. #####################
    torch.backends.cudnn.deterministic = True;
    np.random.seed(opt.seed);
    random.seed(opt.seed)
    torch.manual_seed(opt.seed);
    torch.cuda.manual_seed(opt.seed);
    torch.cuda.manual_seed_all(opt.seed)

    """==================================================================================================="""
    ##################### NETWORK SETUP ##################
    # NOTE: Networks that can be used: 'bninception, resnet50, resnet101, alexnet...'
    # >>>>  see import pretrainedmodels; pretrainedmodels.model_names
    opt.device = torch.device('cuda')
    model = archs.select(opt.arch, opt)

    # regularized (0.6852)
    # path_weights = f'/export/home/tmilbich/PycharmProjects/Deep_Metric_Learning_Research_PyTorch/Training_Results/cub200/MixManifold_interpolate_ProxyAnchor_Eucl_s0_20/checkpoint_Test_discriminative_e_recall@1.pth.tar'
    # baseline (0.6632)
    path_weights = f'/export/home/tmilbich/PycharmProjects/Deep_Metric_Learning_Research_PyTorch/Training_Results/cub200/MixManifold_interpolate_ProxyAnchor_Eucl_s0/checkpoint_Test_discriminative_e_recall@1.pth.tar'
    model.load_state_dict(torch.load(path_weights)['state_dict'])
    model.to(opt.device)
    model.eval()

    """============================================================================"""
    #################### DATALOADER SETUPS ##################
    dataloaders = {}
    datasets = datasets.select(opt.dataset, opt, opt.source_path)
    train_data_sampler = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict,
                                          datasets['training'].image_list)
    dataloaders['evaluation'] = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels,
                                                            batch_size=opt.bs, shuffle=False)
    dataloaders['testing'] = torch.utils.data.DataLoader(datasets['testing'], num_workers=opt.kernels,
                                                         batch_size=opt.bs, shuffle=False)

    # visualize tsngise
    perp = 20.0
    n = 3000
    print(f'Performing TSNE (p={perp}, n={n}):')
    print(f'Dataset: {opt.dataset}')
    print(f'Weights: {path_weights}')

    # tSNE(opt, dataloaders['testing'], model, n_samples=n, perplex=perp)
    tSNE_test_train(opt, dataloaders, model, n_samples=n, perplex=perp)