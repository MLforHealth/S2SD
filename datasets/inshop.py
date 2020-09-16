from datasets.basic_dataset_scaffold import BaseDataset
import os, numpy as np, pandas as pd


def Give(opt, datapath):
    data_info = np.array(pd.read_table(datapath+'/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[1:,:]
    train, query, gallery   = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
    train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
    query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
    gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

    train_image_dict    = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(datapath+'/'+img_path)

    query_image_dict    = {}
    for img_path, key in query:
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(datapath+'/'+img_path)

    gallery_image_dict    = {}
    for img_path, key in gallery:
        if not key in gallery_image_dict.keys():
            gallery_image_dict[key] = []
        gallery_image_dict[key].append(datapath+'/'+img_path)

    super_train_image_dict, counter, super_assign = {},0,{}
    for img_path, _ in train:
        key = '_'.join(img_path.split('/')[1:3])
        if key not in super_assign.keys():
            super_assign[key] = counter
            counter += 1
        key = super_assign[key]

        if not key in super_train_image_dict.keys():
            super_train_image_dict[key] = []
        super_train_image_dict[key].append(datapath+'/'+img_path)

    query_keys   = list(query_image_dict.keys())
    gallery_keys = list(gallery_image_dict.keys())

    if opt.train_val_split!=1:
        #NOTE: In In-Shop, training-validation split by class is generally disallowed due to classes having very low membernumbers!
        train_val_split = int(len(query_keys)*opt.train_val_split)
        train, val = query_keys[:train_val_split], query_keys[train_val_split:]
        query_train, gallery_train = train[:len(train)//2], train[len(train)//2:]
        query_val, gallery_val     = val[:len(val)//2], val[len(val)//2:]
        query_image_dict_train, query_image_dict_val     = {key:train_image_dict[key] for key in query_train},{key:train_image_dict[key] for key in query_val}
        gallery_image_dict_train, gallery_image_dict_val = {key:train_image_dict[key] for key in gallery_train},{key:train_image_dict[key] for key in gallery_val}
        query_dataset_val   = BaseDataset(query_image_dict_val, opt, is_validation=True)
        gallery_dataset_val = BaseDataset(gallery_image_dict_val, opt, is_validation=True)
    else:
        query_image_dict_train, gallery_image_dict_train = query_image_dict, gallery_image_dict
        query_dataset_val, gallery_dataset_val     = None, None

    train_dataset         = BaseDataset(train_image_dict, opt)
    super_train_dataset   = BaseDataset(super_train_image_dict, opt, is_validation=True)
    eval_dataset          = BaseDataset(train_image_dict, opt, is_validation=True)
    query_dataset_train   = BaseDataset(query_image_dict, opt, is_validation=True)
    gallery_dataset_train = BaseDataset(gallery_image_dict, opt, is_validation=True)

    return {'training':train_dataset, 'testing_query':query_dataset_train, 'evaluation':eval_dataset,
            'validation_query':query_dataset_val, 'validation_gallery':gallery_dataset_val,
            'testing_gallery':gallery_dataset_train, 'super_evaluation':super_train_dataset}
