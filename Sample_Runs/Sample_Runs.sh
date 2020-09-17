###### NOTE: These runs don't use learning rate scheduling. Depending on dataset and setup, simply include --tau x y --gamma z.

"""==== CUB200-2011  ===="""
#ResNet50 - 128
python main.py --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group R50-128_cub200_msim_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 128 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr
#ResNet50 - 512
python main.py --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group R50-512_cub200_msim_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 512 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr
#Inception-BN - 512
python main.py --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group IBN-512_cub200_msim_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch bninception_normalize --embed_dim 512 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr
#Inception-BN - 512 with frozen Batchnorm
python main.py --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group IBNf-512_cub200_msim_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch bninception_normalize --embed_dim 512 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr


"""==== CARS196  ===="""
#ResNet50 - 128
python main.py --dataset cars196 --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group R50-128_cars196_msim_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 128 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr
#ResNet50 - 512
python main.py --dataset cars196 --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group R50-512_cars196_msim_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 512 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr
#Inception-BN - 512
python main.py --dataset cars196 --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group IBN-512_cars196_msim_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch bninception_normalize --embed_dim 512 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr
#Inception-BN - 512 with frozen Batchnorm
python main.py --dataset cars196 --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group IBNf-512_cars196_msim_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch bninception_normalize --embed_dim 512 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr


"""==== Stanford Online Products  ===="""
#ResNet50 - 128
python main.py --dataset online_products --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group R50-128_sop_margin_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 128 --loss_s2sd_source margin --loss_s2sd_target margin --batch_mining distance --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 5 --loss_s2sd_feat_distill_delay 1000
#ResNet50 - 512
python main.py --dataset online_products --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group R50-512_sop_margin_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch resnet50_frozen_normalize --embed_dim 512 --loss_s2sd_source margin --loss_s2sd_target margin --batch_mining distance --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 5 --loss_s2sd_feat_distill_delay 1000
#Inception-BN - 512
python main.py --dataset online_products --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group IBN-512_sop_margin_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch bninception_normalize --embed_dim 512 --loss_s2sd_source margin --loss_s2sd_target margin --batch_mining distance --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 5 --loss_s2sd_feat_distill_delay 1000
#Inception-BN - 512 with frozen Batchnorm
python main.py --dataset online_products --source $datapath --n_epochs 150 --log_online --project Sample_Runs --group IBNf-512_sop_margin_s2sd --seed 0 --gpu $gpu --bs 112 --loss s2sd --arch bninception_normalize --embed_dim 512 --loss_s2sd_source margin --loss_s2sd_target margin --batch_mining distance --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 5 --loss_s2sd_feat_distill_delay 1000
