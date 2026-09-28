dataset=$1
workspace=$2
gpu_id=$3

audio_extractor='ave'

pretrain_project_path="output/pretrain_ave/"

pretrain_face_path=${pretrain_project_path}/chkpnt_ema_face_latest.pth
pretrain_mouth_path=${pretrain_project_path}/chkpnt_ema_mouth_latest.pth

# 5 seconds
n_views=125

export CUDA_VISIBLE_DEVICES=$gpu_id

VPR_ARGS="\
--use_viseme_residual \
--num_visemes 9 \
--viseme_rank 8 \
--viseme_res_scale 1.0 \
--viseme_reg 0.0001"

python train_face.py \
    --type face \
    -s $dataset \
    -m $workspace \
    --init_num 2000 \
    --densify_grad_threshold 0.0005 \
    --audio_extractor $audio_extractor \
    --pretrain_path $pretrain_face_path \
    --iterations 10000 \
    --sh_degree 1 \
    --N_views $n_views \
    $VPR_ARGS

python train_mouth.py \
    --type mouth \
    -s $dataset \
    -m $workspace \
    --audio_extractor $audio_extractor \
    --pretrain_path $pretrain_mouth_path \
    --init_num 5000 \
    --iterations 10000 \
    --sh_degree 1 \
    --N_views $n_views \
    $VPR_ARGS

python train_fuse_con.py \
    -s $dataset \
    -m $workspace \
    --opacity_lr 0.001 \
    --audio_extractor $audio_extractor \
    --iterations 2000 \
    --sh_degree 1 \
    --N_views $n_views \
    $VPR_ARGS

python synthesize_fuse.py \
    -s $dataset \
    -m $workspace \
    --eval \
    --audio_extractor $audio_extractor \
    --dilate \
    $VPR_ARGS

python metrics.py \
    $workspace/test/ours_None/renders/out.mp4 \
    $workspace/test/ours_None/gt/out.mp4