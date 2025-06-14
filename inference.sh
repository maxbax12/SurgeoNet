python3 inference.py \
    --data_root data/varjo_stereo/ \
    --sequence_name objects_4_9_13 \
    --resolution 1152 \
    --model_size s \
    --device cpu \
    --model_extension pt \
    --fix_kps \
    --use_transformer \
    --save \
    --visualize \

# add options to visualize frames and or save them