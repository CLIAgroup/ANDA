# ["resnet50", "densenet121", "resnet101", "vgg19", "densenet169", "inception_v3"]

model=vgg19
python -m attack.multianda_ct  --batch_size 1 \
    --input_csv datasets/dev_dataset.csv \
    --input_dir datasets/images \
    --output_dir "outputs/multianda_ct" \
    --n_ens 25 --num_iter 10 --nproc 5 \
    --victim_model ${model} \
    --device '0,1,2,3,4'

python -m attack.multianda  --batch_size 1 \
    --input_csv datasets/dev_dataset.csv \
    --input_dir datasets/images \
    --output_dir "outputs/multianda" \
    --n_ens 25 --num_iter 10 --nproc 5 \
    --victim_model ${model} \
    --device '0,1,2,3,4'
