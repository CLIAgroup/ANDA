# ["resnet50", "densenet121", "resnet101", "vgg19", "densenet169", "inception_v3"]

model=vgg19
python -m attack.anda_ct --batch_size 1 --device 0 \
    --input_csv datasets/dev_dataset.csv --input_dir datasets/images --output_dir "outputs/anda_ct" \
    --n_ens 25 --num_iter 10 --victim_model ${model}

python -m attack.anda  --batch_size 1 --device 3 \
    --input_csv datasets/dev_dataset.csv --input_dir datasets/images --output_dir "outputs/anda" \
    --n_ens 25 --num_iter 10 --victim_model ${model}