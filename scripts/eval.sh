# simply change input_dir to adv-dir to evaluate advs
# for example, evaluating adv samples, --input_dir outputs/anda/vgg19
# for example, evaluating clean samples, --input_dir datasets/images

python -m attack.eval \
    --batch_size 10 \
    --workers 4 \
    --input_csv datasets/dev_dataset.csv \
    --input_dir outputs/anda/vgg19
    # --input_dir datasets/images