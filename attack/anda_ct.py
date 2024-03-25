# coding=utf-8
import math
import torch
import os.path
import argparse
from utils import *
import torchvision.models as models
from torch.nn import functional as F
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='attack in PyTorch')
parser.add_argument('--batch_size', type=int, default=10, help='mini-batch size (default: 1)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--max_epsilon', default=16.0, type=float, help='max magnitude of adversarial perturbations')
parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
parser.add_argument('--n_ens', default=25, type=int, help='augmentation number')
parser.add_argument('--aug_max', default=0.3, type=float, help='augmentation degree of the attack')
parser.add_argument('--input_csv', default='./datasets/dev.csv', type=str, help='csv info of clean examples')
parser.add_argument('--input_dir', default='./datasets/images/', type=str, help='directory of clean examples')
parser.add_argument('--output_dir', default='', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--victim_model', default='vgg19', type=str, help='directory for test')
parser.add_argument('--device', default='2', type=str, help='gpu device')

parser.add_argument('--kernel_size', default=7, type=int, help='kernel_size')
parser.add_argument('--sigma', default=3, type=int, help='sigma of kernel')
parser.add_argument('--kernel_name', default='gaussian', type=str)
parser.add_argument('--m', default=5, type=int, help='the number of scale copies')

args = parser.parse_args()


def attack(x, y, model, num_iter, eps, alpha, minibatch=False, sample=False):
    # enable minibatch to save CUDA memory (change mini_batchsize below if necessary)
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()
    n_ens = thetas.shape[0]
    min_x = x - eps
    max_x = x + eps

    xt = x.clone()

    anda = ANDA(data_shape=(1, 3, 299, 299), device=torch.device('cuda'))

    with torch.enable_grad():
        for i in range(num_iter):
            if not minibatch:
                xt_batch = xt.repeat(n_ens, 1, 1, 1)
                xt_batch.requires_grad = True
                aug_xt_batch = translation(thetas, xt_batch)
                # sifgsm
                si_xts = scale_transform(aug_xt_batch, m=args.m)
                # difgsm
                di_xts = input_diversity(si_xts, resize=330, diversity_prob=0.5)
                ys = y.repeat(di_xts.shape[0])
                outputs = model(di_xts)
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(0)
                loss = F.cross_entropy(outputs, ys, reduction="sum")
                loss.backward()
                new_grad = xt_batch.grad
            else:
                xt_batch = xt.repeat(n_ens, 1, 1, 1)
                xt_batch.requires_grad = True
                aug_xts = translation(thetas, xt_batch)
                # sifgsm
                si_xts = scale_transform(aug_xts, m=args.m)
                # difgsm
                di_xts = input_diversity(si_xts, resize=330, diversity_prob=0.5)
                ys = y.repeat(di_xts.shape[0])
                new_grad = xt_batch.new_zeros(xt_batch.shape)
                mini_batchsize = 10 # change minibatch here to fit your device
                for xt_tmp, yt_tmp in get_minibatch(di_xts, ys, min(mini_batchsize, args.m)):
                    output = model(xt_tmp)
                    if output.ndim == 1:
                        output = output.unsqueeze(0)
                    loss = F.cross_entropy(output, yt_tmp, reduction="sum")
                    loss.backward(retain_graph=True)
                    new_grad = new_grad + xt_batch.grad
                    xt_batch.grad.zero_()

            anda.collect_model(new_grad)
            sample_noise = anda.noise_mean                

            # tifgsm
            if sample and i == num_iter - 1:
                sample_noises = anda.sample(n_sample=1, scale=1)
                sample_noises = F.conv2d(sample_noises, gaussian_kernel, stride=1, padding='same', groups=3)
                sample_xt = alpha * sample_noises.squeeze().sign() + xt
                sample_xt = torch.clamp(sample_xt, 0.0, 1.0).detach()
                sample_xt = torch.max(torch.min(sample_xt, max_x), min_x).detach()
            
            
            sample_noise = F.conv2d(sample_noise, gaussian_kernel, stride=1, padding='same', groups=3)

            xt = xt + alpha * sample_noise.sign()

            xt = torch.clamp(xt, 0.0, 1.0).detach()
            xt = torch.max(torch.min(xt, max_x), min_x).detach()

    if sample:
        adv = sample_xt.detach().clone()
    else:
        adv = xt.detach().clone()

    with torch.no_grad():
        output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    pred_top5 = output.topk(k=5, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()
    return adv, (pred_top1 == y).sum().item(), \
           (pred_top5 == y.unsqueeze(dim=1).expand(-1, 5)).sum().item()


def main():
    model_name = args.victim_model
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    model = torch.nn.Sequential(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                getattr(models, model_name)(pretrained=True).eval()).cuda()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    clean_dataset = NIPS_GAME(args.input_dir, args.input_csv, preprocess)
    no_samples = len(clean_dataset)
    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    correct_1 = 0
    correct_5 = 0

    kwargs = {
        "num_iter": args.num_iter,
        "eps": args.max_epsilon / 255,
        "alpha": args.max_epsilon / 255 / args.num_iter,
    }
    
    for i, (x, name, y, _) in enumerate(clean_loader):
        x = x.cuda()
        y = y.cuda()
        
        adv_x, corr_1, corr_5 = attack(x, y, model, **kwargs)

        correct_1 += corr_1
        correct_5 += corr_5
        for k in range(adv_x.shape[0]):
            save_img(os.path.join(output_dir, name[k]), adv_x[k].detach().permute(1, 2, 0).cpu())
        print('attack in process, i = %d, top1 = %.3f, top5 = %.3f' % (i, corr_1 / args.batch_size, corr_5 / args.batch_size))

    print('attack finished')
    print('top1 = %.3f, top5 = %.3f' % (correct_1 / no_samples, correct_5 / no_samples))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    assert is_sqr(args.n_ens)
    thetas = get_thetas(int(math.sqrt(args.n_ens)), -args.aug_max, args.aug_max)
    ti_kernel = Translation_Kernel(len_kernel=args.kernel_size, nsig=args.sigma, kernel_name=args.kernel_name)
    gaussian_kernel = torch.from_numpy(ti_kernel.kernel_generation()).cuda()
    main()