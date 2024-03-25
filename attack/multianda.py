# coding=utf-8
import argparse
import os.path
import os
from utils import *
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import functional as F
import math

parser = argparse.ArgumentParser(description='attack in PyTorch')
parser.add_argument('--batch_size', type=int, default=10, help='mini-batch size (default: 1)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--max_epsilon', default=16.0, type=float, help='max magnitude of adversarial perturbations')
parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
parser.add_argument('--n_ens', default=25, type=int, help='max iteration')
parser.add_argument('--aug_max', default=0.3, type=float, help='param of the attack')
parser.add_argument('--input_csv', default='./data/input_dir/dev.csv', type=str, help='csv info of clean examples')
parser.add_argument('--input_dir', default='./data/input_dir/images/', type=str, help='directory of clean examples')
parser.add_argument('--output_dir', default='./data/adv/anda', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--victim_model', default='inceptionv3', type=str, help='directory for test')
parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--nproc', default=3, type=int)

args = parser.parse_args()

def attack(x, y, model, num_iter, eps, alpha, thetas, sample=False):
    min_x = x - eps
    max_x = x + eps

    n_ens = thetas.shape[0]
    xt = x.clone()

    anda = ANDA(data_shape=(1, 3, 299, 299), device=x.device)
    last_xt = None
    with torch.enable_grad():
        for i in range(num_iter):
            xt_batch = xt.repeat(n_ens, 1, 1, 1)
            # random init for multianda
            batch_noise = 2 * (torch.rand_like(xt_batch).to(xt_batch.device) - 0.5) * 1 / 255. / 2 # add uniform noise with bound [-0.5/255, 0.5/255]
            xt_batch = xt_batch + batch_noise
            xt_batch.requires_grad = True
            aug_xt_batch = translation(thetas, xt_batch)
            ys = y.repeat(xt_batch.shape[0])
            outputs = model(aug_xt_batch)
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            loss = F.cross_entropy(outputs, ys, reduction="sum")
            loss.backward()
            new_grad = xt_batch.grad

            anda.collect_model(new_grad)
            sample_noise = anda.noise_mean

            if i == num_iter - 1:
                last_xt = xt.detach().clone()

            xt = xt + alpha * sample_noise.sign()
            xt = torch.clamp(xt, 0.0, 1.0).detach()
            xt = torch.max(torch.min(xt, max_x), min_x).detach()

    size = float(dist.get_world_size())
    dist.all_reduce(last_xt.data, op=dist.ReduceOp.SUM)
    last_xt.data /= size
    
    if sample:
        # todo: sample (option b)
        sample_noise = anda.sample(n_sample=1, scale=1)
        dist.all_reduce(sample_noise.data, op=dist.ReduceOp.SUM)
        sample_noise.data /= size

        samples_xt = last_xt + alpha * sample_noise.squeeze().sign()
        samples_xt = torch.clamp(samples_xt, 0.0, 1.0).detach()
        adv = torch.max(torch.min(samples_xt, max_x), min_x).detach().clone()
    else:
        # todo: baseline (option a)   
        dist.all_reduce(anda.noise_mean.data, op=dist.ReduceOp.SUM)
        anda.noise_mean.data /= size
        baseline_noise = anda.noise_mean
        xt = last_xt + alpha * baseline_noise.sign()
        xt = torch.clamp(xt, 0.0, 1.0).detach()
        adv = torch.max(torch.min(xt, max_x), min_x).detach().clone()

    with torch.no_grad():
        output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    pred_top5 = output.topk(k=5, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()

    return adv, (pred_top1 == y).sum().item(), \
           (pred_top5 == y.unsqueeze(dim=1).expand(-1, 5)).sum().item()

def run(rank, clean_loader, model, output_dir, kwargs):
    device = torch.device("cuda:{}".format(rank))
    no_samples = len(clean_loader.dataset)
    correct_1 = 0
    correct_5 = 0
    model = model.to(device)
    for i, (x, name, y, target) in enumerate(clean_loader):
        assert x.shape[0] == 1
        save_path = os.path.join(output_dir, name[0])
        if os.path.exists(save_path):
            print(f'{save_path} already exists!')
            continue

        x, y = x.to(device), y.to(device)
        adv_x, corr_1, corr_5 = attack(x, y, model, **kwargs) 
        if rank == 0:
            correct_1 += corr_1
            correct_5 += corr_5
            save_img(os.path.join(output_dir, name[0]), adv_x[0].detach().cpu().permute(1, 2, 0))
            print('attack in process, i = %d, top1 = %.3f, top5 = %.3f' % (i, corr_1 / args.batch_size, corr_5 / args.batch_size))

    if rank == 0:
        print('attack finished')
        print('memory image attack: top1 = %.3f, top5 = %.3f' % (correct_1 / no_samples, correct_5 / no_samples))
        

def init_process(rank, size, clean_loader, model, output_dir, kwargs, run, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(rank, clean_loader, model, output_dir, kwargs)


def main():
    model_name = args.victim_model
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    assert model_name in ["resnet50", "densenet121", "resnet101", "vgg19", "densenet169", "inception_v3"]
    model = torch.nn.Sequential(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                getattr(models, model_name)(pretrained=True).eval())

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    clean_dataset = NIPS_GAME(args.input_dir, args.input_csv, preprocess)
    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    thetas = get_thetas(int(math.sqrt(args.n_ens)), -args.aug_max, args.aug_max)
    
    kwargs = {
        "num_iter": args.num_iter,
        "eps": args.max_epsilon / 255,
        "alpha": args.max_epsilon / 255 / args.num_iter,
        "thetas": thetas,
    }

    size = args.nproc
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, clean_loader, model, output_dir, kwargs, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    mp.set_start_method("spawn")
    print(args)
    assert is_sqr(args.n_ens)
    main()