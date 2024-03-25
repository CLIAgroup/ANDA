# coding=utf-8
import argparse
import os.path
from utils import NIPS_GAME
import torch
import torchvision.models as models
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='eval attack in PyTorch')
parser.add_argument('--batch_size', type=int, default=10, help='mini-batch size (default: 1)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--input_csv', default='./datasets/dev.csv', type=str, help='csv info of clean examples')
parser.add_argument('--input_dir', default='./datasets/images/', type=str, help='directory of clean examples')
parser.add_argument('--device', default='2', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='use gpu or not')

args = parser.parse_args()
    
def main():
    # model_names = ["resnet50", "densenet121", "resnet101", "vgg19", "densenet169", "inception_v3"]
    model_names = ["vgg19"]
    res_dict = {model:{"top1":0, "top5":0} for model in model_names}
    model_dict = {model:torch.nn.Sequential(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                getattr(models, model)(pretrained=True).eval()).cuda() for model in model_names}
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    clean_dataset = NIPS_GAME(args.input_dir, args.input_csv, preprocess)
    no_samples = len(clean_dataset)
    clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    with torch.no_grad():
        for i, (x, name, y, _) in enumerate(clean_loader):
            if args.use_gpu:
                x = x.cuda()
                y = y.cuda()
            
            for model_name, model in model_dict.items():
                output = model(x)
                pred_top1 = output.topk(k=1, largest=True).indices
                pred_top5 = output.topk(k=5, largest=True).indices
                if pred_top1.dim() >= 2:
                    pred_top1 = pred_top1.squeeze()

                corr_1 = (pred_top1 == y).sum().item()
                corr_5 = (pred_top5 == y.unsqueeze(dim=1).expand(-1, 5)).sum().item()

                res_dict[model_name]['top1'] += corr_1
                res_dict[model_name]['top5'] += corr_5

    
    for model_name, res in res_dict.items():
        print('%s: top1=%.3f, top5=%.3f' % (model_name, res['top1'] / no_samples, res['top5'] / no_samples))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()