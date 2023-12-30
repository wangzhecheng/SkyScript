import torch
import numpy as np
from PIL import Image
import os
import sys
import pandas as pd
import pickle
from os.path import join, exists
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.open_clip.model import get_cast_dtype, trace_model
from src.open_clip.factory import create_model_and_transforms
from src.training.zero_shot import zero_shot_classifier 
from src.training.logger import setup_logging
from src.training.distributed import is_master, init_distributed_device, broadcast_object
from src.training.precision import get_autocast
import random
from params import parse_args
from prompt_templates import template_dict
from benchmark_dataset_info import BENCHMARK_DATASET_INFOMATION

Image.MAX_IMAGE_PIXELS = 1000000000 

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


class CsvDatasetForClassification(Dataset):
    """Dataset for multiclass classification"""
    def __init__(self, input_filename, transforms, img_key, label_key, classnames, sep="\t", debugging=False, root_data_dir=None):
#         logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        df = df[df[label_key].isnull() == False]
        if root_data_dir is not None:
            df[img_key] = df[img_key].apply(lambda x: join(root_data_dir, x))
        self.images = df[img_key].tolist()
        self.labels = df[label_key].tolist()
        self.transforms = transforms
        self.debugging = debugging
        
        # mapping classname to class index
        if type(self.labels[0]) == str:
            self.label2idx = {x: i for i, x in enumerate(classnames)}
            self.label_indices = [self.label2idx[x] for x in self.labels]
        else:
            self.idx2label = {i: x for i, x in enumerate(classnames)}
            self.label_indices = self.labels
#         logging.debug('Done loading data.')

    def __len__(self):
        return len(self.label_indices)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.debugging:
            return images, self.label_indices[idx], self.images[idx]
        else:
            return images, self.label_indices[idx]


class CsvDatasetForClassificationBinary(Dataset):
    """Dataset for binary classification"""
    def __init__(self, input_filename, transforms, img_key, label_key, actual_label_key, classnames, sep="\t", 
                debugging=False, root_data_dir=None):
#         logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        df = df[df[label_key].isnull() == False]
        if root_data_dir is not None:
            df[img_key] = df[img_key].apply(lambda x: join(root_data_dir, x))
        self.images = df[img_key].tolist()
        self.labels = df[label_key].tolist()
        self.actual_labels = df[actual_label_key].tolist()
        self.transforms = transforms
        self.debugging = debugging

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.debugging:
            return images, self.actual_labels[idx], self.images[idx]
        else:
            return images, self.actual_labels[idx]
        
        
def test_zero_shot_classification(model, dataloader, label_list, is_binary, args, dataset_name='unnamed', debugging=False):
#     logging.info('Starting zero-shot classification test.')
    templates = template_dict[dataset_name]
    model.eval()
    classifier = zero_shot_classifier(model, label_list, templates, args) # [dim_embedding, N_class]
    
    if is_binary:
        results = run_binary(model, classifier, dataloader, args, dataset_name=dataset_name, debugging=debugging)
    else:
        results = run(model, classifier, dataloader, args, dataset_name=dataset_name, debugging=debugging)
    return results


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args, dataset_name='unnamed', debugging=False):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    predictions = []
    labels = []
    all_img_paths = []
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for tup in tqdm(dataloader, unit_scale=args.batch_size):
            if len(tup) == 2:
                images, target = tup
                image_paths = None
            elif len(tup) == 3:
                images, target, image_paths = tup
            else:
                raise ValueError('Dataloader must return 2 or 3 elements.')
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            
            if debugging:
                prediction = torch.argmax(logits, 1)
                prediction = prediction.cpu().tolist() # predicted class index
                predictions.extend(prediction)
                label = target.cpu().tolist() # ground-truth class index
                labels.extend(label)
                if image_paths is not None:
                    all_img_paths.extend(image_paths)

    top1 = (top1 / n)
    top5 = (top5 / n)
    
    results = {dataset_name + '-top1': top1, dataset_name + '-top5': top5}
    if debugging:
        results[dataset_name + '-predictions'] = predictions
        results[dataset_name + '-labels'] = labels
        if all_img_paths:
            results[dataset_name + '-image_paths'] = all_img_paths
    
    return results


def run_binary(model, classifier, dataloader, args, dataset_name='unnamed', debugging=False):    
    """Run binary classification testing"""
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    thres_list = np.linspace(-1, 1, 101) # the logit is expected to be within [-1, 1]
    metrics_dict = {thres: {'TP': 0., 'TN': 0., 'FP': 0, 'FN': 0} for thres in thres_list}
    predictions = []
    labels = []
    all_img_paths = []
    with torch.no_grad():
        for tup in tqdm(dataloader, unit_scale=args.batch_size):
            if len(tup) == 2:
                images, target = tup
                image_paths = None
            elif len(tup) == 3:
                images, target, image_paths = tup
            else:
                raise ValueError('Dataloader must return 2 or 3 elements.')
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = image_features @ classifier
                for thres in thres_list:
                    preds = logits.view(-1) >= thres
                    metrics_dict[thres]['TP'] += torch.sum((preds == 1) * (target == 1)).cpu().item()
                    metrics_dict[thres]['TN'] += torch.sum((preds == 0) * (target == 0)).cpu().item()
                    metrics_dict[thres]['FP'] += torch.sum((preds == 1) * (target == 0)).cpu().item()
                    metrics_dict[thres]['FN'] += torch.sum((preds == 0) * (target == 1)).cpu().item()
            
            if debugging:
                prediction = logits.view(-1).cpu().tolist() # logit score
                predictions.extend(prediction)
                label = target.cpu().tolist() # ground-truth class index
                labels.extend(label)
                if image_paths is not None:
                    all_img_paths.extend(image_paths)
    
    best_f1 = 0.
    best_thres = 0.
    best_prec = 0.
    best_rec = 0.
    for thres in thres_list:
        prec = (metrics_dict[thres]['TP'] + 1e-9) * 1.0 / (metrics_dict[thres]['TP'] + metrics_dict[thres]['FP'] + 1e-9)
        rec = (metrics_dict[thres]['TP'] + 1e-9) * 1.0 / (metrics_dict[thres]['TP'] + metrics_dict[thres]['FN'] + 1e-9)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
        metrics_dict[thres]['precision'] = prec
        metrics_dict[thres]['recall'] = rec
        metrics_dict[thres]['F1'] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_thres = thres
            best_prec = prec
            best_rec = rec
    
    results = {
        dataset_name + '-best_logit_threshold': best_thres, 
        dataset_name + '-best_F1': best_f1, 
        dataset_name + '-best_precision': best_prec, 
        dataset_name + '-best_recall': best_rec,
    }
    
    if debugging:
        results[dataset_name + '-logits'] = predictions
        results[dataset_name + '-labels'] = labels
        if all_img_paths:
            results[dataset_name + '-image_paths'] = all_img_paths
    
    return results


def get_test_dataloaders(args, preprocess_fn):
    test_dataloaders = {}
    if args.datasets_for_testing is not None:
        benchmark_dataset_info = {}
        for dataset_name in args.datasets_for_testing:
            if dataset_name not in BENCHMARK_DATASET_INFOMATION:
                raise ValueError(f'Dataset {dataset_name} is not in BENCHMARK_DATASET_INFOMATION.')
            benchmark_dataset_info[dataset_name] = BENCHMARK_DATASET_INFOMATION[dataset_name]
            
    elif args.test_data_name is not None:
        benchmark_dataset_info = {
            args.test_data_name: {
                'classification_mode': args.classification_mode,
                'test_data': args.test_data,
                'classnames': args.classnames,
                'csv_separator': args.csv_separator,
                'csv_img_key': args.csv_img_key,
                'csv_class_key': args.csv_class_key,
            }
        }
        if args.classification_mode == 'binary':
            benchmark_dataset_info[args.test_data_name]['csv_actual_label_key'] = args.csv_actual_label_key
    
    else:
        raise ValueError(f'Either datasets_for_testing or test_data_name must be given.')
        
    for dataset_name in benchmark_dataset_info:
        label_list = []
        with open(benchmark_dataset_info[dataset_name]['classnames'], 'r') as f:
            for line in f:
                label_list.append(line.strip())
        if benchmark_dataset_info[dataset_name]['classification_mode'] == 'binary':
            ds = CsvDatasetForClassificationBinary(
                benchmark_dataset_info[dataset_name]['test_data'], 
                preprocess_fn, 
                benchmark_dataset_info[dataset_name]['csv_img_key'], 
                benchmark_dataset_info[dataset_name]['csv_class_key'], 
                benchmark_dataset_info[dataset_name]['csv_actual_label_key'], 
                label_list, 
                benchmark_dataset_info[dataset_name]['csv_separator'],
                debugging=args.debugging,
                root_data_dir=args.root_data_dir,
            )
        else:
            ds = CsvDatasetForClassification(
                benchmark_dataset_info[dataset_name]['test_data'], 
                preprocess_fn, 
                benchmark_dataset_info[dataset_name]['csv_img_key'], 
                benchmark_dataset_info[dataset_name]['csv_class_key'], 
                label_list, 
                benchmark_dataset_info[dataset_name]['csv_separator'], 
                debugging=args.debugging,
                root_data_dir=args.root_data_dir,
            )

        test_dataloaders[dataset_name] = {
            'dataloader': torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, 
                                                      shuffle=False, sampler=None),
            'labels': label_list,
            'is_binary': benchmark_dataset_info[dataset_name]['classification_mode'] == 'binary',
        }
    return test_dataloaders


def test(args):
    args = parse_args(args)
    
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    
    random_seed(42, 0)
    
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )
    
    
    random_seed(42, 0)

   # random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
    # construct all dataloaders for testing
    test_dataloaders = get_test_dataloaders(args, preprocess_val)
    
    results_all = {}
    for dataset_name in test_dataloaders:
        results = test_zero_shot_classification(model, test_dataloaders[dataset_name]['dataloader'], 
                                                test_dataloaders[dataset_name]['labels'], 
                                                test_dataloaders[dataset_name]['is_binary'], args, 
                                                dataset_name=dataset_name, debugging=args.debugging)
        for k in results:
            results_all[k] = results[k]
    
#     print(results_all)
    if args.test_result_save_path:
        with open(args.test_result_save_path, 'wb') as f:
            pickle.dump({'model': args.model, 'checkpoint': args.pretrained, 'results': results_all}, f)
    
    return results_all

if __name__ == "__main__":
    test(sys.argv[1:])
