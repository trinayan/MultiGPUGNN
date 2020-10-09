import torch
from sklearn.metrics import f1_score

from utils import load_data, EarlyStopping


import os

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

import argparse
from utils import setup
import torch as th

import math

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def train(gpu,args):
    rank = args.nr * args.gpus + gpu
    print(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    cuda_string = 'cuda'+':'+str(gpu)
    device = torch.device(cuda_string) if torch.cuda.is_available()  else torch.device("cpu")

    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args.dataset)



    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()


    print(train_mask.size())
    train_mask = th.split(train_mask, math.ceil(len(train_mask) / args.gpus))[rank] 
    labels = th.split(labels, math.ceil(len(labels) / args.gpus))[rank]
    features = th.split(features, math.ceil(len(features) / args.gpus))[rank]
    #g = th.split(g, math.ceil(len(g) / args.gpus))[rank]
    print(train_mask.size(), labels.size(),features.size(),len(g))
    print(type(g))    
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    if args.hetero:
        from model_hetero import HAN
        model = HAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
                    in_size=features.shape[1],
                    hidden_size=args.hidden_units,
                    out_size=num_classes,
                    num_heads=args.num_heads,
                    dropout=args.dropout).to(device)

        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        g = g.to(device)
    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args.hidden_units,
                    out_size=num_classes,
                    num_heads=args.num_heads,
                    dropout=args.dropout).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        g = [graph.to(device) for graph in g]

    stopper = EarlyStopping(patience=args.patience)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))



def main():
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    args = setup(args)
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '129.10.52.124'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))



    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    

if __name__ == '__main__':
    main()
