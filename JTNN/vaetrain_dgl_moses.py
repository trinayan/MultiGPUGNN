import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import argparse
import math, random, sys

from collections import deque
import rdkit
import tqdm

from jtnn import *


import os

import torch.multiprocessing as mp
import torch.distributed as dist


torch.multiprocessing.set_sharing_strategy('file_system')

def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
worker_init_fn(None)





MAX_EPOCH = 100
PRINT_ITER = 20

def train(gpu,args):
    dataset = JTNNDatasetMoses(data=args.train, vocab=args.vocab, training=True)
    vocab = dataset.vocab

    batch_size = int(args.batch_size)
    hidden_size = int(args.hidden_size)
    latent_size = int(args.latent_size)
    depth = int(args.depth)
    beta = float(args.beta)
    lr = float(args.lr)

    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    cuda_string = 'cuda'+':'+str(gpu)
    device = torch.device(cuda_string) if torch.cuda.is_available()  else torch.device("cpu")

    model = DGLJTNNVAE(vocab, hidden_size, latent_size, depth).to(device)

    if args.model_path is not None:
        model.load_state_dict(torch.load(opts.model_path))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal(param)




    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()


    dataset.training = True

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=JTNNCollator(vocab, True),
            drop_last=True,
            worker_init_fn=worker_init_fn,
            sampler = train_sampler)

    for epoch in range(MAX_EPOCH):
        word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0

        for it, batch in enumerate(tqdm.tqdm(dataloader)):
            model.zero_grad()
            try:
                loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta)
            except:
                print([t.smiles for t in batch['mol_trees']])
                raise
            loss.backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100

                print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Loss: %.6f" % (
                    kl_div, word_acc, topo_acc, assm_acc, steo_acc, loss.item()))
                word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0
                sys.stdout.flush()

            if (it + 1) % 1500 == 0: #Fast annealing
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
                torch.save(model.state_dict(),
                           opts.save_path + "/model.iter-%d-%d" % (epoch, it + 1))

        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))

def test():
    dataset.training = False
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=JTNNCollator(vocab, False),
            drop_last=True,
            worker_init_fn=worker_init_fn)

    # Just an example of molecule decoding; in reality you may want to sample
    # tree and molecule vectors.
    for it, batch in enumerate(dataloader):
        gt_smiles = batch['mol_trees'][0].smiles
        print(gt_smiles)
        model.move_to_cuda(batch)
        _, tree_vec, mol_vec = model.encode(batch)
        tree_vec, mol_vec, _, _ = model.sample(tree_vec, mol_vec)
        smiles = model.decode(tree_vec, mol_vec)
        print(smiles)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", dest="train", default='train', help='Training file name')
    parser.add_argument("-v", "--vocab", dest="vocab", default='vocab', help='Vocab file name')
    parser.add_argument("-s", "--save_dir", dest="save_path")
    parser.add_argument("-m", "--model", dest="model_path", default=None)
    parser.add_argument("-b", "--batch", dest="batch_size", default=40)
    parser.add_argument("-w", "--hidden", dest="hidden_size", default=200)
    parser.add_argument("-l", "--latent", dest="latent_size", default=56)
    parser.add_argument("-d", "--depth", dest="depth", default=3)
    parser.add_argument("-z", "--beta", dest="beta", default=1.0)
    parser.add_argument("-q", "--lr", dest="lr", default=1e-3)
    parser.add_argument("-T", "--test", dest="test", action="store_true")
    parser.add_argument('--no-train', default=False)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '129.10.52.124'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))




if __name__ == '__main__':
    main()
