from time import time
import logging
import wandb
import torch

from trainer import BaseTrainer
from datasets import MyDataset

TRAINER_DICT = {
    'base': BaseTrainer,
}


def base_procedure(args):
    if args['model_name'] not in TRAINER_DICT.keys():
        raise NotImplementedError("Model {} not implemented".format(args['model_name']))
    
    if args['verbose']:
        logger = logging.info if args['log'] else print

    if args['wandb']:
        wandb.init(
            project=args['wandb_project'], 
            name=args['saving_name'],
            tags=[args['model'], args['dataset']],
            config=args)
    
    if args['verbose']:
        logger("Loading {} dataset, subset is {}".format(args['dataset'], args['subset']))
    start = time()
    dataset = MyDataset(
        data_dir=args['data_dir'],
        train_ratio=args['train_ratio'],
        valid_ratio=args['valid_ratio'],
        test_ratio=args['test_ratio'],
        train_batchsize=args['train_batchsize'],
        eval_batchsize=args['eval_batchsize'],
        subset=args['subset'],
        num_workers=args['num_workers'],
        pin_memory=args['pin_memory'],
    )
    train_loader = dataset.train_loader
    valid_loader = dataset.valid_loader
    test_loader = dataset.test_loader
    if args['verbose']:
        logger("Loading data costs {: .2f}s".format(time() - start))
    
    # build model
    if args['verbose']:
        logger("Building models")
    start = time()
    trainer = TRAINER_DICT[args['model_name']](args)
    model = trainer.build_model(args)
    model = model.to(args['device'])
    optimizer = torch.optim.Adam(
        model.parameters(), 
        betas=(0.9, 0.999),
        lr=args['lr'],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args['milestones'],
        gamma=args['gamma'],
    )
    criterion = torch.nn.MSELoss()
    
    if args['verbose']:
        logger("Model: {}".format(model))
        logger("Criterion: {}".format(criterion))
        logger("Optimizer: {}".format(optimizer))
        logger("Scheduler: {}".format(scheduler))
        logger("Building models costs {: .2f}s".format(time() - start))

    trainer.process(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )
