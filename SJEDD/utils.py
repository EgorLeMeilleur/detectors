import pdb

import torch, os
import torch.distributed as dist
import copy

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def save_checkpoint(config, args, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  # 'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  # 'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config,
                  'args': args}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def load_checkpoint(config, model, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp64(model):
    for p in model.parameters():
        p.data = p.data.double()
        if p.grad is not None:
            p.grad.data = p.grad.data.double()

def freeze_model(model, logger, opt=0):
    model.logit_scale.requires_grad = False
    if opt == 0: #do nothing
        return
    elif opt == 1: # freeze text encoder
        logger.info(f'freeze text encoder...')
        for p in model.token_embedding.parameters():
            p.requires_grad = False
        for p in model.transformer.parameters():
            p.requires_grad = False
        model.positional_embedding.requires_grad = False
        model.text_projection.requires_grad = False
        for p in model.ln_final.parameters():
            p.requires_grad = False

        # for p in model.text_projection.parameters():
        #     p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in model.parameters():
            p.requires_grad =False

def transfer_label(target, mode=None, dataset=''):

    # if dataset not in ['ffsc']:
    #     gt_level_1 = target[:, 0: 2].argmax(dim=1)  # .numpy().tolist()
    #     gt_level_2 = target[:, 2: 5]
    #     gt_level_3 = target[:, 5:]
    if dataset in ['ffpp']:
        gt_level_1 = target[:, 0]
        gt_level_2 = target[:, 1: 4]
        gt_level_3 = target[:, 4:]
    else:
        gt_level_1 = target[:, 0]#.argmax(dim=1)  # .numpy().tolist()
        gt_level_2 = target[:, 1: 6]
        gt_level_3 = target[:, 6:]

    # if mode == 'celoss_multiclass':
    #     gt_level_2 = torch.where(gt_level_2 > 0)[1]
    # elif mode == 'fidelity':
    #     gt_level_1 = target[:, 0: 2]
    # elif mode == 'celoss_baseline':
    #     gt_level_1 = target[:, 0: 2]

    return gt_level_1, gt_level_2, gt_level_3


def accuracy_global(logits_l2, target):
    probs = logits_l2.cpu()
    _, preds = torch.max(probs.data, 1)
    g1, g2, g3 = transfer_label(target, mode='celoss')

    acc = torch.sum(preds == g2.cpu().data).to(torch.float32) / g2.shape[0]
    return acc

from sklearn.metrics import accuracy_score
def accuracy_local(logits_l3, target):
    probs = logits_l3.cpu()
    g1, g2, g3 = transfer_label(target, mode='celoss')

    preds = torch.zeros_like(g3)
    for i in range(g3.shape[0]):
        _, maxk = torch.topk(probs[i].float(), g3[i].sum(), 0)
        preds[i, maxk] = 1

    acc = accuracy_score(g3.cpu().data, preds.cpu().data)
    return acc



