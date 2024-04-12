# Models
import torch.nn as nn
from torch import optim
from patchtstblind.models.patchtst import PatchTSTBlind

from patchtstblind.utils.schedulers import WarmupCosineSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_model(args):
    return PatchTSTBlind(num_enc_layers=args.num_enc_layers,
                              d_model=args.d_model,
                              d_ff=args.d_ff,
                              num_heads=args.num_heads,
                              num_channels=args.num_channels,
                              seq_len=args.seq_len,
                              pred_len=args.pred_len,
                              attn_dropout=args.attn_dropout,
                              ff_dropout=args.ff_dropout,
                              pred_dropout=args.pred_dropout,
                              batch_first=args.batch_first,
                              norm_mode=args.norm_mode,
                              revin=args.revin,
                              revout=args.revout,
                              revin_affine=args.revin_affine,
                              eps_revin=args.eps_revin,
                              patch_dim=args.patch_dim,
                              stride=args.stride)




def get_optim(args, model, optimizer_type="adamw"):
    print(f"{optimizer_type} optimizer initialized.")
    optimizer_classes = {"adam": optim.Adam, "adamw": optim.AdamW}
    if optimizer_type not in optimizer_classes:
        raise ValueError("Please select a valid optimizer.")
    optimizer_class = optimizer_classes[optimizer_type]
    optimizer = optimizer_class(model.parameters(), weight_decay=args.weight_decay)

    return optimizer


def get_scheduler(args, scheduler_type, training_mode, optimizer):
    print(f"{scheduler_type} scheduler initialized ({training_mode}).")
    if scheduler_type == "cosine_warmup" and training_mode=="pretrain":
        scheduler = WarmupCosineSchedule(optimizer=optimizer,
                                         warmup_steps=args.warmup_steps,
                                         start_lr=args.start_lr,
                                         ref_lr=args.ref_lr,
                                         T_max=args.T_max,
                                         final_lr=args.final_lr)
    elif scheduler_type == "cosine" and training_mode=="downstream":
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.downstream_epochs,
                                      eta_min=args.downstream_final_lr,
                                      last_epoch=args.downstream_last_epoch)
    elif scheduler_type == "cosine" and training_mode=="supervised":
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.sl_epochs,
                                      eta_min=args.final_lr,
                                      last_epoch=args.last_epoch)
    elif scheduler_type == "None":
        return None
    else:
        raise ValueError("Please select a valid scheduler_type.")
    return scheduler


def get_criterion(criterion_type):
    print(f"Using the {criterion_type} criterion.")
    if criterion_type == "MSE":
        return nn.MSELoss()
    elif criterion_type == "SmoothL1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError("Please select a valid criterion_type.")


def compute_loss(output, target, criterion, model_id):
    if model_id in {"PatchTSTBlind"}:
        loss = criterion(output, target)
    else:
        raise ValueError("Please select a valid model_id.")
    return loss


def model_update(model, loss, optimizer, model_id, alpha=0.6):
    if model_id in {"PatchTSTOG", "PatchTSTBlind"}:
        loss.backward()
        optimizer.step()
    else:
        raise ValueError("Please select a valid model_id.")
