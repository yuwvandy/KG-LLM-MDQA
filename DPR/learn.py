import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils import move_to_cuda
import numpy as np

def train(model, dataloader, optimizer, scheduler, args):
    model.train()
    losses = []

    for batch in tqdm(dataloader):
        batch = move_to_cuda(batch)

        loss = mp_loss(model, batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        losses.append(loss.item())
    
    return np.mean(losses)
    

def mp_loss(model, batch):
    embs = model(batch)
    loss_fct = CrossEntropyLoss(ignore_index = -1)

    c_embs = embs["c_emb"]
    n_embs = embs["n_emb"]

    scores = torch.mm(embs["q_emb"], c_embs.t()) # B x 2B
    n_scores = torch.mm(embs["q_emb"], n_embs.t())

    # mask the 1st hop
    scores = torch.cat([scores, n_scores], dim=1)
    target = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device)

    loss = loss_fct(scores, target)

    return loss

@torch.no_grad()
def eval(model, dataloader):
    model.eval()

    rrs = []
    for batch in tqdm(dataloader):
        batch = move_to_cuda(batch)

        embs = model(batch)
        eval_results = mhop_eval(embs)

        _rrs = eval_results['rrs']
        rrs += _rrs
    
    return np.mean(rrs)

@torch.no_grad()
def mhop_eval(embs):
    c_embs = embs['c_emb']
    n_embs = embs["n_emb"]

    scores = torch.mm(embs["q_emb"], c_embs.t()) #B * 2B
    n_scores = torch.mm(embs["q_emb"], n_embs.t())

    scores = torch.cat([scores, n_scores], dim=1)
    target = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device)

    ranked_hop = scores.argsort(dim=1, descending=True)
    idx2ranked = ranked_hop.argsort(dim=1)
    
    rrs = []
    for t, idx2ranked in zip(target, idx2ranked):
        rrs.append(1 / (idx2ranked[t].item() + 1))

    return {"rrs": rrs}
