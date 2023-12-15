import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils import move_to_cuda, move_to_cuda2
import numpy as np

def train(model, dataloader, optimizer, scheduler, args):
    model.train()
    losses = []

    for batch in tqdm(dataloader):
        batch = move_to_cuda2(batch)

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

    c_embs = torch.cat([embs["c1_emb"], embs["c2_emb"]], dim = 0) # 2B x d
    n_embs = torch.cat([embs["n1_emb"].unsqueeze(1), embs["n2_emb"].unsqueeze(1)], dim = 1) # B*2*M*h

    scores_1 = torch.mm(embs["q_emb"], c_embs.t()) # B x 2B
    n_scores_1 = torch.bmm(embs["q_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B x 2B
    scores_2 = torch.mm(embs["q_c1_emb"], c_embs.t()) # B x 2B
    n_scores_2 = torch.bmm(embs["q_c1_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B x 2B

    # mask the 1st hop
    bsize = embs["q_emb"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(embs["q_emb"].device)
    scores_1 = scores_1.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1)
    scores_1 = torch.cat([scores_1, n_scores_1], dim=1)
    scores_2 = torch.cat([scores_2, n_scores_2], dim=1)

    target_1 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device)
    target_2 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device) + embs["q_emb"].size(0)

    loss = loss_fct(scores_1, target_1) + loss_fct(scores_2, target_2)

    return loss

@torch.no_grad()
def eval(model, dataloader):
    model.eval()

    rrs_1, rrs_2 = [], []
    for batch in tqdm(dataloader):
        batch = move_to_cuda2(batch)

        embs = model(batch)
        eval_results = mhop_eval(embs)

        _rrs_1, _rrs_2 = eval_results['rrs_1'], eval_results['rrs_2']
        rrs_1 += _rrs_1
        rrs_2 += _rrs_2
    
    return np.mean(rrs_1), np.mean(rrs_2)


def mhop_eval(embs):
    c_embs = torch.cat([embs['c1_emb'], embs['c2_emb']], dim=0) # (2B) * D
    n_embs = torch.cat([embs["n1_emb"].unsqueeze(1), embs["n2_emb"].unsqueeze(1)], dim=1) # B * 2 * D


    scores_1 = torch.mm(embs["q_emb"], c_embs.t()) #B * 2B
    n_scores_1 = torch.bmm(embs["q_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B * 2
    scores_2 = torch.mm(embs["q_c1_emb"], c_embs.t()) #B * 2B
    n_scores_2 = torch.bmm(embs["q_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B * 2


    bsize = embs["q_emb"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(embs["q_emb"].device)
    scores_1 = scores_1.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1)
    scores_1 = torch.cat([scores_1, n_scores_1], dim=1)
    scores_2 = torch.cat([scores_2, n_scores_2], dim=1)
    target_1 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device)
    target_2 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device) + embs["q_emb"].size(0)

    ranked_1_hop = scores_1.argsort(dim=1, descending=True)
    ranked_2_hop = scores_2.argsort(dim=1, descending=True)
    idx2ranked_1 = ranked_1_hop.argsort(dim=1)
    idx2ranked_2 = ranked_2_hop.argsort(dim=1)
    
    rrs_1, rrs_2 = [], []
    for t, idx2ranked in zip(target_1, idx2ranked_1):
        rrs_1.append(1 / (idx2ranked[t].item() + 1))

    for t, idx2ranked in zip(target_2, idx2ranked_2):
        rrs_2.append(1 / (idx2ranked[t].item() + 1))
    
    return {"rrs_1": rrs_1, "rrs_2": rrs_2}
