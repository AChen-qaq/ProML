import torch
from torch import nn, optim
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from util.word_encoder import BERTWordEncoder
from util.data_loader import get_loader
from model.container import Container


def train_svm(x: torch.Tensor, tag: torch.Tensor, l2_reg: float):
    # x: [num_tokens, hidden_dim]
    tag = tag.to(x.device)
    labels = torch.where(tag != 0, 1, -1)
    f = nn.Linear(x.shape[1], 1)
    f.to(x.device)
    optimizer = optim.SGD(f.parameters(), lr=1e-2)
    for _ in range(int(3e3)):
        optimizer.zero_grad()
        loss1 = (1 - labels * (f(x).squeeze(1))).clamp(0).mean()
        loss2 = torch.sum(f.weight * f.weight)
        # print('{:.3f} {:.3f}'.format(loss1.item(), loss2.item()), flush=True)
        (loss1 + l2_reg * loss2).backward()
        optimizer.step()
    return f


if __name__ == '__main__':
    device = 'cuda'
    word_encoder = BERTWordEncoder('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = Container(word_encoder, dot=False, ignore_index=-1, gaussian_dim=32)
    model.load_state_dict(
        torch.load('checkpoint/container0.pth.tar', map_location=device)['state_dict']
    )
    model.to(device)
    model.eval()
    test_data_loader = get_loader(
        'data/intra/test.txt',
        tokenizer,
        N=5,
        K=5,
        Q=5,
        batch_size=1,
        max_length=32,
        ignore_index=-1,
        use_sampled_data=False,
    )
    vars = {}
    pbar = tqdm(total=500)
    fp_list=[]
    fn_list=[]
    for index, (support, query) in enumerate(test_data_loader):
        for k in support:
            if k != 'label' and k != 'sentence_num':
                support[k] = support[k].to(device)
                query[k] = query[k].to(device)
        # assert False
    # support, query = torch.load('outputs/sampled_batch.pt')
        support_emb = model.word_encoder(support['word'], support['mask'])
        query_emb = model.word_encoder(query['word'], query['mask'])
        tag = torch.cat(support['label'], dim=0).to(device)
        tag_q = torch.cat(query['label'], dim=0).to(device)
        f = train_svm(support_emb[support['text_mask'] == 1].detach(), tag, 1e-2)
        # pred = f(support_emb[support['text_mask'] == 1].detach()) > 0
        # pred.squeeze_(1)
        # fp = ((pred == True) * (tag == 0)).sum() / (tag == 0).sum()
        # fn = ((pred == False) * (tag != 0)).sum() / (tag != 0).sum()
        # print(fp, fn)
        pred = f(query_emb[query['text_mask'] == 1].detach()) > 0
        pred.squeeze_(1)
        fp = ((pred == True) * (tag_q == 0)).sum() / (tag_q == 0).sum()
        fn = ((pred == False) * (tag_q != 0)).sum() / (tag_q != 0).sum()
        fp_list.append(fp.item())
        fn_list.append(fn.item())
        # print(fp, fn)
        pbar.set_description('fp: {:.4f} fn: {:.4f}'.format(np.mean(fp_list), np.mean(fn_list)))
        pbar.update()
        if index == 500:
            break
