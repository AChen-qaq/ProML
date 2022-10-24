import argparse
import torch
from torch.functional import F
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from tqdm import tqdm
from model.container import Container
from util.data_loader import get_loader
from util.word_encoder import BERTWordEncoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--get-stat', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    if args.get_stat:
        device = 'cuda'
        word_encoder = BERTWordEncoder('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = Container(word_encoder, dot=False, ignore_index=-1, gaussian_dim=32)
        model.load_state_dict(
            torch.load(
                'checkpoint/container_32_INTRA-0.pth.tar', map_location=device
            )['state_dict']
        )
        model.to(device)
        model.eval()
        test_data_loader = get_loader(
            # 'data/intra/test.txt',
            'data/intra/train.txt',
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
        for index, (support, query) in enumerate(test_data_loader):
            for k in support:
                if k != 'label' and k != 'sentence_num':
                    support[k] = support[k].to(device)
                    query[k] = query[k].to(device)
            for batch in [support, query]:
                with torch.no_grad():
                    emb = model.word_encoder(batch['word'], batch['mask'])
                    s = model.elu(model.fs(F.relu(emb))) + 1 + 1e-6
                    s = s[batch['text_mask'] == 1].view(-1, s.shape[-1])
                    tag = torch.cat(batch['label'], dim=0)
                    for label in range(torch.max(tag) + 1):
                        tmp = s[tag == label].cpu()
                        tmp = tmp.view(-1)
                        label = 'O' if label == 0 else 'non-O'
                        if label in vars:
                            vars[label] = torch.cat([vars[label], tmp], dim=0)
                        else:
                            vars[label] = tmp
            pbar.update()
            if index == 500:
                break
        torch.save(vars, 'outputs/var_dist.pt')
    if args.plot:
        vars = torch.load('outputs/var_dist.pt')
        for label in reversed(vars):
            plt.hist(
                vars[label].detach().cpu().numpy(),
                bins=40,
                density=True,
                alpha=0.8,
                label=label,
            )
            # plt.savefig(f'outputs/var_dist_{label}.jpg')
            # plt.close()
        plt.legend()
        plt.savefig('outputs/var_dist.jpg',dpi=500)


if __name__ == '__main__':
    args = get_args()
    main(args)
