from torch._C import device
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch
import math
import util.framework
import sys

sys.path.append('..')


class Container(util.framework.FewShotNERModel):
    def __init__(
        self,
        word_encoder,
        dot=False,
        ignore_index=-1,
        gaussian_dim: int = 32,
        o_ambg: float = None,
    ):
        util.framework.FewShotNERModel.__init__(
            self, word_encoder, ignore_index=ignore_index
        )
        self.dot = dot
        self.gaussian_dim = gaussian_dim
        self.fu = nn.Linear(768, gaussian_dim)
        self.fs = nn.Linear(768, gaussian_dim)
        self.elu = nn.ELU()
        self.o_ambg = o_ambg

    def __dist_JS__(self, xs, ys, xu, yu):
        KL = 0.5 * (
            (ys / xs + xs / ys + (xu - yu) ** 2 * (1 / xs + 1 / ys)).sum(dim=-1)
            - self.gaussian_dim * 2
        )
        return KL / (self.gaussian_dim / 32)

    def __batch_dist_JS__(
        self,
        su: torch.Tensor,
        ss: torch.Tensor,
        qu: torch.Tensor,
        qs: torch.Tensor,
    ):
        return self.__dist_JS__(
            ss.unsqueeze(0), qs.unsqueeze(1), su.unsqueeze(0), qu.unsqueeze(1)
        )

    def compute_NCA_loss(
        self,
        su: torch.Tensor,
        ss: torch.Tensor,
        tag: torch.Tensor,
        mask: torch.Tensor,
        qu: torch.Tensor,
        qs: torch.Tensor,
        tag_q: torch.Tensor,
        mask_q: torch.Tensor,
        it = None
    ):
        su = su[mask == 1].view(-1, su.shape[-1])
        ss = ss[mask == 1].view(-1, ss.shape[-1])
        tag = torch.cat(tag, dim=0)
        assert su.shape[0] == tag.shape[0]
        qu = qu[mask_q == 1].view(-1, qu.shape[-1])
        qs = qs[mask_q == 1].view(-1, qs.shape[-1])
        tag_q = torch.cat(tag_q, dim=0)
        assert qu.shape[0] == tag_q.shape[0]

        dists = self.__batch_dist_JS__(
            su, ss, qu, qs
        )  # [num_query_tokens, num_support_tokens]

        if it == -1:
            dists = dists.masked_fill( torch.eye(len(su)).cuda().bool(), 0)
        losses = []
        for label in range(torch.max(tag) + 1):
            Xp = (tag == label).sum()
            if Xp.item():
                A = (
                    torch.exp(-dists[tag_q == label, :][:, tag == label]).sum(dim=1)
                    / Xp
                )
                B = torch.exp(-dists[tag_q == label, :]).sum(dim=1)
                losses.append(-torch.log(A) + torch.log(B))
        loss = torch.cat(losses, dim=0).mean()
        if self.o_ambg is not None:
            Yp = (tag_q == 0).sum()
            if Yp.item():
                tmp = -dists[tag_q == 0, :][:, tag != 0]
                loss += self.o_ambg * (tmp.softmax(dim=1)*tmp.log_softmax(dim=-1)).mean()
        return loss

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask == 1].view(-1, Q.size(-1))
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_nearest_dist__(self, embedding, tag, mask, query, q_mask):
        nearest_dist = []
        S = embedding[mask == 1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == S.size(0)
        dist = self.__batch_dist__(
            S, query, q_mask
        )  # [num_of_query_tokens, num_of_support_tokens]
        for label in range(torch.max(tag) + 1):
            nearest_dist.append(torch.max(dist[:, tag == label], 1)[0])
        nearest_dist = torch.stack(
            nearest_dist, dim=1
        )  # [num_of_query_tokens, class_num]
        return nearest_dist
    
    def get_emb(self, query):
        query_emb = self.word_encoder(query['word'], query['mask'])
        query_text_mask = query['text_mask']
        return query_emb[query_text_mask == 1].view(-1, query_emb.size(-1))

    def get_emb_query(self, query):
        return self.get_emb(query)

    def get_emb_support(self, support):
        return self.get_emb(support)

    def forward(self, support, query, it=None):
        support_emb = self.word_encoder(support['word'], support['mask'])
        query_emb = self.word_encoder(query['word'], query['mask'])
        # [num_sent, num_tokens, hidden_size]

        support_u = self.fu(F.relu(support_emb))
        support_s = self.elu(self.fs(F.relu(support_emb))) + 1 + 1e-6
        query_u = self.fu(F.relu(query_emb))
        query_s = self.elu(self.fs(F.relu(query_emb))) + 1 + 1e-6

        logits = []
        current_support_num = 0
        current_query_num = 0
        bsz = len(support['sentence_num'])
        loss = torch.tensor(0.0, device=self.fu.weight.device)
        for sent_support_num, sent_query_num in zip(
            support['sentence_num'], query['sentence_num']
        ):
            loss += (
                self.compute_NCA_loss(
                    support_u[
                        current_support_num : current_support_num + sent_support_num
                    ],
                    support_s[
                        current_support_num : current_support_num + sent_support_num
                    ],
                    support['label'][
                        current_support_num : current_support_num + sent_support_num
                    ],
                    support['text_mask'][
                        current_support_num : current_support_num + sent_support_num
                    ],
                    query_u[current_query_num : current_query_num + sent_query_num],
                    query_s[current_query_num : current_query_num + sent_query_num],
                    query['label'][
                        current_query_num : current_query_num + sent_query_num
                    ],
                    query['text_mask'][
                        current_query_num : current_query_num + sent_query_num
                    ],
                    it
                )
                / bsz
            )
            logits.append(
                self.__get_nearest_dist__(
                    support_emb[
                        current_support_num : current_support_num + sent_support_num
                    ],
                    support['label'][
                        current_support_num : current_support_num + sent_support_num
                    ],
                    support['text_mask'][
                        current_support_num : current_support_num + sent_support_num
                    ],
                    query_emb[current_query_num : current_query_num + sent_query_num],
                    query['text_mask'][
                        current_query_num : current_query_num + sent_query_num
                    ],
                )
            )
            current_support_num += sent_support_num
            current_query_num += sent_query_num
        logits = torch.cat(logits, dim=0)
        _, pred = torch.max(logits, dim=1)
        return loss, logits, pred
