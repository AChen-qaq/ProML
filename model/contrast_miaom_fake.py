from torch._C import device
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch
import math
import util
import sys

sys.path.append('..')


class Bailan(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(
            self, word_encoder, ignore_index=ignore_index
        )
        self.dot = dot
        self.fu = nn.Linear(1,1)

    def __dist_JS__(self, xu, yu):
        # KL = 0.5 * (
        #     (ys / xs + xs / ys + (xu - yu) ** 2 * (1 / xs + 1 / ys)).sum(dim=-1)
        #     - 64  # TODO:
        # )
        L2 = torch.exp(-((xu - yu)**2).sum(dim=-1))
        # print(L2.shape, xw.shape, yw.shape)
        # exit()
        # L2 = L2 * xw.squeeze(-1) * yw.squeeze(-1)
        return L2

    def __batch_dist_JS__(
        self,
        su: torch.Tensor,
        qu: torch.Tensor,
    ):
        return self.__dist_JS__(
            su.unsqueeze(0), qu.unsqueeze(1)
        )

    def compute_NCA_loss(
        self,
        su: torch.Tensor,
        tag: torch.Tensor,
        mask: torch.Tensor,
        qu: torch.Tensor,
        tag_q: torch.Tensor,
        mask_q: torch.Tensor,
    ):
        su = su[mask == 1].view(-1, su.shape[-1])
        tag = torch.cat(tag, dim=0)
        assert su.shape[0] == tag.shape[0]
        qu = qu[mask_q == 1].view(-1, qu.shape[-1])
        tag_q = torch.cat(tag_q, dim=0)
        assert qu.shape[0] == tag_q.shape[0]

        dists = self.__batch_dist_JS__(
            su, qu
        )  # [num_query_tokens, num_support_tokens]

        losses = []
        for label in range(torch.max(tag) + 1):
            Xp = (tag == label).sum()
            if Xp.item():
                A = (
                    dists[tag_q == label, :][:, tag == label].sum(dim=1)
                    / Xp
                )
                B = dists[tag_q == label, :].sum(dim=1)
                losses.append(-torch.log(A) + torch.log(B))
                # print(A, B, losses[-1])
        loss = torch.cat(losses, dim=0).mean()
        
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

    def forward(self, support, query):
        support_emb = self.word_encoder(support['word'], support['mask'])
        query_emb = self.word_encoder(query['word'], query['mask'])
        # [num_sent, num_tokens, hidden_size]

        support_u = F.normalize(support_emb)
        query_u = F.normalize(query_emb)
        # support_w = self.lu(self.fw(F.relu(support_emb)))
        # query_w = self.lu(self.fw(F.relu(query_emb)))

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
                    support['label'][
                        current_support_num : current_support_num + sent_support_num
                    ],
                    support['text_mask'][
                        current_support_num : current_support_num + sent_support_num
                    ],
                    query_u[current_query_num : current_query_num + sent_query_num],
                    query['label'][
                        current_query_num : current_query_num + sent_query_num
                    ],
                    query['text_mask'][
                        current_query_num : current_query_num + sent_query_num
                    ],
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
