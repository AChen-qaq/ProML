from torch.nn import functional as F
from torch import nn, distributions
import torch
import util.framework
import sys

sys.path.append('..')


class Triplet(util.framework.FewShotNERModel):
    def __init__(
        self,
        word_encoder,
        dot=False,
        ignore_index=-1,
        proj_dim=32,
        margin: float = 30,
        triplet_mode: int = 0,
        mix_NCA: bool = False,
    ):
        util.framework.FewShotNERModel.__init__(
            self, word_encoder, ignore_index=ignore_index
        )
        self.dot = dot
        self.dim = proj_dim
        # self.dim = 768
        # self.dim = 32
        self.fu = nn.Linear(768, self.dim)
        self.fu.weight.data /= (self.dim / 32) ** 0.5
        self.fu.bias.data /= (self.dim / 32) ** 0.5
        self.drop = nn.Dropout(0.3)
        self.margin = margin
        self.triplet_mode = triplet_mode
        self.mix_NCA = mix_NCA
        self.cnt = 1000

    def __dist_JS__(self, xu, yu):
        # KL = 0.5 * (
        #     (ys / xs + xs / ys + (xu - yu) ** 2 * (1 / xs + 1 / ys)).sum(dim=-1)
        #     - 64  # TODO:
        # )
        L2 = ((xu - yu) ** 2).sum(dim=-1)
        # print(L2)
        # exit()
        # L2 = torch.clamp(L2, 0.03, 2) # 试试试试，记得删
        # print(L2.shape, xw.shape, yw.shape)
        # exit()
        # L2 = L2 * xw.squeeze(-1) * yw.squeeze(-1) # 记得加
        return L2

    def __batch_dist_JS__(
        self,
        su: torch.Tensor,
        qu: torch.Tensor,
    ):
        return self.__dist_JS__(su.unsqueeze(0), qu.unsqueeze(1))

    def compute_triplet_loss(
        self,
        su: torch.Tensor,
        tag: torch.Tensor,
        mask: torch.Tensor,
        qu: torch.Tensor,
        tag_q: torch.Tensor,
        mask_q: torch.Tensor,
    ):
        device = self.fu.weight.device
        su = su[mask == 1].view(-1, su.shape[-1])
        tag = torch.cat(tag, dim=0).to(device)
        assert su.shape[0] == tag.shape[0]
        qu = qu[mask_q == 1].view(-1, qu.shape[-1])
        tag_q = torch.cat(tag_q, dim=0).to(device)
        assert qu.shape[0] == tag_q.shape[0]

        dists = self.__batch_dist_JS__(su, qu)  # [num_query_tokens, num_support_tokens]

        if self.mix_NCA and self.cnt > 0:
            self.cnt -= 1
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
            return torch.cat(losses, dim=0).mean()
        same_class_mask = tag_q.unsqueeze(1) == tag.unsqueeze(0)
        if self.triplet_mode == 2:
            tmp = (
                dists.masked_fill(same_class_mask == 0, float('inf'))
                .topk(15, dim=1, largest=False)
                .values
            )
            indices = torch.randint(0, 5, (tmp.shape[0], 1), device=tmp.device)
            d1 = tmp.gather(1, indices)
        else:
            d1 = dists.masked_fill(same_class_mask == 0, float('inf')).min(dim=1).values
            if self.triplet_mode == 1:
                d1_ = (
                    dists.masked_fill(same_class_mask == 0, float('-inf'))
                    .max(dim=1)
                    .values
                )
                d1 = torch.where(tag_q == 0, d1, d1_)
        d2 = dists.masked_fill(same_class_mask, float('inf')).min(dim=1).values
        loss = (d1 - d2 + self.margin).clamp(0).mean()
        return loss
        # TODO: 还没写完

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

        support_u = self.fu(F.relu(self.drop(support_emb)))
        # print(self.dim)
        # print(support_u.shape, support_u.norm(2,dim=-1).shape)
        # print(support_u.norm(2,dim=-1).mean())
        # exit()
        query_u = self.fu(F.relu(self.drop(query_emb)))
        # if self.normalize:
        #     support_u = F.normalize(support_u) * self.norm
        #     query_u = F.normalize(query_u) * self.norm
        # else:
        #     support_u = support_u * self.norm
        #     query_u = query_u * self.norm

        logits = []
        current_support_num = 0
        current_query_num = 0
        bsz = len(support['sentence_num'])
        loss = torch.tensor(0.0, device=self.fu.weight.device)
        for sent_support_num, sent_query_num in zip(
            support['sentence_num'], query['sentence_num']
        ):
            loss += (
                self.compute_triplet_loss(
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
