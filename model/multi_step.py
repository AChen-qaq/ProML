import torch
from torch import nn
from torch.functional import F
import util.framework


class MultiStep(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(
            self, word_encoder, ignore_index=ignore_index
        )
        self.dot = dot
        self.proj = nn.Linear(768, 32)
        self.token_type_emb = nn.Embedding(2, 768)
        self.support_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(768, 8), 2
        )
        self.weight_proj = nn.Linear(768, 769)

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

    def compute_NCA_loss(
        self,
        S: torch.Tensor,
        tag: torch.Tensor,
        mask: torch.Tensor,
        Q: torch.Tensor,
        tag_q: torch.Tensor,
        mask_q: torch.Tensor,
    ):
        S = F.normalize(S[mask == 1].view(-1, S.shape[-1]), dim=-1)
        assert S.shape[0] == tag.shape[0]
        Q = F.normalize(Q, dim=-1)

        dists = self.__batch_dist__(
            S, Q, mask_q
        )  # [num_query_tokens, num_support_tokens]

        losses = []
        for label in range(1, torch.max(tag) + 1):
            Xp = (tag == label).sum()
            if Xp.item():
                A = torch.exp(dists[tag_q == label, :][:, tag == label]).sum(dim=1)
                B = torch.exp(dists[tag_q == label, :]).sum(dim=1)
                losses.append(-torch.log(A) + torch.log(B))
        loss = torch.cat(losses, dim=0).mean()
        return loss

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

        S = self.proj(support_emb)
        Q = self.proj(query_emb)

        logits = []
        current_support_num = 0
        current_query_num = 0
        bsz = len(support['sentence_num'])
        loss = torch.tensor(0.0, device=self.proj.weight.device)
        for sent_support_num, sent_query_num in zip(
            support['sentence_num'], query['sentence_num']
        ):
            mask = support['text_mask'][
                current_support_num : current_support_num + sent_support_num
            ]
            tag = torch.cat(
                support['label'][
                    current_support_num : current_support_num + sent_support_num
                ],
                0,
            )
            mask_q = query['text_mask'][
                current_query_num : current_query_num + sent_query_num
            ]
            tag_q = torch.cat(
                query['label'][current_query_num : current_query_num + sent_query_num],
                0,
            ).to(self.proj.weight.device)

            loss += (
                self.compute_NCA_loss(
                    S[current_support_num : current_support_num + sent_support_num],
                    tag,
                    mask,
                    Q[current_query_num : current_query_num + sent_query_num],
                    tag_q,
                    mask_q,
                )
                / bsz
            )

            tmp = support_emb[
                current_support_num : current_support_num + sent_support_num
            ]
            tmp = tmp[mask == 1].view(-1, tmp.shape[-1])
            feat = (
                self.support_encoder(
                    (
                        tmp
                        + self.token_type_emb(
                            (tag.to(self.proj.weight.device) != 0).long()
                        )
                    ).unsqueeze(1)
                )
                .squeeze(1)
                .mean(dim=0)
            )
            feat = self.weight_proj(feat)
            weight = feat[:-1]
            bias = feat[-1]
            tmp = query_emb[current_query_num : current_query_num + sent_query_num]
            score = (
                torch.mm(
                    tmp[mask_q == 1].view(-1, tmp.shape[-1]), weight.unsqueeze(1)
                ).squeeze(1)
                + bias
            )
            loss += (
                F.binary_cross_entropy(torch.sigmoid(score), (tag_q != 0).float()) / bsz
            )

            logits_ = self.__get_nearest_dist__(
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
            tmp = torch.exp(logits_[:, 1:]).sum(dim=1).log() - score
            logits.append(
                torch.cat([tmp.unsqueeze(1), logits_[:, 1:]], dim=1)
                - torch.log(1 + torch.exp(-score)).unsqueeze(1)
            )

            current_support_num += sent_support_num
            current_query_num += sent_query_num
        logits = torch.cat(logits, dim=0)
        _, pred = torch.max(logits, dim=1)
        return loss, logits, pred
