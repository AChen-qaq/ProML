from torch._C import device
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd, optim, nn
import random
import torch
import math
import util
import sys

sys.path.append('..')


class Contrast(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(
            self, word_encoder, ignore_index=ignore_index
        )
        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.dot = dot
        self.fu = nn.Linear(768, 32)
        self.fs = nn.Linear(768, 32)
        # self.gu = nn.Linear(768, 20)
        # self.gs = nn.Linear(768, 20)
        self.elu = nn.ELU()
        self.u = nn.Parameter(torch.randn(100, 32))
        self.s = nn.Parameter(F.elu(torch.randn(100, 32))+1+1e-6)

    def __dist_support__(self, xs, ys, xu, yu):
        # print(xs.shape, ys.shape, xu.shape, yu.shape)
        assert xs.shape[1] == ys.shape[0]
        #print(xu.shape, yu.shape)
        #return (F.normalize(xu,dim=-1)*F.normalize(yu,dim=-1)).sum(dim=-1)

        KL = 0.5 * (( ys / xs + xs / ys + (xu - yu)**2 *
                    (1 / xs + 1 / ys)).sum(dim=-1) - 64)
        #KL = KL.clamp(0, 10)
        #KL = F.sigmoid(KL)*10
        #return KL
        return KL.masked_fill(
            torch.eye(ys.shape[0], dtype=torch.bool,
                      device='cuda'), float('inf')
        )

        return xs.shape[1]*F.normalize(
            KL, p=1).masked_fill(
            torch.eye(ys.shape[0], dtype=torch.bool,
                      device='cuda'), float('inf')
        )

    def __batch_dist_support__(self, S, U):
        return self.__dist_support__(
            S.unsqueeze(0), S.unsqueeze(1), U.unsqueeze(0), U.unsqueeze(1)
        )

    def __get_NCAloss__(self, embs, embu, tag, mask, output=False, skip_O=False):
        tmp = []
        embs = embs[mask == 1].view(-1, embs.size(-1))
        embu = embu[mask == 1].view(-1, embu.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embs.size(0)
        dist = self.__batch_dist_support__(embs, embu)
        if output:
            print(dist[0])
        totlen = 0
        for label in range(1 if skip_O else 0, torch.max(tag) + 1):
            Xp = (tag == label).sum()
            totlen += Xp
            # print(dist.shape, dist[tag == label, :].shape,
            #       dist[tag == label, tag == label].shape)
            A = (torch.exp(-dist[tag == label, :][:, tag == label])).sum(dim=1)
            B = torch.exp(-dist[tag == label, :]).sum(dim=1)
            loss = -(torch.log(A).sum(dim=0)-torch.log(B).sum(dim=0))
            if label != 0:
                loss = loss * 0.5
            tmp.append(loss)

        tmp = torch.stack(tmp).sum() / totlen
        return tmp

    def __batch_proxydist_support__(self, S, U):
        S = S.unsqueeze(1)
        U = U.unsqueeze(1)
        s = self.s.unsqueeze(0)
        u = self.u.unsqueeze(0)
        # l2 = 0.5*((F.normalize(U, p=2, dim=-1)-F.normalize(u, p=2, dim=-1))**2).sum(-1)
        # return l2
        KL = 0.5 * ((S / s + s / S + (U - u)
                    ** 2 * (1 / s + 1 / S)).sum(dim=-1) - 64)
        return KL

    def __get_proxyNCAloss__(self, embs, embu, tag, mask, output=False):
        tmp = []
        embs = embs[mask == 1].view(-1, embs.size(-1))
        embu = embu[mask == 1].view(-1, embu.size(-1))
        tag = torch.cat(tag, 0).to('cuda:0')
        #print(tag.size(), embs.size())
        assert tag.size(0) == embs.size(0)
        dist = self.__batch_proxydist_support__(embs, embu)
        # print(dist.size())
        if output:
            print(dist[0])
        # print(tag)
        return F.cross_entropy(-dist[tag != -1, :], tag[tag != -1])

    def __dist_query__(self, x, y, dim):
        if self.dot:
            return (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist_query__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask == 1].view(-1, Q.size(-1))
        return self.__dist_query__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_nearest_dist__(self, embedding, tag, mask, query, q_mask):
        nearest_dist = []
        S = embedding[mask == 1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == S.size(0)
        # [num_of_query_tokens, num_of_support_tokens]
        dist = self.__batch_dist_query__(S, query, q_mask)
        for label in range(torch.max(tag) + 1):
            nearest_dist.append(torch.max(dist[:, tag == label], 1)[0])
        # [num_of_query_tokens, class_num]
        nearest_dist = torch.stack(nearest_dist, dim=1)
        return nearest_dist

    def __get_circleloss__(self, embedding, tag, mask, query, q_tag, q_mask):
        S = embedding[mask == 1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        q_tag = torch.cat(q_tag, 0)
        assert tag.size(0) == S.size(0)
        # [num_of_query_tokens, num_of_support_tokens]
        dist = self.__batch_dist_query__(S, query, q_mask)
        #print(dist[0])
        gamma, m = 16, 0.25
        op, on, dp, dn = 1+m, -m, 1-m, m
        tmp = []
        #print(dist[0])
        for label in range(torch.max(q_tag) + 1):
            # print(dist.shape, dist[tag == label, :].shape,
            #       dist[tag == label, tag == label].shape)
            disn = dist[q_tag == label, :][:, tag != label]
            disp = dist[q_tag == label, :][:, tag == label]
            N = (torch.exp(gamma*(disn-on).clamp(0)*(disn-dn))).sum(dim=1)
            P = (torch.exp(-gamma*(op-disp).clamp(0)*(disp-dp))).sum(dim=1)
            tmp.append(torch.log(1+N*P).sum())

        tmp = torch.stack(tmp).sum() / q_tag.size(0)
        return tmp

    def __get_circlelossKL__(self, embs, embu, tag, mask):
        embs = embs[mask == 1].view(-1, embs.size(-1))
        embu = embu[mask == 1].view(-1, embu.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embs.size(0)
        dist = self.__batch_dist_support__(embs, embu)
        #print(dist.shape)

        # [num_of_query_tokens, num_of_support_tokens]
        #print(dist[0])
        gamma, m = 16, 0.25
        op, on, dp, dn = 1+m, -m, 1-m, m
        tmp = []
        # print(dist[0])
        for label in range(torch.max(tag) + 1):
            # print(dist.shape, dist[tag == label, :].shape,
            #       dist[tag == label, tag == label].shape)
            disn = dist[tag == label, :][:, tag != label]
            disp = dist[tag == label, :][:, tag == label]
            N = (torch.exp(gamma*(disn-on).clamp(0)*(disn-dn))).sum(dim=1)
            P = (torch.exp(-gamma*(op-disp).clamp(0)*(disp-dp))).sum(dim=1)
            tmp.append(torch.log(1+N*P).sum())

        tmp = torch.stack(tmp).sum() / tag.size(0)
        return tmp

    def forward(self, support, query, output=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(
            support['word'], support['mask']
        )  # [num_sent, number_of_tokens, 768]
        # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask'])
        #support_emb = F.normalize(support_emb, dim=-1)
        #query_emb = F.normalize(query_emb, dim=-1)

        # support_emb = self.drop(support_emb)
        # query_emb = self.drop(query_emb)
        #support_emb_u = self.pool(torch.cat([self.fu(F.relu(support_emb)), self.gu((support_emb))], dim=-1))
        #support_emb_s = self.pool(torch.cat([self.elu(self.fs(F.relu(support_emb))) + 1 + 1e-6, self.elu(self.gs((support_emb))) + 1 + 1e-6], dim=-1))
        support_emb_u = self.fu(F.relu(support_emb))
        support_emb_s = self.elu(self.fs(F.relu(support_emb))) + 1 + 1e-6
        #support_emb_u = support_emb
        #support_emb_s = support_emb
        #query_emb_u = self.fu(query_emb)
        #query_emb_s = self.elu(self.fs(query_emb)) + 1 + 1e-6

        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()
        loss = torch.tensor(0., device=self.fu.weight.device)

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]

            loss = loss + self.__get_NCAloss__(  # TODO:
                support_emb_s[
                    current_support_num: current_support_num + sent_support_num
                ],
                support_emb_u[
                    current_support_num: current_support_num + sent_support_num
                ],
                support['label'][
                    current_support_num: current_support_num + sent_support_num
                ],
                support['text_mask'][
                    current_support_num: current_support_num + sent_support_num
                ],
                skip_O = False
            )

            # loss = loss + self.__get_circleloss__(support_emb[
            #     current_support_num: current_support_num + sent_support_num
            # ],
            #     support['label'][
            #     current_support_num: current_support_num + sent_support_num
            # ],

            #     support['text_mask'][
            #     current_support_num: current_support_num + sent_support_num
            # ],
            #     query_emb[current_query_num: current_query_num +
            #               sent_query_num
            # ],
            #     query['label'][
            #     current_query_num: current_query_num + sent_query_num
            # ],
            #     query['text_mask'][
            #     current_query_num: current_query_num + sent_query_num
            # ],)

            logits.append(
                self.__get_nearest_dist__(
                    support_emb[
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support['label'][
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support['text_mask'][
                        current_support_num: current_support_num + sent_support_num
                    ],
                    query_emb[current_query_num: current_query_num +
                              sent_query_num],
                    query['text_mask'][
                        current_query_num: current_query_num + sent_query_num
                    ],
                )
            )

            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        return loss, logits, pred
