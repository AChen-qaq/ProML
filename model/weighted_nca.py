from torch._C import device
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch
import math
import util.framework
import sys
from kmeans_pytorch import kmeans, kmeans_predict

sys.path.append('..')


class GaussianDropout(nn.Module):

    def __init__(self, p: float):
        """
        Multiplicative Gaussian Noise dropout with N(1, p/(1-p))
        It is NOT (1-p)/p like in the paper, because here the
        noise actually increases with p. (It can create the same
        noise as the paper, but with reversed p values)

        Source:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

        :param p: float - determines the the standard deviation of the
        gaussian noise, where sigma = p/(1-p).
        """
        super().__init__()
        assert 0 <= p < 1
        self.t_mean = torch.ones((0,))
        self.shape = ()
        self.p = p
        self.t_std = self.compute_std()

    def compute_std(self):
        return self.p / (1 - self.p)

    def forward(self, t_hidden):
        if self.training and self.p > 0.:
            if self.t_mean.shape != t_hidden.shape:
                self.t_mean = torch.ones_like(
                    input=t_hidden, dtype=t_hidden.dtype, device=t_hidden.device)
            elif self.t_mean.device != t_hidden.device:
                self.t_mean = self.t_mean.to(
                    device=t_hidden.device, dtype=t_hidden.dtype)

            t_gaussian_noise = torch.normal(self.t_mean, self.t_std)
            t_hidden = t_hidden.mul(t_gaussian_noise)
        return t_hidden


class WeightedNCA(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1, proj_dim=32, normalize=True, norm=1, with_dropout=True, train_without_proj=False, eval_with_proj=False):
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

        self.fw = nn.Linear(768, 1)
        self.lu = nn.Sigmoid()
        self.normalize = normalize
        self.norm = norm
        self.drop = nn.Dropout(0.3)
        self.with_dropout = with_dropout
        self.train_without_proj = train_without_proj
        self.eval_with_proj = eval_with_proj

        #self.vote = nn.LSTM(input_size=768, hidden_size=128, num_layers=2, bias=True, batch_first=True, proj_size=1)
        self.vote = nn.Linear(5, 1)
        self.cnt = 0

    def __dist_JS__(self, xw, yw, xu, yu):
        # KL = 0.5 * (
        #     (ys / xs + xs / ys + (xu - yu) ** 2 * (1 / xs + 1 / ys)).sum(dim=-1)
        #     - 64  # TODO:
        # )
        L2 = torch.exp(-((xu - yu)**2).sum(dim=-1))

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
        sw: torch.Tensor,
        tag: torch.Tensor,
        mask: torch.Tensor,
        qu: torch.Tensor,
        qw: torch.Tensor,
        tag_q: torch.Tensor,
        mask_q: torch.Tensor,
    ):
        su = su[mask == 1].view(-1, su.shape[-1])
        sw = sw[mask == 1].view(-1, sw.shape[-1])
        tag = torch.cat(tag, dim=0)
        assert su.shape[0] == tag.shape[0]
        qu = qu[mask_q == 1].view(-1, qu.shape[-1])
        qw = qw[mask_q == 1].view(-1, qw.shape[-1])
        tag_q = torch.cat(tag_q, dim=0)
        assert qu.shape[0] == tag_q.shape[0]

        dists = self.__batch_dist_JS__(
            su, sw, qu, qw
        )  # [num_query_tokens, num_support_tokens]
        self.dists = dists
        losses = []
        alpha, gamma = 0.75, 0.5
        for label in range(torch.max(tag) + 1):
            Xp = (tag == label).sum()
            if Xp.item():
                # if label == 0 or True:
                #     A = torch.topk(dists[tag_q == label, :][:, tag == label], k=5, dim=1).values.mean(dim=1)

                #     B = torch.stack([torch.topk(dists[tag_q == label, :][:, tag == t], k=5, dim=1).values.mean(
                #         dim=1) for t in range(1, torch.max(tag) + 1)], dim=1).sum(dim=1) + A
                #     # #B = dists[tag_q == label, :][:, tag != 0].sum(dim=1) + A
                #     ce = -torch.log(A/B)
                #     losses.append(ce)
                #     # #losses.append(-torch.log(alpha * torch.pow(A, gamma)) + torch.log(B + A)))

                
                A = dists[tag_q == label, :][:, tag == label].sum(dim=1)

                B = dists[tag_q == label, :].sum(dim=1)
                ce = -torch.log(A/B)
                losses.append(ce)                



        loss = torch.cat(losses, dim=0).mean()


        # fst = dists[:, tag == 0].unsqueeze(-1)
        # out, (h, c) = self.lstm(fst)
        # print(fst.shape)
        # logits = self.outproj(h)
        # self.pred_o = torch.argmax(logits, dim=-1)
        # target = (tag_q == 0).cuda().long()

        # oloss = F.cross_entropy(logits, target)

        return loss
    
    def compute_vote_loss(
        self,
        S: torch.Tensor,
        tag: torch.Tensor,
        mask: torch.Tensor,
        Q: torch.Tensor,
        tag_q: torch.Tensor,
        mask_q: torch.Tensor,
        ):
        return 0
        S = S[mask == 1].view(-1, S.shape[-1])
        tag = torch.cat(tag, dim=0)
        dists = (self.__batch_dist__(
            S, Q, mask_q
        ) / 32)
        Q = Q[mask_q == 1].view(-1, Q.shape[-1])
        tag_q = torch.cat(tag_q, dim=0)
        #print(dists)
        

        
        scores = []
        losses = []
        for label_s in range(torch.max(tag) + 1):
            # QQ = Q.unsqueeze(1)
            # SS = S[tag == label_s].unsqueeze(0).expand(len(Q), -1, -1)
            # indices = torch.topk(dists[:, tag == label_s], k=4, dim=1).indices.unsqueeze(-1).expand(-1, -1, 768)
            # # print(indices.shape)
            # # print(S[tag == label_s].unsqueeze(0).expand(len(Q), -1, -1).shape)
            # SS = torch.gather(SS, dim=1, index=indices)
            # inputs = torch.cat([QQ, SS], dim=1)
            # out, _ = self.vote(inputs.detach())
            # # print(out[:,-1,0].shape)
            # scores.append(torch.exp(out[:,-1,0]))
            #print(self.vote(torch.topk(dists[:, tag == label_s], k=5, dim=1).values).shape)
            #scores.append(torch.exp(self.vote(torch.topk(dists[:, tag == label_s], k=5, dim=1).values)))
            scores.append( torch.exp(torch.topk(dists[:, tag == label_s], k=5, dim=1).values.mean(dim=1)) )
        scores = torch.stack(scores, dim=-1)
        #print(scores.shape)
        for label_q in range(torch.max(tag) + 1):
            losses.append( -torch.log( scores[tag_q == label_q, label_q]/( scores[tag_q == label_q, :].sum(dim=1) ) ) )

        return torch.cat(losses, dim=0).mean()

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
        dist = (self.__batch_dist__(
            S, query, q_mask
        ) / 32)  # [num_of_query_tokens, num_of_support_tokens]
        for label in range(torch.max(tag) + 1):
            nearest_dist.append(torch.topk(dist[:, tag == label], k=5, dim=1).values.mean(dim=1))
        nearest_dist = torch.stack(
            nearest_dist, dim=1
        )  # [num_of_query_tokens, class_num]

        # nearest_dist[self.pred_o == 1, 0] = torch.inf
        return nearest_dist

    def __get_vote_scores__(self, embedding, tag, mask, query, q_mask):
        scores = []
        S = embedding[mask == 1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == S.size(0)
        Q = query[q_mask == 1].view(-1, query.size(-1))

        dists = self.__batch_dist__(
            S, query, q_mask
        )/32  # [num_of_query_tokens, num_of_support_tokens]

        for label_s in range(torch.max(tag) + 1):
            # QQ = Q.unsqueeze(1)
            # SS = S[tag == label_s].unsqueeze(0).expand(len(Q), -1, -1)
            # indices = torch.topk(dists[:, tag == label_s], k=4, dim=1).indices.unsqueeze(-1).expand(-1, -1, 768)
            # # print(indices.shape)
            # # print(S[tag == label_s].unsqueeze(0).expand(len(Q), -1, -1).shape)
            # SS = torch.gather(SS, dim=1, index=indices)
            # inputs = torch.cat([QQ, SS], dim=1)
            # out, _ = self.vote(inputs)
            # scores.append(out[:, -1, 0])
            scores.append(self.vote(torch.topk(dists[:, tag == label_s], k=5, dim=1).values))


            
        scores = torch.stack(
            scores, dim=1
        )  # [num_of_query_tokens, class_num]

        # nearest_dist[self.pred_o == 1, 0] = torch.inf
        return scores

    def get_emb(self, support, query):
    
        support_emb = self.word_encoder(support['word'], support['mask'])
        query_emb = self.word_encoder(query['word'], query['mask'])

        # inputs = torch.cat([support['word'], query['word']], dim=0)
        # masks = torch.cat([support['mask'], query['mask']], dim=0)
        # print(support['word'])
        # __import__('pdb').set_trace()
        # embs = self.word_encoder(inputs, masks)
        # support_emb, query_emb = embs[:len(
        #     support['word'])], embs[len(support['word']):]

        # [num_sent, num_tokens, hidden_size]
        support_u = self.drop(support_emb) if self.with_dropout else support_emb
        query_u = self.drop(query_emb) if self.with_dropout else query_emb

        if not self.train_without_proj:
            support_u = (self.fu(F.relu(support_u)))
            query_u = (self.fu(F.relu(query_u)))

        if self.normalize:
            support_emb = F.normalize(support_emb) * self.norm
            query_emb = F.normalize(query_emb) * self.norm
            support_u = F.normalize(support_u) * self.norm
            query_u = F.normalize(query_u) * self.norm
        else:
            support_u = support_u * self.norm
            query_u = query_u * self.norm

        return support_emb, query_emb, support_u, query_u

    def forward(self, support, query):
        support_emb, query_emb, support_u, query_u = self.get_emb(support, query)
        support_emb_dropped, query_emb_dropped = (self.drop(support_emb)), (self.drop(query_emb))
        # support_w = self.lu(self.fw(F.relu(support_emb)))
        # query_w = self.lu(self.fw(F.relu(query_emb)))
        support_w, query_w = support_u, query_u

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
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support_w[
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support['label'][
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support['text_mask'][
                        current_support_num: current_support_num + sent_support_num
                    ],
                    query_u[current_query_num: current_query_num + sent_query_num],
                    query_w[current_query_num: current_query_num + sent_query_num],
                    query['label'][
                        current_query_num: current_query_num + sent_query_num
                    ],
                    query['text_mask'][
                        current_query_num: current_query_num + sent_query_num
                    ],
                )
                +
                self.compute_vote_loss(
                    support_emb_dropped[
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support['label'][
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support['text_mask'][
                        current_support_num: current_support_num + sent_support_num
                    ],
                    query_emb_dropped[current_query_num: current_query_num + sent_query_num],
                    query['label'][
                        current_query_num: current_query_num + sent_query_num
                    ],
                    query['text_mask'][
                        current_query_num: current_query_num + sent_query_num
                    ],
                )
            ) / bsz
            logits.append(
                self.__get_nearest_dist__(
                    support_u[
                        current_support_num: current_support_num + sent_support_num
                    ] if self.eval_with_proj else
                    support_emb[
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support['label'][
                        current_support_num: current_support_num + sent_support_num
                    ],
                    support['text_mask'][
                        current_support_num: current_support_num + sent_support_num
                    ],
                    query_u[current_query_num: current_query_num + sent_query_num]
                    if self.eval_with_proj else
                    query_emb[current_query_num: current_query_num +
                              sent_query_num],
                    query['text_mask'][
                        current_query_num: current_query_num + sent_query_num
                    ],
                )
            )
            current_support_num += sent_support_num
            current_query_num += sent_query_num
        logits = torch.cat(logits, dim=0)
        _, pred = torch.max(logits, dim=1)



        return loss, logits, pred
