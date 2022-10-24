from torch._C import device
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch
import math
import util.framework
import sys
from kmeans_pytorch import kmeans, kmeans_predict
import random

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


class PromptNCA(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1, proj_dim=32, normalize=True, norm=1, with_dropout=True, train_without_proj=False, eval_with_proj=False, mix_rate=0.5, topk=1):
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

        self.fs = nn.Linear(768, self.dim)
        self.fs.weight.data /= (self.dim / 32) ** 0.5
        self.fs.bias.data /= (self.dim / 32) ** 0.5

        # self.mix_rate = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        
        self.mix_rate = mix_rate

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
        self.elu = nn.ELU()
        self.topk = topk
    

    def __dist_JS__(self, xs, ys, xu, yu):
        KL = 0.5 * (
            (ys / xs + xs / ys + (xu - yu) ** 2 * (1 / xs + 1 / ys)).sum(dim=-1)
            - self.dim * 2
        )
        return KL

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
        tag: torch.Tensor,
        qu: torch.Tensor,
        tag_q: torch.Tensor,
    ):

        dists = self.merge_dist(
            su, qu
        )  # [num_query_tokens, num_support_tokens]
        dists = torch.exp(dists)


        # print(dists)
        losses = []
        for label in range(torch.max(tag) + 1):
            Xp = (tag == label).sum()
            Yp = (tag_q == label).sum()
            if Xp.item() and Yp.item():
                
                A = dists[tag_q == label, :][:, tag == label].sum(dim=1)
                B = dists[tag_q == label, :].sum(dim=1)
                ce = -torch.log(A/B)
                # losses.append(torch.where(torch.isnan(ce), torch.full_like(ce, 0), ce))
                losses.append(ce)

        

        # dists = self.support_dist(su)
        # dists = torch.exp(dists)

        # for label in range(torch.max(tag) + 1):
        #     Xp = (tag == label).sum()
        #     if Xp.item():
        #         A = dists[tag == label, :][:, tag == label].sum(dim=1)
        #         B = dists[tag == label, :].sum(dim=1)
        #         ce = -torch.log(A/B)
        #         losses.append(torch.where(torch.isnan(ce), torch.full_like(ce, 0), ce))

        # print((torch.cat(losses, dim=0) == 0).sum() / len(losses))
        loss = 0 if len(losses)==0 else torch.cat(losses, dim=0).mean()
        del dists, losses
        return loss
    

    def compute_NCA_loss_by_dist(
        self,
        dists,
        tag: torch.Tensor,
        tag_q: torch.Tensor,
    ):


        # print(dists)
        losses = []
        for label in range(torch.max(tag) + 1):
            Xp = (tag == label).sum()
            Yp = (tag_q == label).sum()
            if Xp.item() and Yp.item():
                
                A = dists[tag_q == label, :][:, tag == label].sum(dim=1)
                B = dists[tag_q == label, :].sum(dim=1)
                ce = -torch.log(A/B)
                losses.append(torch.where(torch.isfinite(ce), ce, torch.full_like(ce, 0)))
                # losses.append(ce)

        loss = 0 if len(losses)==0 else torch.cat(losses, dim=0).mean()
        del losses
        return loss

    def compute_vote_loss(self, dists, tag, tag_q):
        scores = []
        losses = []
        for label_s in range(torch.max(tag) + 1):
            scores.append( torch.exp(torch.topk(dists[:, tag == label_s], k=1, dim=1).values.mean(dim=1)) )
        scores = torch.stack(scores, dim=-1)
        #print(scores.shape)
        for label_q in range(torch.max(tag) + 1):
            ce = -torch.log( scores[tag_q == label_q, label_q]/( scores[tag_q == label_q, :].sum(dim=1) ) )
            losses.append(torch.where(torch.isnan(ce), torch.full_like(ce, 0), ce))
        
        
        loss = 0 if len(losses)==0 else torch.cat(losses, dim=0).mean()

        return loss

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_nearest_dist__(self, dist, tag):

        nearest_dist = []
        for label in range(torch.max(tag) + 1):
            nearest_dist.append(torch.topk(dist[:, tag == label], k=self.topk, dim=1).values.mean(dim=1))
        nearest_dist = torch.stack(
            nearest_dist, dim=1
        )  # [num_of_query_tokens, class_num]

        # nearest_dist[self.pred_o == 1, 0] = torch.inf
        return nearest_dist

    def __get_vote_scores__(self, dists, tag, tag_q):

        for label_s in range(torch.max(tag) + 1):
            scores.append(self.vote(torch.topk(dists[:, tag == label_s], k=1, dim=1).values))


            
        scores = torch.stack(
            scores, dim=1
        )  # [num_of_query_tokens, class_num]

        # nearest_dist[self.pred_o == 1, 0] = torch.inf
        return scores

    def merge_dist(self, su_list, qu_list):
        # if self.training:
        #     mask_rate = 0.9
        #     mask = torch.distributions.Categorical(
        #                     torch.tensor([1-mask_rate,mask_rate], device='cuda')).sample((768, )).bool()
        # # support['word'] = support['word'].masked_fill(mask.masked_fill(support['token_mask']==0, False), mask_id)
        #     dists = []
        #     for su, qu in zip(su_list, qu_list):
        #         dists.append(self.__batch_dist__(su[:, mask == 1], qu[:, mask == 1]))
        # else:
        #     dists = []
        #     for su, qu in zip(su_list, qu_list):
        #         dists.append(self.__batch_dist__(su, qu))

        dists = []
        # for su, qu in zip(su_list, qu_list):
        #     dists.append(self.__batch_dist__(self.drop(su), self.drop(qu)))

        for support_emb, query_emb in zip(su_list, qu_list):
            ss, qs = self.elu(self.fs(F.relu(support_emb))) + 1 + 1e-6, self.elu(self.fs(F.relu(query_emb))) + 1 + 1e-6
            su, qu = self.fu(F.relu(support_emb)), self.fu(F.relu(query_emb))
            dists.append(torch.exp(-self.__batch_dist_JS__(su, ss, qu, qs)))
        
        return torch.cat(dists, dim=1)


    def support_dist(self, su_list):
        su = torch.cat(su_list, dim=0)
        return self.__batch_dist__(su, su)


    def postnet(self, su_list, qu_list):
        for i in range(len(su_list)):
            su_list[i] = self.fu(F.relu(self.drop(su_list[i])))

        for i in range(len(qu_list)):
            qu_list[i] = self.fu(F.relu(self.drop(qu_list[i])))

    def _forward(self, support, query):

        logits = []
        current_support_num = 0
        current_query_num = 0
        bsz = len(support['sentence_num'])
        loss = torch.tensor(0.0, device=self.fu.weight.device)
        for sent_support_num, sent_query_num in zip(
            support['sentence_num'], query['sentence_num']
        ):
            cur_support_sents = support['word_tanl'][current_support_num: current_support_num + sent_support_num]
            cur_support_masks = support['text_mask_tanl'][current_support_num: current_support_num + sent_support_num]
            cur_support_labels = support['label'][current_support_num: current_support_num + sent_support_num]


            cur_query_sents = query['word'][current_query_num: current_query_num + sent_query_num]
            cur_query_masks = query['text_mask'][current_query_num: current_query_num + sent_query_num]
            cur_query_labels = query['label'][current_query_num: current_query_num + sent_query_num]


            # cur_support_sents_raw = support['word'][current_support_num: current_support_num + sent_support_num]
            # cur_support_masks_raw = support['text_mask'][current_support_num: current_support_num + sent_support_num]

            cur_logits = torch.zeros(0, torch.max(torch.cat(cur_support_labels))+1, device = 'cuda')
            # print(cur_query_sents.shape)
            for i in range(sent_query_num):
                query_sent, query_mask, tag_q = cur_query_sents[i], cur_query_masks[i], cur_query_labels[i]
                # print(query_sent.shape, query_mask.shape)

                embs_tanl, support_text_mask_tanl = self.word_encoder.encode_multisupport_singlequery_tanl(cur_support_sents, cur_support_masks, query_sent, query_mask)
                # print(inputs.shape, mask.shape)
                # embs = self.word_encoder(inputs, mask)
                tag = torch.cat(cur_support_labels)
                query_emb = [embs_tanl[i, 1:1+len(tag_q), :] for i in range(len(embs_tanl))]
                # query_emb = embs[:, 1:1+len(tag_q), :].view(-1, embs.size(-1))
                # support_emb = embs[support_text_mask == 1].view(-1, embs.size(-1))
                support_emb = [embs_tanl[i, support_text_mask_tanl[i] == 1, :] for i in range(len(embs_tanl))]
                # print(len(support_emb), len(tag))
                # print(support_emb)

                # query_emb, support_emb = self.drop(query_emb), self.drop(support_emb)
                # qu, su = self.fu(F.relu(query_emb)), self.fu(F.relu(support_emb))
                # qu, su = self.drop(qu), self.drop(su)
                # assert len(support_emb) == len(tag) and len(query_emb) == len(tag_q)



                dists = self.merge_dist(support_emb, query_emb)
                # self.postnet(support_emb, query_emb)
                

                # dist = torch.exp(self.__batch_dist__(su, qu))
                loss += self.compute_NCA_loss_by_dist(dists, tag, tag_q) / (sent_query_num * bsz)

                # dists = torch.cat([dists, dists_], dim=1)
                cur_logits = torch.cat([cur_logits, self.__get_nearest_dist__(dists.detach(), tag)], dim=0)

                # torch.cuda.empty_cache()


            logits.append(cur_logits)


            current_support_num += sent_support_num
            current_query_num += sent_query_num
        logits = torch.cat(logits, dim=0)
        _, pred = torch.max(logits, dim=1)



        return loss, logits, pred

    def get_emb_support(self, support):
        # cur_query_sents = query['word']
        # cur_query_masks = query['text_mask']
        # cur_query_labels = query['label']

        # cur_query_sents_aug = query['word_aug']
        # cur_query_masks_aug = query['text_mask_aug']
        # cur_query_labels_aug = query['label_aug']
        
        

        # query_emb = self.word_encoder.encode_list(cur_query_sents_aug, cur_query_masks_aug)

        # return query_emb



        cur_support_sents = support['word_tanl']
        cur_support_masks = support['text_mask_tanl']
        cur_support_labels = support['label']


        cur_support_sents_aug = support['word_aug']
        cur_support_masks_aug = support['text_mask_aug']
        cur_support_labels_aug = support['label_aug']

        cur_support_sents_raw = support['word']
        cur_support_masks_raw = support['text_mask']


        e_tanl, e_aug = self.word_encoder.encode_list(cur_support_sents, cur_support_masks), self.word_encoder.encode_list(cur_support_sents_aug, cur_support_masks_aug)


        support_emb = e_tanl * self.mix_rate + e_aug * (1 - self.mix_rate)

        return support_emb
    
    def get_emb_query(self, query):
        cur_query_sents = query['word']
        cur_query_masks = query['text_mask']
        cur_query_labels = query['label']

        cur_query_sents_aug = query['word_aug']
        cur_query_masks_aug = query['text_mask_aug']
        cur_query_labels_aug = query['label_aug']
        
        

        query_emb = self.word_encoder.encode_list(cur_query_sents_aug, cur_query_masks_aug)

        return query_emb



    def forward(self, support, query, it=None):

        logits = []
        current_support_num = 0
        current_query_num = 0
        bsz = len(support['sentence_num'])
        loss = torch.tensor(0.0, device=self.fu.weight.device)
        for sent_support_num, sent_query_num in zip(
            support['sentence_num'], query['sentence_num']
        ):
            cur_support_sents = support['word_tanl'][current_support_num: current_support_num + sent_support_num]
            cur_support_masks = support['text_mask_tanl'][current_support_num: current_support_num + sent_support_num]
            cur_support_labels = support['label'][current_support_num: current_support_num + sent_support_num]


            cur_support_sents_aug = support['word_aug'][current_support_num: current_support_num + sent_support_num]
            cur_support_masks_aug = support['text_mask_aug'][current_support_num: current_support_num + sent_support_num]
            cur_support_labels_aug = support['label_aug'][current_support_num: current_support_num + sent_support_num]

            cur_support_sents_raw = support['word'][current_support_num: current_support_num + sent_support_num]
            cur_support_masks_raw = support['text_mask'][current_support_num: current_support_num + sent_support_num]



            cur_query_sents = query['word'][current_query_num: current_query_num + sent_query_num]
            cur_query_masks = query['text_mask'][current_query_num: current_query_num + sent_query_num]
            cur_query_labels = query['label'][current_query_num: current_query_num + sent_query_num]

            cur_query_sents_aug = query['word_aug'][current_query_num: current_query_num + sent_query_num]
            cur_query_masks_aug = query['text_mask_aug'][current_query_num: current_query_num + sent_query_num]
            cur_query_labels_aug = query['label_aug'][current_query_num: current_query_num + sent_query_num]
            
            

            query_emb = self.word_encoder.encode_list(cur_query_sents_aug, cur_query_masks_aug)
            tag_q = torch.cat(cur_query_labels)

            # edited!
            if self.mix_rate < 0:
                e_tanl, e_aug = self.word_encoder.encode_list(cur_support_sents, cur_support_masks), self.word_encoder.encode_list(cur_support_sents_raw, cur_support_masks_raw)
            else:
                e_tanl, e_aug = self.word_encoder.encode_list(cur_support_sents, cur_support_masks), self.word_encoder.encode_list(cur_support_sents_aug, cur_support_masks_aug)
            # support_emb = torch.max(torch.stack([e_raw, e_tanl], dim=0), dim=0)[0]
            # support_emb = e_raw * (1 - self.mix_rate) + e_tanl * self.mix_rate

            # if it is not None and it % 10 == 0:
            #     # self.mix_rate = 0.5 + random.random() * 0.4
            #     self.mix_rate = 1 - it / 10000
            # if it is None:
            #     self.mix_rate = 0.5

            # support_emb = e_raw * self.mix_rate + e_tanl * (1 - self.mix_rate)
            # support_emb = torch.cat([support_emb, e_aug], dim=0)
            if self.mix_rate < 0:
                support_emb = e_tanl * -self.mix_rate + e_aug * (1 + self.mix_rate)
            else:
                support_emb = e_tanl * self.mix_rate + e_aug * (1 - self.mix_rate)

            # support_emb = self.word_encoder.encode_list(cur_support_sents_aug, cur_support_masks_aug)

            # print(e_raw.shape, e_tanl.shape, e_aug.shape)
            # support_emb = F.softmax(torch.stack([e_raw, e_tanl], dim=0), dim=0)
            # print(support_emb.shape)
            # support_emb = self.mix(torch.cat([e_raw, e_tanl], dim=-1))
            # support_emb, query_emb = self.drop(support_emb), self.drop(query_emb)
            # support_emb = self.word_encoder.encode_list(cur_support_sents, cur_support_masks)

            tag = torch.cat(cur_support_labels)
            # tag_aug = torch.cat(cur_support_labels_aug)
            # tag = torch.cat([tag, tag_aug])

            # print(len(support_emb), len(tag))
            
            # support_emb = torch.cat([e_raw, e_tanl])
            # tag = torch.cat(cur_support_labels)
            # tag = torch.cat([tag, tag])
            # if it is not None and it <= 1000:
            #     loss += self.compute_vote_loss(self.__batch_dist__(F.relu(e_tanl), F.relu(query_emb)) / 32, tag, tag_q) / bsz * 0.01
            
            
            ss, qs = self.elu(self.fs(F.relu(support_emb))) + 1 + 1e-6, self.elu(self.fs(F.relu(query_emb))) + 1 + 1e-6
            su, qu = self.fu(F.relu(support_emb)), self.fu(F.relu(query_emb))
            # su, qu =self.drop(support_emb) / 8, self.drop(query_emb) / 8
            if it == -1:
                dist = torch.exp(-self.__batch_dist_JS__(su, ss, qu, qs))
                # dist = self.__batch_dist__(support_emb, query_emb)
                dist = dist.masked_fill( torch.eye(len(su)).cuda().bool(), 0)
                loss += self.compute_NCA_loss_by_dist(dist, tag, tag_q) / bsz
        
                logits.append(self.__get_nearest_dist__(self.__batch_dist__(support_emb, query_emb).detach().masked_fill( torch.eye(len(su)).cuda().bool(), -torch.inf), tag))

            else:
                dist = torch.exp(-self.__batch_dist_JS__(su, ss, qu, qs))
                loss += self.compute_NCA_loss_by_dist(dist, tag, tag_q) / bsz
            
                logits.append(self.__get_nearest_dist__(self.__batch_dist__(support_emb, query_emb).detach(), tag))
            # dist = torch.exp(self.__batch_dist__(su, qu))
            



            current_support_num += sent_support_num
            current_query_num += sent_query_num
        logits = torch.cat(logits, dim=0)
        _, pred = torch.max(logits, dim=1)



        return loss, logits, pred

