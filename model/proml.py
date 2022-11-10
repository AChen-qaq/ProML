from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch
import math
import util.framework
import sys

sys.path.append('..')



class ProML(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1, proj_dim=32, mix_rate=0.5, topk=1):
        util.framework.FewShotNERModel.__init__(
            self, word_encoder, ignore_index=ignore_index
        )
        self.dot = dot
        self.dim = proj_dim
        self.fu = nn.Linear(768, self.dim)
        self.fu.weight.data /= (self.dim / 32) ** 0.5
        self.fu.bias.data /= (self.dim / 32) ** 0.5

        self.fs = nn.Linear(768, self.dim)
        self.fs.weight.data /= (self.dim / 32) ** 0.5
        self.fs.bias.data /= (self.dim / 32) ** 0.5

        
        self.mix_rate = mix_rate

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
    

    def compute_NCA_loss_by_dist(
        self,
        dists: torch.Tensor,
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

    def support_dist(self, su_list):
        su = torch.cat(su_list, dim=0)
        return self.__batch_dist__(su, su)



    def get_emb_support(self, support):


        cur_support_sents = support['word_tanl']
        cur_support_masks = support['text_mask_tanl']
        cur_support_labels = support['label']


        cur_support_sents_aug = support['word_aug']
        cur_support_masks_aug = support['text_mask_aug']
        cur_support_labels_aug = support['label_aug']

        cur_support_sents_raw = support['word']
        cur_support_masks_raw = support['text_mask']


        e_tanl, e_aug = self.word_encoder.encode_list(cur_support_sents, cur_support_masks), self.word_encoder.encode_list(cur_support_sents_aug, cur_support_masks_aug)


        support_emb = e_aug * self.mix_rate + e_tanl * (1 - self.mix_rate)

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

            
            e_tanl, e_aug = self.word_encoder.encode_list(cur_support_sents, cur_support_masks), self.word_encoder.encode_list(cur_support_sents_aug, cur_support_masks_aug)
            

            support_emb = e_aug * self.mix_rate + e_tanl * (1 - self.mix_rate)

            tag = torch.cat(cur_support_labels)

            
            
            ss, qs = self.elu(self.fs(F.relu(support_emb))) + 1 + 1e-6, self.elu(self.fs(F.relu(query_emb))) + 1 + 1e-6
            su, qu = self.fu(F.relu(support_emb)), self.fu(F.relu(query_emb))

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
            



            current_support_num += sent_support_num
            current_query_num += sent_query_num
        logits = torch.cat(logits, dim=0)
        _, pred = torch.max(logits, dim=1)



        return loss, logits, pred

