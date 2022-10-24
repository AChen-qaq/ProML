import sys
sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class KLnnshot(util.framework.FewShotNERModel):
    
    def __init__(self,word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        # self.d = 32
        self.d = 32
        self.fu = nn.Linear(768, self.d)
        self.fs = nn.Linear(768, self.d)
        self.elu = nn.ELU()

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1))
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)
    
    
    def __KL_dist__(self, xu, xs, yu, ys):
        KL = 0.5 * ((ys / xs + xs / ys + (xu - yu)
                    ** 2 * (1 / xs + 1 / ys)).sum(dim=-1) - self.d * 2)
        #KL = F.sigmoid(KL)*10
        # print(KL.shape, xu.shape, xs.shape, yu.shape, ys.shape)
        return -KL

    def __batch_KL_dist__(self, S_u, S_s, Q_u, Q_s, q_mask):
        # assert Q.size()[:2] == q_mask.size()
        Q_u = Q_u[q_mask==1].view(-1, Q_u.size(-1))
        Q_s = Q_s[q_mask==1].view(-1, Q_s.size(-1))
        return self.__KL_dist__(S_u.unsqueeze(0), S_s.unsqueeze(0), Q_u.unsqueeze(1), Q_s.unsqueeze(1))


    def __get_KL_dist__(self, embedding, tag, mask, query, q_mask):
        
        # support_emb_u = embedding
        support_emb_u = self.fu(F.relu(embedding))
        support_emb_s = self.elu(self.fs(F.relu(embedding))) + 1 + 1e-6
        # query_emb_u = query
        query_emb_u = self.fu(F.relu(query))
        query_emb_s = self.elu(self.fs(F.relu(query))) + 1 + 1e-6

        S_u = support_emb_u[mask==1].view(-1, support_emb_u.size(-1))
        S_s = support_emb_s[mask==1].view(-1, support_emb_s.size(-1))
        
        tag = torch.cat(tag, 0)
        assert tag.size(0) == S_u.size(0)
        assert tag.size(0) == S_s.size(0)
        dist = self.__batch_KL_dist__(S_u, S_s, query_emb_u, query_emb_s, q_mask) # [num_of_query_tokens, num_of_support_tokens]
        return dist

    def __get_nearest_dist__(self, embedding, tag, mask, query, q_mask):
        nearest_dist = []
        S = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == S.size(0)
        dist = self.__batch_dist__(S, query, q_mask) # [num_of_query_tokens, num_of_support_tokens]
        for label in range(torch.max(tag)+1):
            nearest_dist.append(torch.max(dist[:,tag==label], 1)[0])
        nearest_dist = torch.stack(nearest_dist, dim=1) # [num_of_query_tokens, class_num]
        return nearest_dist

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        KLlogits = []
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]

            # Calculate nearest distance to single entity in each class in support set
            KLlogits.append(self.__get_KL_dist__(support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num],
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num]))
                
            logits.append(self.__get_nearest_dist__(support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num],
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num]))
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        KLlogits = torch.cat(KLlogits, 0)
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        return KLlogits, logits, pred
        # return logits, logits, pred

    
    
    
