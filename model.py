import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet,  Encoder, CrossAttention, SelfAttentionAudio, SelfAttentionVision


class MemoryBank(nn.Module):
    def __init__(self, embedding_dim, memory_size=0):
        """
        初始化Memory Bank。

        参数:
        - embedding_dim: 特征表示的维度。
        - memory_size: Memory Bank中存储的最大样本数量。
        """
        super(MemoryBank, self).__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.memory_bank_ta = torch.empty((self.memory_size, self.embedding_dim),device=torch.device("cuda"))
        self.memory_bank_tv = torch.empty((self.memory_size, self.embedding_dim),device=torch.device("cuda"))
        self.full = False
        self.memory_cursor = 0

    def update(self, embeddings_ta_neg,embeddings_tv_neg):
        """
        更新Memory Bank。

        参数:
        - embeddings: 形状为(batch_size, embedding_dim)的张量，表示当前批次样本的特征。
        """
        batch_size = embeddings_ta_neg.size(0)
        embedding_dim = embeddings_tv_neg.size(1)
        if self.memory_cursor + batch_size > self.memory_size:
            # 如果超出范围，你可能需要重置 memory_cursor 或者扩展 memory_bank 的大小
            self.memory_cursor = 0
        # 将新样本添加到Memory Bank的末尾
        self.memory_bank_ta[self.memory_cursor:self.memory_cursor + batch_size,:] = embeddings_ta_neg
        self.memory_bank_tv[self.memory_cursor:self.memory_cursor + batch_size,:] = embeddings_tv_neg
        # 更新计数器
        self.memory_cursor += batch_size
        # 如果计数器超过Memory Bank的大小，则重置为0
        if self.memory_cursor >= self.memory_size:
            self.full = True
            self.memory_cursor = (self.memory_cursor + batch_size) % self.memory_size

    def get_all(self):
        """
        返回Memory Bank中的所有样本表示。
        """
        return self.memory_bank_ta,self.memory_bank_tv

    def get_full(self):
        return self.full

class UCMIB(nn.Module):

    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp

        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.uni_text_enc = LanguageEmbeddingLayer(hp)  # BERT Encoder
        self.uni_visual_enc = RNNEncoder(  # 视频特征提取
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.uni_acoustic_enc = RNNEncoder(  # 音频特征提取
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        # For MI maximization   互信息最大化
        # Modality Mutual Information Lower Bound（MMILB）
        self.mi_tv = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_vout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_aout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        if hp.add_va:  # 一般是tv和ta   若va也要MMILB
            self.mi_va = MMILB(
                x_size=hp.d_vout,
                y_size=hp.d_aout,
                mid_activation=hp.mmilb_mid_activation,
                last_activation=hp.mmilb_last_activation
            )

        # CPC MI bound   d_prjh是什么？？？
        self.cpc_zt = CPC(
            x_size=hp.d_tout,  # to be predicted  各个模态特征提取后得到的维度
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size=hp.d_vout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size=hp.d_aout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )

        self.memory_bank = MemoryBank(embedding_dim=hp.model_dim_cross,memory_size=hp.batch_size*hp.ratio)

        self.uni_audio_encoder = SelfAttentionAudio(hp, d_in=hp.d_ain, d_model=hp.model_dim_self,
                                                   nhead=hp.num_heads_self,
                                                   dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_self,
                                                   num_layers=hp.num_layers_self)


        self.uni_vision_encoder = SelfAttentionVision(hp, d_in=hp.d_vin, d_model=hp.model_dim_self,
                                                    nhead=hp.num_heads_self,
                                                    dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_self,
                                                    num_layers=hp.num_layers_self)

        self.encoder_ta = nn.Sequential(nn.Linear(hp.model_dim_cross, hp.d_prjh),
                                     nn.Tanh())
        self.encoder_tv = nn.Sequential(nn.Linear(hp.model_dim_cross, hp.d_prjh),
                                     nn.Tanh())

        self.fc_mu_ta  = nn.Linear(hp.d_prjh, hp.model_dim_cross)
        self.fc_std_ta = nn.Linear(hp.d_prjh, hp.model_dim_cross)

        self.fc_mu_tv  = nn.Linear(hp.d_prjh, hp.model_dim_cross)
        self.fc_std_tv = nn.Linear(hp.d_prjh, hp.model_dim_cross)

        self.ta_head =  SubNet(in_size=hp.model_dim_cross,hidden_size=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)
        self.tv_head = SubNet(in_size=hp.model_dim_cross,hidden_size=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)


        if self.hp.dataset == "sims":
            self.audio_classifer = SubNet(in_size=hp.d_aout, hidden_size=hp.d_aout*2,
                                    n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_ain]
            self.vision_classifer = SubNet(in_size=hp.d_vout, hidden_size=hp.d_vout*2,
                                     n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_vin]
            self.text_classifer = SubNet(in_size=hp.d_tin, hidden_size=hp.d_tin*2,
                                   n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_tin]


        self.ta_ps_mapping = nn.Sequential(nn.Linear(3, hp.d_prjh//2),nn.Tanh(),nn.Linear(hp.d_prjh//2,hp.n_class),nn.ReLU(inplace=True))
        self.ta_pn_mapping = nn.Sequential(nn.Linear(3, hp.d_prjh//2),nn.Tanh(),nn.Linear(hp.d_prjh//2,hp.n_class),nn.ReLU(inplace=True))
        self.tv_ps_mapping = nn.Sequential(nn.Linear(3, hp.d_prjh//2),nn.Tanh(),nn.Linear(hp.d_prjh//2,hp.n_class),nn.ReLU(inplace=True))
        self.tv_pn_mapping = nn.Sequential(nn.Linear(3, hp.d_prjh//2),nn.Tanh(),nn.Linear(hp.d_prjh//2,hp.n_class),nn.ReLU(inplace=True))

        # 用MULT融合 每个模块的输出都是[bs,query_length,model_dim_cross]
        self.ta_cross_attn=CrossAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_ain,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross)
        self.tv_cross_attn=CrossAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_vin,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross)

        self.fusion_mlp_for_regression = SubNet(in_size=hp.model_dim_cross*2,hidden_size=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)
        self.ideal_fusion_mlp_for_regression = SubNet(in_size=hp.model_dim_cross*2,hidden_size=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)


    def gen_mask(self, a, length=None):
        if length is None:
            msk_tmp = torch.sum(a, dim=-1)
            # 特征全为0的时刻加mask
            mask = (msk_tmp == 0)
            return mask
        else:
            b = a.shape[0]
            l = a.shape[1]
            msk = torch.ones((b, l))
            x = []
            y = []
            for i in range(b):
                for j in range(length[i], l):
                    x.append(i)
                    y.append(j)
            msk[x, y] = 0
            return (msk == 0)


    def encode_ta(self, x):
        """
        x : [batch_size,cross_dim]
        """
        x = self.encoder_ta(x)
        return self.fc_mu_ta(x), F.softplus(self.fc_std_ta(x)-5, beta=1)

    def encode_tv(self, x):
        """
        x : [batch_size,cross_dim]
        """
        x = self.encoder_tv(x)
        return self.fc_mu_tv(x), F.softplus(self.fc_std_tv(x)-5, beta=1)

    def normalize(self,*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def transpose(self,x):
        return x.transpose(-2, -1)

    def uncertain_contrastive_learning(self,query,positive_keys,negative_keys=None,temperature=0.1):
        query, positive_keys,negative_keys= self.normalize(query, positive_keys,negative_keys)
        positive_logit = torch.sum(query * positive_keys, dim=1, keepdim=True)
        negative_logits = query @ self.transpose(negative_keys)
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits / temperature, labels), logits

    def calculate_prob(self,ta_pred,tv_pred,all_pred,y,threshold,type="ta_ps"):
        assert ta_pred.shape == tv_pred.shape == all_pred.shape == y.shape
        bs = ta_pred.size(0)
        if type=="ta_pn":
            diff_1 = torch.abs(all_pred - y)
            diff_2 = torch.abs(tv_pred - y)
            common_mask = (diff_1 < threshold) & (diff_2 > threshold)
            return common_mask.long()
        elif type=="ta_ps":
            random_guessing = 6 * torch.rand(1, device=torch.device("cuda:0")) - 3
            if self.hp.dataset == "ch-sims":
                random_guessing/=3
            diff_1 = torch.abs(ta_pred - y)
            diff_2 = torch.abs(random_guessing - y)
            common_mask = (diff_1 < threshold) & (diff_2 > threshold)
            return common_mask.long()
        elif type=="tv_pn":
            diff_1 = torch.abs(all_pred - y)
            diff_2 = torch.abs(ta_pred - y)
            common_mask = (diff_1 < threshold) & (diff_2 > threshold)
            return common_mask.long()
        elif type=="tv_ps":
            random_guessing = 6 * torch.rand(1, device=torch.device("cuda:0")) - 3
            if self.hp.dataset == "ch-sims":
                random_guessing/=3
            diff_1 = torch.abs(tv_pred - y)
            diff_2 = torch.abs(random_guessing - y)
            common_mask = (diff_1 < threshold) & (diff_2 > threshold)
            return common_mask.long()
        else:
            raise TypeError

    def calculate_prob_acc2(self,ta_pred,tv_pred,all_pred,y,threshold=None,type="ta_ps"):
        assert ta_pred.shape == tv_pred.shape == all_pred.shape
        bs = ta_pred.size(0)
        if type=="ta_pn":
            diff_1 = (torch.argmax(all_pred, dim=1) == y)
            diff_2 = (torch.argmax(tv_pred, dim=1) == y)
            common_mask = diff_1 & diff_2
            return common_mask.long().view(-1,1)
        elif type=="ta_ps":
            rand_value = torch.rand(1, device=torch.device("cuda:0"))
            random_guessing = (rand_value < 0.5).int()
            diff_1 = (torch.argmax(ta_pred, dim=1) == y)
            diff_2 = (random_guessing == y)
            common_mask = diff_1 & diff_2
            return common_mask.long().view(-1,1)
        elif type=="tv_pn":
            diff_1 = (torch.argmax(all_pred, dim=1) == y)
            diff_2 = (torch.argmax(ta_pred, dim=1) == y)
            common_mask = diff_1 & diff_2
            return common_mask.long().view(-1,1)
        elif type=="tv_ps":
            rand_value = torch.rand(1, device=torch.device("cuda:0"))
            random_guessing = (rand_value < 0.5).int()
            diff_1 = (torch.argmax(tv_pred, dim=1) == y)
            diff_2 = (random_guessing == y)
            common_mask = diff_1 & diff_2
            return common_mask.long().view(-1,1)
        else:
            raise TypeError

    def kl_divergence(self,p,q):
        # 计算方差
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss

    def reparameterize(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std,device=std.device)
        return mu + std*eps

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None,
                mem=None,v_mask=None,a_mask=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        sentences: torch.Size([0, 32])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        For Bert input, the length of text is "seq_len + 2"
        """
        with torch.no_grad():
            maskT = (bert_sent_mask == 0)
            maskV = self.gen_mask(visual.transpose(0,1),v_len)
            maskA = self.gen_mask(acoustic.transpose(0,1),a_len)

        enc_word= self.uni_text_enc(sentences, bert_sent, bert_sent_type,bert_sent_mask)  # 32*50*768 (batch_size, seq_len, emb_size)
        text_trans = enc_word.transpose(0, 1)  # torch.Size([50, 32, 768]) (seq_len, batch_size,emb_size)

        acoustic = self.uni_audio_encoder(acoustic)  # [seq_len,bs,dim] 自注意力
        visual = self.uni_vision_encoder(visual)  # [seq_len,bs,dim] 自注意力


        vision_trans = visual
        audio_trans = acoustic
        # 2. 跨模态注意力部分
        cross_tv = self.tv_cross_attn(text_trans, vision_trans,Tmask=maskT,Vmask=maskV).mean(dim=0)
        cross_ta = self.ta_cross_attn(text_trans, audio_trans,Tmask=maskT,Amask=maskA).mean(dim=0)

        mu_tv, std_tv = self.encode_tv(cross_tv)
        mu_ta, std_ta = self.encode_ta(cross_ta)
        ## projection 以后 bottleneck 再对比

        if self.training:
            if self.memory_bank.get_full():
                #得到所有的存储值准备对比
                ta_neg,tv_neg=self.memory_bank.get_all()

                #大量重采样
                reparameter_tv_anchor = self.reparameterize(mu_tv, std_tv)
                reparameter_tv_augment = self.reparameterize(mu_tv, std_tv).detach()

                reparameter_ta_anchor = self.reparameterize(mu_ta, std_ta)
                reparameter_ta_augment = self.reparameterize(mu_ta, std_ta).detach()

                self.memory_bank.update(embeddings_ta_neg=reparameter_ta_augment,embeddings_tv_neg=reparameter_tv_augment)

                infonce_ta,_=self.uncertain_contrastive_learning(query=reparameter_ta_anchor,positive_keys=reparameter_tv_augment,negative_keys=tv_neg)
                infonce_tv,_=self.uncertain_contrastive_learning(query=reparameter_tv_anchor,positive_keys=reparameter_ta_augment,negative_keys=ta_neg)

                ##KL损失计算
                kl_div_ta = self.kl_divergence(reparameter_tv_anchor,reparameter_ta_anchor)
                kl_div_tv = self.kl_divergence(reparameter_ta_anchor,reparameter_tv_anchor)

                infonce_loss = infonce_ta+infonce_tv
                kl_loss = kl_div_ta+kl_div_tv
            else:
                ## 不计算损失，只存入mu和std值
                reparameter_tv_augment = self.reparameterize(mu_tv, std_tv).detach()
                reparameter_ta_augment = self.reparameterize(mu_ta, std_ta).detach()
                infonce_loss = torch.tensor(0.0,dtype=torch.float32,device=torch.device("cuda:0"))
                kl_loss = torch.tensor(0.0,dtype=torch.float32,device=torch.device("cuda:0"))
                self.memory_bank.update(embeddings_ta_neg=reparameter_ta_augment,embeddings_tv_neg=reparameter_tv_augment)

        _,ta_pred = self.ta_head(cross_ta)
        _,tv_pred = self.tv_head(cross_tv)
        fusion, preds = self.fusion_mlp_for_regression(torch.cat([cross_ta, cross_tv], dim=1))  # 32*128,32*1


        if self.training and self.hp.dataset == "ur_funny":
            loss_ta = nn.CrossEntropyLoss(reduction="mean")(ta_pred,y)
            loss_tv = nn.CrossEntropyLoss(reduction="mean")(tv_pred,y)
            head_loss = loss_ta + loss_tv

        if self.training and not self.hp.dataset == "ur_funny":
            loss_ta = nn.L1Loss(reduction="mean")(ta_pred, y)
            loss_tv = nn.L1Loss(reduction="mean")(tv_pred, y)
            head_loss = loss_ta + loss_tv

        if self.training:
            if not self.hp.dataset == "ur_funny":
                ta_ps=self.calculate_prob(ta_pred=ta_pred,tv_pred=tv_pred,all_pred=preds,y=y,threshold=self.hp.threshold,type="ta_ps")
                ta_pn=self.calculate_prob(ta_pred=ta_pred,tv_pred=tv_pred,all_pred=preds,y=y,threshold=self.hp.threshold,type="ta_pn")
                tv_ps=self.calculate_prob(ta_pred=ta_pred,tv_pred=tv_pred,all_pred=preds,y=y,threshold=self.hp.threshold,type="tv_ps")
                tv_pn=self.calculate_prob(ta_pred=ta_pred,tv_pred=tv_pred,all_pred=preds,y=y,threshold=self.hp.threshold,type="tv_pn")
            else:
                ta_ps=self.calculate_prob_acc2(ta_pred=ta_pred,tv_pred=tv_pred,all_pred=preds,y=y,threshold=None,type="ta_ps")
                ta_pn=self.calculate_prob_acc2(ta_pred=ta_pred,tv_pred=tv_pred,all_pred=preds,y=y,threshold=None,type="ta_pn")
                tv_ps=self.calculate_prob_acc2(ta_pred=ta_pred,tv_pred=tv_pred,all_pred=preds,y=y,threshold=None,type="tv_ps")
                tv_pn=self.calculate_prob_acc2(ta_pred=ta_pred,tv_pred=tv_pred,all_pred=preds,y=y,threshold=None,type="tv_pn")
            weight_ta = torch.exp(ta_ps + self.hp.balance_hyper * ta_pn) / (
                    torch.exp(ta_ps + self.hp.balance_hyper * ta_pn) + torch.exp(
                tv_ps + self.hp.balance_hyper * tv_pn))
            weight_tv = torch.exp(tv_ps + self.hp.balance_hyper * tv_pn) / (
                    torch.exp(ta_ps + self.hp.balance_hyper * ta_pn) + torch.exp(
                tv_ps + self.hp.balance_hyper * tv_pn))
            _, ideal_pred = self.ideal_fusion_mlp_for_regression(torch.cat([weight_ta.detach() * cross_ta, weight_tv.detach() * cross_tv], dim=1))
            #_, ideal_pred = self.ideal_fusion_mlp_for_regression(torch.cat([weight_ta * cross_ta, weight_tv * cross_tv], dim=1))
            if self.training and self.hp.dataset == "ur_funny":
                pred_final = nn.CrossEntropyLoss(reduction="mean")(ideal_pred,y)
            if self.training and not self.hp.dataset == "ur_funny":
                pred_final = nn.L1Loss(reduction="mean")(ideal_pred, y)
        '''
        ta_ps_pred = self.ta_ps_mapping(torch.cat([ta_pred,tv_pred,all_pred],dim=-1))
        tv_ps_pred = self.tv_ps_mapping(torch.cat([ta_pred,tv_pred,all_pred],dim=-1))
        ta_pn_pred = self.ta_pn_mapping(torch.cat([ta_pred,tv_pred,all_pred],dim=-1))
        tv_pn_pred = self.tv_pn_mapping(torch.cat([ta_pred,tv_pred,all_pred],dim=-1))

        if self.training:
            loss_ta_ps=nn.L1Loss(reduction="mean")(ta_ps_pred,ta_ps)
            loss_tv_ps=nn.L1Loss(reduction="mean")(tv_ps_pred,tv_ps)
            loss_ta_pn=nn.L1Loss(reduction="mean")(ta_pn_pred,ta_pn)
            loss_tv_pn=nn.L1Loss(reduction="mean")(tv_pn_pred,tv_pn)
            PNS_mapping_loss = loss_ta_ps + loss_tv_ps + loss_ta_pn + loss_tv_pn

        weight_ta = torch.exp(ta_ps_pred + self.hp.balance_hyper*ta_pn_pred)/(torch.exp(ta_ps_pred + self.hp.balance_hyper*ta_pn_pred)+torch.exp(tv_ps_pred + self.hp.balance_hyper*tv_pn_pred))
        weight_tv = torch.exp(tv_ps_pred + self.hp.balance_hyper*tv_pn_pred)/(torch.exp(ta_ps_pred + self.hp.balance_hyper*ta_pn_pred)+torch.exp(tv_ps_pred + self.hp.balance_hyper*tv_pn_pred))
        '''

        if self.training:
            text = enc_word[:,0,:] # 32*768 (batch_size, emb_size)
            acoustic = self.uni_acoustic_enc(acoustic, a_len)  # 32*16
            visual = self.uni_visual_enc(visual, v_len)  # 32*16

            if self.hp.dataset == "sims":
                _, acoustic_pred = self.audio_classifer(acoustic)
                _, visual_pred = self.vision_classifer(visual)
                _, text_pred = self.text_classifer(text)
            else:
                acoustic_pred,visual_pred,text_pred =None,None,None

            if y is not None:
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
                # for ablation use
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
            else:  # 默认进这
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)  # mi_tv 模态互信息
                # lld_tv:-2.1866  tv_pn:{'pos': None, 'neg': None}  H_tv:0.0
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)

            nce_t = self.cpc_zt(text, fusion)  # 3.4660
            nce_v = self.cpc_zv(visual, fusion)  # 3.4625
            nce_a = self.cpc_za(acoustic, fusion)  # 3.4933

            nce = nce_t + nce_v + nce_a  # 10.4218  CPC loss

            pn_dic = {'tv': tv_pn, 'ta': ta_pn, 'va': va_pn if self.add_va else None}
            # {'tv': {'pos': None, 'neg': None}, 'ta': {'pos': None, 'neg': None}, 'va': None}
            lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)  # -5.8927
            H = H_tv + H_ta + (H_va if self.add_va else 0.0)
        if self.training:
            return lld, nce, preds, pn_dic, H,infonce_loss,head_loss,kl_loss,pred_final,text_pred,visual_pred,acoustic_pred
        else:
            return None,None, preds, None, None, None, None, None, None,None,None,None

if __name__=="__main__":
    net=Encoder(4, 8, 2,32,0.1,'relu',2)
    data=torch.randn(30,32,4)
    data_mask=pad_sequence([torch.zeros(torch.FloatTensor(sample).size(0)) for sample in data])
    data_mask[:,4:].fill_(float(1.0))
    output=net(data,data_mask.transpose(1,0))
    print(data_mask,data)