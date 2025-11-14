import torch.nn as nn
from models import adv_layer


class ENSTAN_SSTEA(nn.Module):
    def __init__(self, graph_layer, encoder_layer, num_classes):
        super(ENSTAN_SSTEA, self).__init__()
        self.graph_layer = graph_layer
        self.encoder_layer = encoder_layer
        self.fc_emo = nn.Linear(98, num_classes)
        self.domain_classifier = adv_layer.DomainAdversarialLoss(hidden_1=98)
    def forward(self, data_eeg_hbo_s, data_eeg_hbr_s, data_hbo_hbr_s, data_eeg_hbo_t, data_eeg_hbr_t,
                data_hbo_hbr_t,alpha, query_key_padding_mask=None, query_attn_mask=None,src_key_padding_mask=None,
                src_attn_mask=None, hbr_key_padding_mask=None,  hbr_attn_mask=None,mode='train'):
        if mode == 'train':
            eeg_s, hbo_s, hbr_s = self.graph_layer(data_eeg_hbo_s, data_eeg_hbr_s, data_hbo_hbr_s)
            eeg_t, hbo_t, hbr_t = self.graph_layer(data_eeg_hbo_t, data_eeg_hbr_t, data_hbo_hbr_t)
            eeg_hbo_hbr_s = torch.cat((eeg_s,hbo_s,hbr_s),dim=-1)
            eeg_hbo_hbr_t = torch.cat((eeg_t, hbo_t, hbr_t), dim=-1)

            eeg_hbo_hbr_s, eeg_hbo_map_s, hbr_eeg_map_s = self.encoder_layer(eeg_s, hbo_s, hbr_s,query_key_padding_mask,
                                                                                query_attn_mask, src_key_padding_mask,
                                                                                src_attn_mask, hbr_key_padding_mask, hbr_attn_mask)
            eeg_hbo_hbr_t, eeg_hbo_map_t, hbr_eeg_map_t = self.encoder_layer(eeg_t, hbo_t, hbr_t,query_key_padding_mask,
                                                                                query_attn_mask, src_key_padding_mask,
                                                                                src_attn_mask, hbr_key_padding_mask, hbr_attn_mask)
            pooled_eeg_hbo_hbr_features_s = eeg_hbo_hbr_s.mean(dim=1)
            pooled_eeg_hbo_hbr_features_t = eeg_hbo_hbr_t.mean(dim=1)

            output_emo = self.fc_emo(pooled_eeg_hbo_hbr_features_s)
            loss_adv = self.domain_classifier(pooled_eeg_hbo_hbr_features_s,
                                              pooled_eeg_hbo_hbr_features_t,alpha)
            return output_emo, loss_adv
        elif mode == 'test':

            eeg_t, hbo_t, hbr_t = self.graph_layer(data_eeg_hbo_t, data_eeg_hbr_t, data_hbo_hbr_t)

            eeg_hbo_hbr_t, eeg_hbo_map_t, hbr_eeg_map_t = self.encoder_layer(eeg_t, hbo_t, hbr_t,
                                                                             query_key_padding_mask,
                                                                             query_attn_mask, src_key_padding_mask,
                                                                             src_attn_mask, hbr_key_padding_mask,
                                                                             hbr_attn_mask)
            eeg_hbo_hbr_t = torch.cat((eeg_t, hbo_t, hbr_t), dim=-1)
            pooled_eeg_hbo_hbr_features_t = eeg_hbo_hbr_t.mean(dim=1)

            output_emo = self.fc_emo(pooled_eeg_hbo_hbr_features_t)
            return output_emo
        else:
            raise ValueError("Mode should be 'train' or 'test'")