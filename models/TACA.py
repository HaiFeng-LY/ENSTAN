import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
    def forward(self, x):
        seq_len = x.size(1)
        if self.max_len is None or self.max_len >= seq_len:
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
        else:
            print(
                f"Warning: max_len ({self.max_len}) is smaller than the input sequence length ({seq_len}). Truncating the input sequence.")
            pe = torch.zeros(self.max_len, self.d_model, device=x.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe
        return self.dropout(x)


class TACA(nn.Module):
    def __init__(self,d_model_eeg,d_model_hbo,nhead,dim_feedforward=512,dropout=dropouts,activation="relu"):
        super(TACA, self).__init__()
        self.d_model_eeg = d_model_eeg
        self.d_model_hbo = d_model_hbo
        self.nhead = nhead

        self.position_eeg = PositionalEncoding(62, dropout=dropout)
        self.position_hbo = PositionalEncoding(18, dropout=dropout)
        self.position_hbr = PositionalEncoding(18, dropout=dropout)

        self.self_attn_eeg = nn.MultiheadAttention(62, nhead, dropout=dropout)
        self.self_attn_hbo = nn.MultiheadAttention(18, nhead, dropout=dropout)
        self.self_attn_hbr = nn.MultiheadAttention(18, nhead, dropout=dropout)

        self.dropout1_eeg = nn.Dropout(dropout)
        self.norm1_eeg = nn.LayerNorm(62)
        self.linear1_eeg = nn.Linear(62, dim_feedforward)
        self.dropout_eeg = nn.Dropout(dropout)
        self.linear2_eeg = nn.Linear(dim_feedforward, 62)
        self.dropout2_eeg = nn.Dropout(dropout)
        self.norm2_eeg = nn.LayerNorm(62)

        self.dropout1_hbo = nn.Dropout(dropout)
        self.norm1_hbo   = nn.LayerNorm(18)
        self.linear1_hbo = nn.Linear(18, dim_feedforward)
        self.dropout_hbo = nn.Dropout(dropout)
        self.linear2_hbo = nn.Linear(dim_feedforward, 18)
        self.dropout2_hbo = nn.Dropout(dropout)
        self.norm2_hbo = nn.LayerNorm(18)

        self.dropout1_hbr = nn.Dropout(dropout)
        self.norm1_hbr   = nn.LayerNorm(18)
        self.linear1_hbr = nn.Linear(18, dim_feedforward)
        self.dropout_hbr = nn.Dropout(dropout)
        self.linear2_hbr = nn.Linear(dim_feedforward, 18)
        self.dropout2_hbr = nn.Dropout(dropout)
        self.norm2_hbr   = nn.LayerNorm(18)

        self.con1d_eeg_hbo  = nn.Conv1d(in_channels=62, out_channels=18, kernel_size=1, padding=0)
        self.con1d_eeg_hbo2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1, padding=0)

        self.con1d_eeg_hbr  = nn.Conv1d(in_channels=62, out_channels=18, kernel_size=1, padding=0)
        self.con1d_eeg_hbr2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1, padding=0)

        self.con1d_hbo_eeg  = nn.Conv1d(in_channels=18, out_channels=62, kernel_size=1, padding=0)
        self.con1d_hbo_eeg2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1, padding=0)

        self.con1d_hbr_eeg  = nn.Conv1d(in_channels=18, out_channels=62, kernel_size=1, padding=0)
        self.con1d_hbr_eeg2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1, padding=0)

        self.cross_attn_eeg_hbo = nn.MultiheadAttention(18, nhead, dropout=dropout)
        self.cross_attn_eeg_hbr = nn.MultiheadAttention(18, nhead, dropout=dropout)
        self.cross_attn_hbo_eeg = nn.MultiheadAttention(62, nhead, dropout=dropout)
        self.cross_attn_hbr_eeg = nn.MultiheadAttention(62, nhead, dropout=dropout)

        self.dropout3_eeg = nn.Dropout(dropout)
        self.norm3_eeg_hbo = nn.LayerNorm(18)

        self.dropout4_eeg = nn.Dropout(dropout)
        self.norm4_eeg_hbr = nn.LayerNorm(18)

        self.dropout3_hbo = nn.Dropout(0.5)
        self.norm3_hbo = nn.LayerNorm(62)
        self.dropout4_hbo = nn.Dropout(dropout)

        self.dropout3_hbr = nn.Dropout(0.5)
        self.norm3_hbr = nn.LayerNorm(62)
        self.dropout4_hbr = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

        self.con1d_eeg_hbohbr1 = nn.Conv1d(in_channels=62, out_channels=18, kernel_size=1, padding=0)
        self.con1d_eeg_hbohbr2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1, padding=0)

        self.fusion1 = nn.Conv1d(in_channels=62, out_channels=18, kernel_size=1, padding=0)
        self.fusion2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=1, padding=0)

    def forward(self,query,src,hbr,query_key_padding_mask=None, query_attn_mask=None,
                src_key_padding_mask=None, src_attn_mask=None,hbr_key_padding_mask=None,
                hbr_attn_mask=None):
        query = query.to(torch.float32)
        src   = src.to(torch.float32)
        hbr   = hbr.to(torch.float32)

        eeg   = self.position_eeg(query)
        src   = self.position_hbo(src)
        hbr   = self.position_hbr(hbr)


        eeg_self = self.self_attn_eeg(eeg,eeg,eeg,attn_mask=src_attn_mask,
                                      key_padding_mask=src_key_padding_mask)[0]
        eeg  = eeg + self.dropout1_eeg(eeg_self)
        eeg  = self.norm1_eeg(eeg)
        eeg2 = self.activation(self.linear1_eeg(eeg))
        eeg2 = self.dropout_eeg(eeg2)
        eeg2 = self.linear2_eeg(eeg2)
        eeg  = eeg + self.dropout2_eeg(eeg2)
        eeg  = self.norm2_eeg(eeg)


        src_self = self.self_attn_hbo(src, src, src, attn_mask=src_attn_mask,  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1_hbo(src_self)
        src = self.norm1_hbo(src)
        src2 = self.activation(self.linear1_hbo(src))
        src2 = self.dropout_hbo(src2)
        src2 = self.linear2_hbo(src2)
        src = src + self.dropout2_hbo(src2)
        src = self.norm2_hbo(src)

        hbr_self = self.self_attn_hbr(hbr, hbr, hbr, attn_mask=hbr_attn_mask, key_padding_mask=hbr_key_padding_mask)[0]
        hbr = hbr + self.dropout1_hbr(hbr_self)
        hbr = self.norm1_hbr(hbr)
        hbr2 = self.activation(self.linear1_hbr(hbr))
        hbr2 = self.dropout_hbr(hbr2)
        hbr2 = self.linear2_hbr(hbr2)
        hbr = hbr + self.dropout2_hbr(hbr2)
        hbr = self.norm2_hbr(hbr)
        eeg_con1 = self.con1d_eeg_hbohbr1(eeg.permute(0,2,1)).permute(0,2,1)
        eeg_con2 = self.con1d_eeg_hbohbr2(eeg_con1)


        eeg_hbo_cnn = self.con1d_eeg_hbo(eeg.permute(0,2,1)).permute(0,2,1)
        bsz, seq_len_eeg, _ = eeg_hbo_cnn.size()
        seq_len_src = src.size(1)
        eeg_hbo_attn_mask = torch.zeros(bsz,seq_len_eeg,seq_len_src,device=eeg_hbo_cnn.device)
        for qq in range(seq_len_eeg):
            for kk in range(seq_len_src):
                if qq / 200.0 > kk / 11.0:
                    eeg_hbo_attn_mask[:, qq, kk] = 1.0
        eeg_hbo,eeg_hbo_map = self.cross_attn_eeg_hbo(eeg_hbo_cnn.permute(1,0,2), src.permute(1,0,2),
                                                      src.permute(1,0,2),attn_mask=eeg_hbo_attn_mask,
                                                      key_padding_mask=query_key_padding_mask)
        eeg_hbo_cnn2 = self.con1d_eeg_hbo2(src)
        eeg_hbo = eeg_hbo_cnn2 + self.dropout3_eeg(eeg_hbo.permute(1,0,2))
        eeg_hbo = self.norm3_eeg_hbo(eeg_hbo)

        eeg_hbr_cnn = self.con1d_eeg_hbr(eeg.permute(0, 2, 1)).permute(0, 2, 1)

        bsz, seq_len_eeg2, _ = eeg_hbr_cnn.size()
        seq_len_src2 = hbr.size(1)
        eeg_hbr_attn_mask = torch.zeros(bsz, seq_len_eeg2, seq_len_src2, device=eeg_hbr_cnn.device)
        for qq2 in range(seq_len_eeg2):
            for kk2 in range(seq_len_src2):
                if qq2 / 200.0 > kk2 / 11.0:
                    eeg_hbr_attn_mask[:, qq2, kk2] = 1.0
        eeg_hbr,eeg_hbr_map = self.cross_attn_eeg_hbr(eeg_hbr_cnn.permute(1,0,2), hbr.permute(1,0,2),
                                                      hbr.permute(1,0,2), attn_mask=eeg_hbr_attn_mask,
                                                       key_padding_mask=query_key_padding_mask)
        eeg_hbr_cnn2 = self.con1d_eeg_hbr2(hbr)
        eeg_hbr = eeg_hbr_cnn2 + self.dropout4_eeg(eeg_hbr.permute(1,0,2))
        eeg_hbr = self.norm4_eeg_hbr(eeg_hbr)

        hbo_eeg_cnn = self.con1d_hbo_eeg(src.permute(0, 2, 1)).permute(0, 2, 1)
        bsz, seq_len_hbo, _ = hbo_eeg_cnn.size()
        seq_len_eeg = eeg.size(1)
        hbo_eeg_attn_mask = torch.zeros(bsz, seq_len_hbo, seq_len_eeg, device=hbo_eeg_cnn.device)
        for qq3 in range(seq_len_hbo):
            for kk3 in range(seq_len_eeg):
                if qq3 / 11.0 <= kk3 / 200.0:
                    hbo_eeg_attn_mask[:, qq3, kk3] = 1.0
        hbo_eeg,hbo_eeg_map = self.cross_attn_hbo_eeg(hbo_eeg_cnn.permute(1,0,2),eeg.permute(1,0,2),
                                            eeg.permute(1,0,2),
                                                      attn_mask=hbo_eeg_attn_mask,
                                            key_padding_mask=query_key_padding_mask)
        hbo_eeg_cnn2 = self.con1d_hbo_eeg2(hbo_eeg.permute(1,0,2))
        hbo_eeg = eeg + self.dropout3_hbo(hbo_eeg_cnn2)
        hbo_eeg = self.norm3_hbo(hbo_eeg)
        hbo_eeg = self.dropout4_hbo(hbo_eeg)

        hbr_eeg_cnn = self.con1d_hbr_eeg(hbr.permute(0, 2, 1)).permute(0, 2, 1)
        bsz, seq_len_eeg4, _ = hbr_eeg_cnn.size()
        seq_len_src4= eeg.size(1)
        hbr_eeg_attn_mask = torch.zeros(bsz, seq_len_eeg4, seq_len_src4, device=hbr_eeg_cnn.device)
        for qq4 in range(seq_len_eeg4):
            for kk4 in range(seq_len_src4):
                if qq4 / 11.0 <= kk4 / 200.0:
                    hbr_eeg_attn_mask[:, qq4, kk4] = 1.0
        hbr_eeg,hbr_eeg_map = self.cross_attn_hbr_eeg(hbr_eeg_cnn.permute(1,0,2), eeg.permute(1,0,2),
                                            eeg.permute(1,0,2),
                                            attn_mask=hbr_eeg_attn_mask,
                                            key_padding_mask=query_key_padding_mask)
        hbr_eeg_cnn2 = self.con1d_hbr_eeg2(hbr_eeg.permute(1,0,2))
        hbr_eeg = eeg + self.dropout3_hbr(hbr_eeg_cnn2)
        hbr_eeg = self.norm3_hbr(hbr_eeg)
        hbr_eeg = self.dropout4_hbr(hbr_eeg)

        eeg_hbo_hbr = eeg_hbo + eeg_hbr
        hbo_hbr_eeg = hbo_eeg + hbr_eeg
        return hbo_hbr_eeg,eeg_hbo_hbr, eeg_hbo_map,hbr_eeg_map
