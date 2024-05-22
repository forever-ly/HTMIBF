__all__ = ['PatchTSM_backbone']

# Cell
from typing import Callable, Optional
import torch
from mamba_ssm import Mamba
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from layers.PatchTST_backbone import TSTEncoderLayer
import torch.nn.functional as F

# Cell
class PatchTSM_backbone2(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False,share=True,type="split",t_layers=2,tran_first=True,K = 2,IB=True,seperate_mlp=False,mlp_dropout=0.3,**kwargs):

        super().__init__()

        # RevIn

        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSMiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose,share=share,type=type,t_layers=t_layers,tran_first=tran_first,K = K,IB=IB,seperate_mlp=seperate_mlp,mlp_dropout=mlp_dropout,**kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        # head_type =  "conv"
        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in,
                                                  fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.head_nf, target_window,
                                     head_dropout=head_dropout)
        elif head_type =="conv":
            K=4
            self.head = Conv_Head(d_model,patch_num,int(self.head_nf/4), target_window,
                                     head_dropout=head_dropout,K=K)

            # self.head = Flatten_Head(self.individual, self.n_vars, d_model*64, target_window,
            #                          head_dropout=head_dropout)
            # self.head = nn.Sequential(
            #     nn.Linear(d_model,target_window)
            # )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z,KL_loss = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]

        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z,KL_loss

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, vars, 1)
                             )
class Conv_Head(nn.Module):
    def __init__(self, d_model,patch_num,nf, target_window, head_dropout=0,K=4):
        super().__init__()

        self.K = K
        self.d_model = d_model
        self.patch_num = patch_num
        self.flatten = nn.Flatten(start_dim=3)


        self.conv = nn.Conv2d(1, d_model, kernel_size=(K, d_model), stride=(K, d_model))
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]]
        bs, n_vars, d_model, patch_num = x.shape
        x = x.permute(0, 1, 3, 2)

        x = torch.stack(x.chunk(self.K, dim=-2), dim=2)
        x = self.flatten(x)
        x = self.conv(x.reshape(bs * n_vars, 1, self.K, -1)).reshape(bs, n_vars, -1)


        x = self.linear(x)  # [bs,nvars,target_window]
        x = self.dropout(x)
        return x


class Flatten_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()


        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.flatten(x)  # [bs,nvars,x d_model * patch_num]

        x = self.linear(x)  # [bs,nvars,target_window]
        x = self.dropout(x)
        return x






class Last_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(nf, target_window)
        )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.linear(x)  # [bs,nvars,target_window]
        x = self.dropout(x)
        return x


class TSMiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False,share=True,type="split",t_layers=2,tran_first=True,K = 2,temp=1.0 ,IB=True,seperate_mlp=False,mlp_dropout=0.2,**kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSMEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn,share=share,type=type,t_layers=t_layers,tran_first=tran_first,K = K,temp=temp,n_vars=c_in,IB=IB,seperate_mlp=seperate_mlp,mlp_dropout=mlp_dropout)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z,KL_loss = self.encoder(u)

        patch_num, d_model = z.shape[1], z.shape[2]


        z = torch.reshape(z, (-1, n_vars, patch_num, d_model))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z,KL_loss

    # Cell


class TSMEncoder(nn.Module):

    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False,share=True,type="split",t_layers=2,tran_first=True,K = 2,temp=1.0,n_vars=21,IB=True,seperate_mlp=False,mlp_dropout=0.3):
        super().__init__()
        self.K = K
        self.tran_first = tran_first
        self.share = share
        self.type = type
        self.temp = temp
        print("sss",K,share,type,temp,n_vars,IB)
        self.t_layers = t_layers
        self.n_vars = n_vars
        self.IB = IB
        self.seperate_mlp = seperate_mlp
        if not self.seperate_mlp:
            self.compressor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Dropout(mlp_dropout),
                nn.BatchNorm2d(self.n_vars),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.compressor = nn.ModuleList(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(q_len),
                nn.Dropout(mlp_dropout),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            ) for i in range(self.n_vars)
        )

        if self.share:
            self.trans = nn.ModuleList([TSTEncoderLayer(q_len / self.K, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                          attn_dropout=attn_dropout, dropout=dropout,
                                                          activation=activation, res_attention=res_attention,
                                                          pre_norm=pre_norm, store_attn=store_attn) for i in range(t_layers)])
        else:
            self.trans = nn.ModuleList(nn.ModuleList([TSTEncoderLayer(q_len / self.K, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                          attn_dropout=attn_dropout, dropout=dropout,
                                                          activation=activation, res_attention=res_attention,
                                                          pre_norm=pre_norm, store_attn=store_attn) for i in range(t_layers)]) for j in range(self.K))

        self.mamba_layer = nn.ModuleList([TSMEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention



    def compress(self, z,temperature=1.0):
        if not self.seperate_mlp:
            p = self.compressor(z)
        else:
            bs, nvars, L, d_model = z.shape
            p_list = []
            for i in range(nvars):
                p_mid = self.compressor[i](z[:, i, :, :])
                p_list.append(p_mid)
            p = torch.stack(p_list, dim=1)
        bias = 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(z.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs.squeeze()).unsqueeze(-1)
        p = torch.sigmoid(p.squeeze()).unsqueeze(-1)
        return gate_inputs, p


    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        output = src

        scores = None

        for mod in self.mamba_layer:
            output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        bs_nvar, L, d_model = output.shape
        KL_loss = torch.tensor(0).to(output.device)
        if self.IB:

            # output = output * self.mlp(output)
            o = output.reshape(-1, self.n_vars, L, d_model)  # # bs,vars, patch_num,d_model

            lambda_pos, p = self.compress(o,temperature=self.temp)  # bs,vars, logit

            if not torch.is_grad_enabled():
                lambda_pos = p

            lambda_neg = 1 - lambda_pos

            feature_mean = o.mean(-2).detach().unsqueeze(-2)
            feature_std = o.std(-2).detach().unsqueeze(-2)
            noisy_patch_feature_mean = lambda_pos * o + lambda_neg * feature_mean
            noisy_patch_feature_std = lambda_neg * feature_std
            output = noisy_patch_feature_mean + torch.rand_like(noisy_patch_feature_mean) * noisy_patch_feature_std
            output = output.reshape(-1, L, d_model)
            epsilon = 1e-7
            # KL_tensor = 0.5 * ((noisy_patch_feature_std ** 2) / (feature_std + epsilon) ** 2) + \
            #             torch.sum(((noisy_patch_feature_mean - feature_mean) / (feature_std + epsilon)) ** 2,dim=-2).unsqueeze(-2)
            # KL_loss = torch.mean(KL_tensor)
            # KL_loss = (lambda_neg ** 2).mean() + \
            #             torch.mean(((noisy_patch_feature_mean - feature_mean) / (feature_std + epsilon)) ** 2) - torch.log((lambda_neg ** 2).sum())

            # KL_tensor = ((lambda_neg ** 2).mean(-1).unsqueeze(dim=-1) + \
            #             torch.mean((torch.sum(noisy_patch_feature_mean - feature_mean,dim=2) / (feature_std + epsilon).squeeze())**2 ) / L ) - torch.log((lambda_neg ** 2).sum(-1).unsqueeze(dim=-1)+0.00001)

            KL_tensor = ((lambda_neg ** 2).mean(-1).unsqueeze(dim=-1) + \
                        torch.mean((torch.sum(noisy_patch_feature_mean - feature_mean,dim=2) / (feature_std + epsilon).squeeze())**2 ) / L )
            KL_loss = torch.mean(KL_tensor)
        # output += res
        if self.share:
            if self.type == "interval":
                l = []
                for i in range(self.K):
                    l.append(output[:,i::self.K, :])
                output_sub = torch.concat(l, dim=1)
                for mod in self.trans:
                    output_sub, scores = mod(output_sub, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                output_sub = output_sub.chunk(self.K, dim=1)
                merge_output = output.new_zeros(output.size())
                for i in range(self.K):
                    merge_output[:, i::self.K, :] = output_sub[i]
                output = merge_output
            elif self.type == "split":


                # output_chuncks = torch.concat(torch.chunk(output, self.K, dim=1),dim=0)
                #
                # for mod in self.trans:
                #     output_chuncks, scores = mod(output_chuncks, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                #
                #
                # output_sub = output_chuncks.chunk(self.K, dim=0)
                # output = torch.cat(output_sub, dim=1)


                output_sub = []
                output_chuncks = torch.chunk(output, self.K, dim=1)
                for o in output_chuncks:
                    for mod in self.trans:

                        o, scores = mod(o, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                    output_sub.append(o)

                output = torch.cat(output_sub,dim=1)

            elif self.type == "random":
                perm = torch.randperm(L)
                output_sub = []
                choose_list =[]
                for i in range(self.K):
                    choose = perm[int(L / self.K) * i:int(L / self.K) * (i + 1)]
                    choose_list.append(choose)
                    output_sub.append(output[:, choose, :])
                output_sub = torch.concat(output_sub, dim=1)
                for mod in self.trans:
                    output_sub, scores = mod(output_sub, prev=scores, key_padding_mask=key_padding_mask,
                                             attn_mask=attn_mask)
                output_sub = output_sub.chunk(self.K, dim=1)
                merge_output = output.new_zeros(output.size())
                for i in range(self.K):
                    merge_output[:, choose_list[i], :] = output_sub[i]
                output = merge_output

            else:
                raise Exception("dd")
        else:
            if self.type == "interval":
                l = []
                for i in range(self.K):
                    l.append(output[:,i::self.K, :])
                # output_sub = torch.concat(l, dim=1)
                output_sub = []
                for mods,o in zip(self.trans,l):
                    for mod in mods:
                        o, scores = mod(o, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                    output_sub.append(o)

                merge_output = output.new_zeros(output.size())
                for i in range(self.K):
                    merge_output[:, i::self.K, :] = output_sub[i]
                output = merge_output
            elif self.type == "split":
                # output_chuncks = torch.concat(torch.chunk(output, self.K, dim=1),dim=0)
                output_chuncks = torch.chunk(output, self.K, dim=1)
                output_sub = []
                for mods,o in zip(self.trans,output_chuncks):
                    for mod in mods:
                        o, scores = mod(o, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                    output_sub.append(o)
                output = torch.concat(output_sub,dim=1)
                # output = output.reshape(output.size())
        return output,KL_loss



class TSMEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()

        # Multi-Head attention
        self.res_attention = res_attention
        # self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        self.mamba = Mamba(

            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

        print(sum(p.numel() for p in self.mamba.parameters()))
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        if self.pre_norm:
            src = self.norm_attn(src)

        src2 = self.mamba(src)

        # ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout

        if not self.pre_norm:
            src = self.norm_ffn(src)
        scores = None
        if self.res_attention:
            return src, scores
        else:
            return src








