import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding,SpaceEmbedding,DataEmbedding_scale,DataEmbedding_Spacialpatch,DataEmbedding_Spacialpatch_



"""
for scale Downsample 2024.1
"""
class moving_avg(nn.Module):
    """
    Downsample series using an average pooling
    """
    def __init__(self):
        super(moving_avg, self).__init__()

    def forward(self, x, scale=1,if_all=False,data_num=3):
        if x is None:
            return None
        if if_all:
            tmp=[]
            for i in range(data_num):
                y=x[:,:,:,i]
                y = nn.functional.avg_pool1d(y.permute(0, 2, 1), scale, scale)
                tmp.append(y.permute(0, 2, 1))
            if data_num==3:
                return torch.stack((tmp[0],tmp[1],tmp[2]),axis=3)
            else:
                return torch.stack((tmp[0],tmp[1]),axis=3)
        else:
            x=nn.functional.avg_pool1d(x.permute(0, 2, 1), scale, scale)
            x=x.permute(0, 2, 1)
            return x

class InformerUni(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,use_multi_scale=False, patembed=False,
                scales=[32,16,4,1],scale_factor=4, 
                version='Wavelets',mode_select='low',modes=64,L=3,base='legendre',cross_activation='tanh',
                conv_dff=32,
                device=torch.device('cuda:0')):
        super(InformerUni, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.use_multi_scale=use_multi_scale
        self.label_len=label_len

        # space-module
        self.space_embedding=SpaceEmbedding(seq_len,d_model)

        # Encoding
        # scaleformer 2024.1
        if self.use_multi_scale:
            if not patembed:
                self.enc_embedding = DataEmbedding_scale(enc_in, d_model, embed, freq, dropout)
                self.dec_embedding = DataEmbedding_scale(dec_in, d_model, embed, freq, dropout, is_decoder=True)
            else:
                self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
                self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout, is_decoder=True)
        else:
            if not patembed:
                self.enc_embedding = DataEmbedding(enc_in, d_model,seq_len, embed, freq, dropout)
                self.dec_embedding = DataEmbedding(dec_in, d_model,seq_len, embed, freq, dropout)
            else:
                self.enc_embedding = DataEmbedding(enc_in, d_model,seq_len, embed, freq, dropout)
                self.dec_embedding = DataEmbedding(enc_in, d_model,seq_len, embed, freq, dropout)


        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)

        # cyw
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.projection_one = nn.Linear(d_model, c_out, bias=True)
        self.projection_two = nn.Linear(d_model, c_out, bias=True)
        self.projection_three = nn.Linear(d_model, c_out, bias=True)
        self.mlp_one = nn.Linear(d_model*3, c_out, bias=True)
        self.mlp_two = nn.Linear(d_model*3, c_out, bias=True)
        self.mlp_three = nn.Linear(d_model*3, c_out, bias=True)

        """
        following functions will be used to manage scales 
        2024.1
        """
        if self.use_multi_scale:
            self.scale_factor =scale_factor
            self.scales = scales
            self.mv = moving_avg()
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='linear')
            self.use_stdev_norm = False

        """
        modernTCN-ffn2pw cross-task
        2024.3
        """
        self.ffn2pw1 = nn.Conv1d(in_channels=3 * d_model, out_channels=d_model * conv_dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=d_model * conv_dff, out_channels=3 * d_model , kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=d_model)
        self.ffn2drop1 = nn.Dropout(dropout)
        self.ffn2drop2 = nn.Dropout(dropout)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        ## 2023.12
        ### 联合训练
        x_enc_one=self.space_embedding(x_enc[:,:,:,0])
        x_enc_two=self.space_embedding(x_enc[:,:,:,1])
        x_enc_three=self.space_embedding(x_enc[:,:,:,2])

        x_dec_one=self.space_embedding(x_dec[:,:,:,0])
        x_dec_two=self.space_embedding(x_dec[:,:,:,1])
        x_dec_three=self.space_embedding(x_dec[:,:,:,2])

        # scale 2024.1
        if self.use_multi_scale:
            label_len = self.label_len
            scales = self.scales
            inputs_x_enc=torch.stack((x_enc_one,x_enc_two,x_enc_three), axis=3)
            inputs_x_dec=torch.stack((x_dec_one,x_dec_two,x_dec_three), axis=3)
            for scale in scales:
                enc_out = self.mv(inputs_x_enc, scale,True,3)
                if scale == scales[0]: # initialization
                    dec_out = self.mv(inputs_x_dec, scale,True,3)
                else: # upsampling of the output from the previous steps
                    dec_out =torch.stack((self.upsample(dec_out_coarse[:,:,:,0].detach().permute(0,2,1)).permute(0,2,1),
                                            self.upsample(dec_out_coarse[:,:,:,1].detach().permute(0,2,1)).permute(0,2,1),
                                            self.upsample(dec_out_coarse[:,:,:,2].detach().permute(0,2,1)).permute(0,2,1)),axis=3)
                    if dec_out.shape[1] != x_mark_dec.shape[1]//scale:
                        # temp = torch.zeros([dec_out.shape[0],x_mark_dec.shape[1]//scale-dec_out.shape[1],dec_out.shape[2],dec_out.shape[3]],device=dec_out.device)
                        dec_out=torch.cat((dec_out,torch.zeros([dec_out.shape[0],x_mark_dec.shape[1]//scale-dec_out.shape[1],dec_out.shape[2],dec_out.shape[3]],device=dec_out.device)),dim=1)
                    dec_out[:, :label_len//scale, :,:] = self.mv(inputs_x_dec[:, :label_len, :,:], scale,True,3)

                # cross-scale normalization
                mean = torch.cat((enc_out, dec_out[:, label_len//scale:, :,:]), 1).mean(1).unsqueeze(1)
                enc_out = enc_out - mean
                dec_out = dec_out - mean

                if scale>=4:
                    enc_out_one = self.enc_embedding(enc_out[:,:,:,0], x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                    enc_out_one, attns = self.encoder(enc_out_one)
                    dec_out_one = self.dec_embedding(dec_out[:,:,:,0], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                    dec_out_coarse_one = self.decoder(dec_out_one, enc_out_one, x_mask=self.mv(dec_self_mask, scale), cross_mask=self.mv(dec_enc_mask, scale))
                else:
                    dec_out_coarse_one = self.dec_embedding(dec_out[:,:,:,0], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)

                enc_out_two = self.enc_embedding(enc_out[:,:,:,1], x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                enc_out_two, attns = self.encoder(enc_out_two)
                dec_out_two = self.dec_embedding(dec_out[:,:,:,1], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                dec_out_coarse_two = self.decoder(dec_out_two, enc_out_two, x_mask=self.mv(dec_self_mask, scale), cross_mask=self.mv(dec_enc_mask, scale))

                if scale>=4:
                    enc_out_three = self.enc_embedding(enc_out[:,:,:,2], x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                    enc_out_three, attns = self.encoder(enc_out_three)
                    dec_out_three = self.dec_embedding(dec_out[:,:,:,2], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                    dec_out_coarse_three = self.decoder(dec_out_three, enc_out_three, x_mask=self.mv(dec_self_mask, scale), cross_mask=self.mv(dec_enc_mask, scale))
                else:
                    dec_out_coarse_three = self.dec_embedding(dec_out[:,:,:,2], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                
                if scale>=4:
                    all=torch.cat((dec_out_coarse_one,dec_out_coarse_two,dec_out_coarse_three),dim=2)
                    dec_out_one=self.mlp_one(all)
                    dec_out_two=self.mlp_two(all)
                    dec_out_three=self.mlp_three(all)
                else:
                    all=torch.stack((dec_out_coarse_one,dec_out_coarse_two,dec_out_coarse_three),axis=3) #batch,L,d_model,3
                    B, L, D, N = all.shape
                    # D d_model, N=3
                    all = all.permute(0, 2, 3, 1) # B,D,N,L
                    all = all.reshape(B, D * N, L)
                    all = self.ffn2drop1(self.ffn2pw1(all))
                    all = self.ffn2act(all)
                    all = self.ffn2drop2(self.ffn2pw2(all))
                    all = all.permute(0, 2, 1)
                    dec_out_one=self.mlp_one(all)
                    dec_out_two=self.mlp_two(all)
                    dec_out_three=self.mlp_three(all)

                dec_out_coarse=torch.stack((dec_out_one,dec_out_two,dec_out_three),axis=3)

                dec_out_coarse = dec_out_coarse + mean         

            return dec_out_coarse[:, -self.pred_len:, :,:]

        else:
            enc_out_one = self.enc_embedding(x_enc_one, x_mark_enc)
            enc_out_one, attns_one = self.encoder(enc_out_one, attn_mask=enc_self_mask) 
            dec_out_one = self.dec_embedding(x_dec_one, x_mark_dec)
            dec_out_one = self.decoder(dec_out_one, enc_out_one, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

            enc_out_two = self.enc_embedding(x_enc_two, x_mark_enc)
            enc_out_two, attns_two = self.encoder(enc_out_two, attn_mask=enc_self_mask) 
            dec_out_two = self.dec_embedding(x_dec_two, x_mark_dec)
            dec_out_two = self.decoder(dec_out_two, enc_out_two, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

            enc_out_three = self.enc_embedding(x_enc_three, x_mark_enc)
            enc_out_three, attns_three = self.encoder(enc_out_three, attn_mask=enc_self_mask) 
            dec_out_three = self.dec_embedding(x_dec_three, x_mark_dec)
            dec_out_three = self.decoder(dec_out_three, enc_out_three, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

            all=torch.cat((dec_out_one,dec_out_two,dec_out_three),dim=2)

            dec_out_one=self.mlp_one(all)
            dec_out_two=self.mlp_two(all)
            dec_out_three=self.mlp_three(all)

            if self.output_attention:
                return torch.stack((dec_out_one[:,-self.pred_len:,:],dec_out_two[:,-self.pred_len:,:],dec_out_three[:,-self.pred_len:,:]), axis=3), attns_one
            else:
                return torch.stack((dec_out_one[:,-self.pred_len:,:],dec_out_two[:,-self.pred_len:,:],dec_out_three[:,-self.pred_len:,:]), axis=3) # [B, L, D,F]

            
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,use_multi_scale=False, patembed=False,
                scales=[32,16,4,1],scale_factor=4, 
                version='Wavelets',mode_select='low',modes=64,L=3,base='legendre',cross_activation='tanh',
                conv_dff=32,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.use_multi_scale=use_multi_scale
        self.label_len=label_len

        # space-module
        self.space_embedding=SpaceEmbedding(seq_len,d_model)

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model,seq_len, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(enc_in, d_model,seq_len, embed, freq, dropout)

        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)

        self.projection = nn.Linear(d_model, c_out, bias=True)

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)        

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

class Informer_two(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,use_multi_scale=False,
                scales=[32,16,4,1],scale_factor=4,
                version='Wavelets',mode_select='low',modes=64,L=3,base='legendre',cross_activation='tanh',
                conv_dff=32,
                device=torch.device('cuda:0')):
        super(Informer_two, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.use_multi_scale=use_multi_scale
        self.label_len=label_len

        # space-module
        self.space_embedding=SpaceEmbedding(seq_len,d_model)

        # Encoding
        # scaleformer 2024.1
        if self.use_multi_scale:
            self.enc_embedding = DataEmbedding_scale(enc_in, d_model, embed, freq, dropout)
            self.dec_embedding = DataEmbedding_scale(dec_in, d_model, embed, freq, dropout, is_decoder=True)
        else:
            self.enc_embedding = DataEmbedding(enc_in, d_model,seq_len, embed, freq, dropout)
            # TCN
            # self.enc_embedding = DataEmbedding(int(enc_in/4), d_model,seq_len, embed, freq, dropout)
            self.dec_embedding = DataEmbedding(dec_in, d_model,seq_len, embed, freq, dropout)

        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)

        # cyw
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.mlp_one = nn.Linear(d_model*2, c_out, bias=True)
        self.mlp_two = nn.Linear(d_model*2, c_out, bias=True)

        """
        following functions will be used to manage scales 
        2024.1
        """
        if self.use_multi_scale:
            self.scale_factor =scale_factor
            self.scales = scales
            self.mv = moving_avg()
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='linear')
            self.use_stdev_norm = False

        """
        modernTCN-ffn2pw cross-task
        2024.3
        """
        self.ffn2pw1 = nn.Conv1d(in_channels=2 * d_model, out_channels=d_model * conv_dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=d_model * conv_dff, out_channels=2 * d_model , kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=d_model)
        self.ffn2drop1 = nn.Dropout(dropout)
        self.ffn2drop2 = nn.Dropout(dropout)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        ##  2023.12
        ### 联合训练
        x_enc_one=x_enc[:,:,:,0]
        x_dec_one=x_dec[:,:,:,0]
        x_enc_two=x_enc[:,:,:,1]
        x_dec_two=x_dec[:,:,:,1]

        x_enc_one=self.space_embedding(x_enc_one)
        x_enc_two=self.space_embedding(x_enc_two)

        x_dec_one=self.space_embedding(x_dec_one)
        x_dec_two=self.space_embedding(x_dec_two)

        # scaleformer 2024.1
        label_len = self.label_len
        scales = self.scales
        inputs_x_enc=torch.stack((x_enc_one,x_enc_two), axis=3)
        inputs_x_dec=torch.stack((x_dec_one,x_dec_two), axis=3)
        outputs = [] 
        for scale in scales:
            enc_out = self.mv(inputs_x_enc, scale,True,2)
            if scale == scales[0]: # initialization
                dec_out = self.mv(inputs_x_dec, scale,True,2)
            else: # upsampling of the output from the previous steps
                dec_out =torch.stack((self.upsample(dec_out_coarse[:,:,:,0].detach().permute(0,2,1)).permute(0,2,1),
                                        self.upsample(dec_out_coarse[:,:,:,1].detach().permute(0,2,1)).permute(0,2,1)),axis=3)
                dec_out[:, :label_len//scale, :,:] = self.mv(inputs_x_dec[:, :label_len, :,:], scale,True,2)

            # cross-scale normalization
            mean = torch.cat((enc_out, dec_out[:, label_len//scale:, :,:]), 1).mean(1).unsqueeze(1)
            enc_out = enc_out - mean
            dec_out = dec_out - mean

            if scale>=4:
                enc_out_one = self.enc_embedding(enc_out[:,:,:,0], x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                enc_out_one, attns = self.encoder(enc_out_one)
                dec_out_one = self.dec_embedding(dec_out[:,:,:,0], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                dec_out_coarse_one = self.decoder(dec_out_one, enc_out_one, x_mask=self.mv(dec_self_mask, scale), cross_mask=self.mv(dec_enc_mask, scale))
            else:
                dec_out_coarse_one = self.dec_embedding(dec_out[:,:,:,0], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)

            if scale>=4:
                enc_out_two = self.enc_embedding(enc_out[:,:,:,1], x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                enc_out_two, attns = self.encoder(enc_out_two)
                dec_out_two = self.dec_embedding(dec_out[:,:,:,1], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                dec_out_coarse_two = self.decoder(dec_out_two, enc_out_two, x_mask=self.mv(dec_self_mask, scale), cross_mask=self.mv(dec_enc_mask, scale))
            else:
                dec_out_coarse_two = self.dec_embedding(dec_out[:,:,:,1], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)

            if scale>=4:
                all=torch.cat((dec_out_coarse_one,dec_out_coarse_two),dim=2)
                dec_out_one=self.mlp_one(all)
                dec_out_two=self.mlp_two(all)
            else:
                all=torch.stack((dec_out_coarse_one,dec_out_coarse_two),axis=3) #batch,L,d_model,2
                B, L, D, N = all.shape
                # D d_model, N=2
                all = all.permute(0, 2, 3, 1) # B,D,N,L
                all = all.reshape(B, D * N, L)
                all = self.ffn2drop1(self.ffn2pw1(all))
                all = self.ffn2act(all)
                all = self.ffn2drop2(self.ffn2pw2(all))
                all = all.permute(0, 2, 1)
                dec_out_one=self.mlp_one(all)
                dec_out_two=self.mlp_two(all)

            dec_out_coarse=torch.stack((dec_out_one,dec_out_two),axis=3)

            dec_out_coarse = dec_out_coarse + mean         

        return dec_out_coarse[:, -self.pred_len:, :,:]

