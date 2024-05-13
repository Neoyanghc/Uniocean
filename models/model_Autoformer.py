import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attn import AutoCorrelation, AutoCorrelationLayer
from models.autoformer_encdec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, Decoder_base, DecoderLayer_base
from models.embed import DataEmbedding,SpaceEmbedding,DataEmbedding_scale

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

class AutoformerUni(nn.Module):
    """
    Multi-scale version of Autoformer
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,use_multi_scale=False, Patembed=False,
                scales=[32,16,4,1],scale_factor=4,
                version='Wavelets',mode_select='low',modes=64,L=3,base='legendre',cross_activation='tanh',
                conv_dff=32,
                device=torch.device('cuda:0')):
        super(AutoformerUni, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = out_len
        self.output_attention = output_attention
        self.d_model=d_model

        # Decomp
        self.decomp = series_decomp(move_avg)

        # space-module
        self.space_embedding=SpaceEmbedding(seq_len,d_model)

        # Embedding
        # We use our new DataEmbedding which incldues the scale information
        self.enc_embedding = DataEmbedding_scale(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_scale(dec_in, d_model, embed, freq, dropout, is_decoder=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg = move_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False), d_model,n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg = move_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            # projection=nn.Linear(d_model, c_out, bias=True)
        )

        self.projection1 = nn.Linear(d_model, c_out, bias=True)
        self.projection2 = nn.Linear(d_model, c_out, bias=True)
        self.deprojection = nn.Linear(c_out, d_model, bias=True)
        self.seasonalmlp_one = nn.Linear(d_model*3, c_out, bias=True)
        self.seasonalmlp_two = nn.Linear(d_model*3, c_out, bias=True)
        self.seasonalmlp_three = nn.Linear(d_model*3, c_out, bias=True)
        self.trendmlp_one = nn.Linear(d_model*3, c_out, bias=True)
        self.trendmlp_two = nn.Linear(d_model*3, c_out, bias=True)
        self.trendmlp_three = nn.Linear(d_model*3, c_out, bias=True)


        """
        following functions will be used to manage scales
        """
        self.scale_factor = scale_factor
        self.scales = scales
        self.mv = moving_avg()
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='linear')
        self.input_decomposition_type = 1

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

        x_enc_one=self.space_embedding(x_enc[:,:,:,0])
        x_enc_two=self.space_embedding(x_enc[:,:,:,1])
        x_enc_three=self.space_embedding(x_enc[:,:,:,2])

        x_dec_one=self.space_embedding(x_dec[:,:,:,0])
        x_dec_two=self.space_embedding(x_dec[:,:,:,1])
        x_dec_three=self.space_embedding(x_dec[:,:,:,2])

        scales = self.scales
        label_len = self.label_len
        inputs_x_enc=torch.stack((x_enc_one,x_enc_two,x_enc_three), axis=3)
        inputs_x_dec=torch.stack((x_dec_one,x_dec_two,x_dec_three), axis=3)

        for scale in scales:
            enc_out = self.mv(inputs_x_enc, scale,True,3)
            if scale == scales[0]: # initialize the input of decoder at first step
                if self.input_decomposition_type == 1:
                    mean = enc_out.mean(1).unsqueeze(1)
                    enc_out = enc_out - mean
                    tmp_mean = torch.mean(enc_out, dim=1).unsqueeze(1).repeat(1, self.pred_len//scale, 1,1)
                    zeros = torch.zeros([x_dec.shape[0], self.pred_len//scale, x_dec.shape[2],3], device=x_enc.device)
                    seasonal_init = torch.stack((self.decomp(enc_out[:,:,:,0])[0],self.decomp(enc_out[:,:,:,1])[0],self.decomp(enc_out[:,:,:,2])[0]), axis=3)
                    trend_init = torch.stack((self.decomp(enc_out[:,:,:,0])[1],self.decomp(enc_out[:,:,:,1])[1],self.decomp(enc_out[:,:,:,2])[1]), axis=3)
                    trend_init = torch.cat([trend_init[:, -self.label_len//scale:, :, :], tmp_mean], dim=1)
                    seasonal_init = torch.cat([seasonal_init[:, -self.label_len//scale:, :, :], zeros], dim=1)
                    dec_out = self.mv(inputs_x_dec, scale,True,3) - mean
                else:
                    dec_out = self.mv(inputs_x_dec, scale,True,3)
                    mean = enc_out.mean(1).unsqueeze(1)
                    enc_out = enc_out - mean
                    dec_out[:, :label_len//scale, :,:] = dec_out[:, :label_len//scale, :,:] - mean
            else: # generation the input at each scale and cross normalization
                # dec_out = self.upsample(dec_out_coarse.detach().permute(0,2,1)).permute(0,2,1)
                dec_out =torch.stack((self.upsample(dec_out_coarse[:,:,:,0].detach().permute(0,2,1)).permute(0,2,1),
                                        self.upsample(dec_out_coarse[:,:,:,1].detach().permute(0,2,1)).permute(0,2,1),
                                        self.upsample(dec_out_coarse[:,:,:,2].detach().permute(0,2,1)).permute(0,2,1)),axis=3)
                dec_out[:, :label_len//scale, :,:] = self.mv(x_dec[:, :label_len, :,:], scale,True,3)
                mean = torch.cat((enc_out, dec_out[:, label_len//scale:, :, :]), 1).mean(1).unsqueeze(1)
                enc_out = enc_out - mean
                dec_out = dec_out - mean
            
            # redefining the inputs to the decoder to be scale aware
            trend_init = torch.zeros([dec_out.shape[0],dec_out.shape[1],self.d_model,dec_out.shape[3]],device=enc_out.device)
            # trend_init = torch.zeros([dec_out.shape[0],dec_out.shape[1],dec_out.shape[2],dec_out.shape[3]],device=enc_out.device)

            if scale>=4:
                enc_out_one = self.enc_embedding(enc_out[:,:,:,0], x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                enc_out_one, attns = self.encoder(enc_out_one)                
                dec_out_one = self.dec_embedding(dec_out[:,:,:,0], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                seasonal_part_one, trend_part_one = self.decoder(dec_out_one, enc_out_one, trend=trend_init[:,:,:,0])
            else:
                dec_out_coarse_one = self.dec_embedding(dec_out[:,:,:,0], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                seasonal_part_one=dec_out_coarse_one
                trend_part_one=dec_out_coarse_one

            enc_out_two = self.enc_embedding(enc_out[:,:,:,1], x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
            enc_out_two, attns = self.encoder(enc_out_two)
            dec_out_two = self.dec_embedding(dec_out[:,:,:,1], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
            seasonal_part_two, trend_part_two = self.decoder(dec_out_two, enc_out_two, trend=trend_init[:,:,:,1])

            if scale>=4:
                enc_out_three = self.enc_embedding(enc_out[:,:,:,2], x_mark_enc[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                enc_out_three, attns = self.encoder(enc_out_three)
                dec_out_three = self.dec_embedding(dec_out[:,:,:,2], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                seasonal_part_three, trend_part_three = self.decoder(dec_out_three, enc_out_three, trend=trend_init[:,:,:,2])
            else:
                dec_out_coarse_three = self.dec_embedding(dec_out[:,:,:,2], x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
                seasonal_part_three=dec_out_coarse_three
                trend_part_three=dec_out_coarse_three
                
            if scale>=4:
                all_seasonal=torch.cat((seasonal_part_one,seasonal_part_two,seasonal_part_three),dim=2)
                all_trend=torch.cat((trend_part_one,trend_part_two,trend_part_three),dim=2)

                dec_out_one=self.seasonalmlp_one(all_seasonal)+self.trendmlp_one(all_trend)
                dec_out_two=self.seasonalmlp_two(all_seasonal)+self.trendmlp_two(all_trend)
                dec_out_three=self.seasonalmlp_three(all_seasonal)+self.trendmlp_three(all_trend)
            else:
                seasonal_part_two = self.projection1(seasonal_part_two)
                trend_part_two = self.projection2(trend_part_two)
                dec_out_coarse_two = seasonal_part_two + trend_part_two
                dec_out_coarse_two = self.deprojection(dec_out_coarse_two)
                all=torch.stack((dec_out_coarse_one,dec_out_coarse_two,dec_out_coarse_three),axis=3) #batch,L,d_model,3
                # all_trend=torch.stack((trend_part_one,trend_part_two,trend_part_three),axis=3) 
                B, L, D, N = all.shape
                # D d_model, N=3
                all = all.permute(0, 2, 3, 1) # B,D,N,L
                all = all.reshape(B, D * N, L)
                all = self.ffn2drop1(self.ffn2pw1(all))
                all = self.ffn2act(all)
                all = self.ffn2drop2(self.ffn2pw2(all))
                all = all.permute(0, 2, 1)

                dec_out_one=self.seasonalmlp_one(all)
                dec_out_two=self.seasonalmlp_two(all)
                dec_out_three=self.seasonalmlp_three(all)

            dec_out_coarse=torch.stack((dec_out_one,dec_out_two,dec_out_three),axis=3)

            dec_out_coarse = dec_out_coarse + mean  
                    
        return dec_out_coarse[:, -self.pred_len:, :,:]

class Autoformer(nn.Module):
    """
    Multi-scale version of Autoformer
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,use_multi_scale=False,Patembed=False,
                scales=[32,16,4,1],scale_factor=4,
                version='Wavelets',mode_select='low',modes=64,L=3,base='legendre',cross_activation='tanh',
                conv_dff=32,
                device=torch.device('cuda:0')):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = out_len
        self.output_attention = output_attention
        self.d_model=d_model

        # Decomp
        self.decomp = series_decomp(move_avg)

        # space-module
        self.space_embedding=SpaceEmbedding(seq_len,d_model)

        # Embedding
        # We use our new DataEmbedding which incldues the scale information
        self.enc_embedding = DataEmbedding(enc_in, d_model,seq_len, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model,seq_len, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg = move_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder_base(
            [
                DecoderLayer_base(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False), d_model,n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg = move_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc=self.space_embedding(x_enc)
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]





    