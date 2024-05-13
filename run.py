import argparse
import os
import torch

from exp.exp_uniocean import Exp_UniOcean

parser = argparse.ArgumentParser(description='[UniOcean]')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD),informer_two]')

parser.add_argument('--data', type=str, required=True, default='ALL2', help='data')
parser.add_argument('--root_path', type=str, default='./data/ALL/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ALL2.pkl', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=64800, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=64800, help='decoder input size')
parser.add_argument('--c_out', type=int, default=64800, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--move_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
#############
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type3',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--get_prediction', action='store_true', help='get real prediction', default=False)

##Scale
parser.add_argument('--multi_task', type=bool, default=False,help='multi-task training')
parser.add_argument('--use_multi_scale', action='store_true', help='using mult-scale')
parser.add_argument('--scales', default=[8,4,2,1], nargs='+', type=int , help='scales in mult-scale')
parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')

##ConvFFN
parser.add_argument('--conv_dff', type=int, default=32, help='the hidden dimension of correlation ConvFFN')

##Space_patchembed
parser.add_argument('--patembed', action='store_true', help='use patch embed spacially', default=False)

## supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Wavelets',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='low',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

## ConvLSTM
parser.add_argument('--input_dim', type=int, default=1, help='the number of channels')
parser.add_argument('--hidden_dim', default=[16,8,1], help='the number of channels')
parser.add_argument('--kernel_size', type=tuple, default=(3,3), help='the kernel size of each layer')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_first', type=bool, default=True)
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--return_all_layers', type=bool, default=False)

args = parser.parse_args()

###
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
        ##海洋表面温度数据集
    'SST5':{'data':'SST5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'SST6':{'data':'SST6.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'SST7':{'data':'SST7.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
        ##盐度数据集
    'SAL5':{'data':'SAL5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'SAL6':{'data':'SAL6.pkl','T':'target','M':[10000,10000,10000],'S':[1,1,1],'MS':[10000,10000,1]},
    'SAL8':{'data':'SAL8.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
        ##热通量数据集
    'OHC1':{'data':'OHC1.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OHC5':{'data':'OHC5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
        ##海冰密度数据集
    'ICEC1':{'data':'ICEC1.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
        ##洋流数据集 实验采用OC1
    'OC1':{'data':'OC1.pkl','T':'target','M':[64440,64440,64440],'S':[1,1,1],'MS':[64440,64440,1]},
        ##周平均海盐度数据集OISSS1 2011-2022年13年的数据
    'OISSS1':{'data':'OISSS1.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OISSS3':{'data':'OISSS3.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]}, ##*4
        ##日海洋温度数据集OISST1 分辨率为0.25*0.25 2017-2022六年的日温度数据，输入维度较大，代码memory out
    'OISST1':{'data':'OISST1.pkl','T':'target','M':[1036800,1036800,1036800],'S':[1,1,1],'MS':[1036800,1036800,1]},
        ##日海洋温度数据集OISST2 分辨率为1*1 2017-2022六年的数据
    'OISST2':{'data':'OISST2.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
        ##日海洋温度数据集OISST3 分辨率为1*1 1993-2022年30年的数据
    'OISST3':{'data':'OISST3.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OISST4':{'data':'OISST4.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]}, ## 1993-2022 weekly
    'OISST5':{'data':'OISST5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]}, ## 2011.08.28~2022.12.15 daily
        ##数据集集合
    'ALL2':{'data':'ALL2.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]}, #SSS SST OHC  freq: m w m
    'ALL3':{'data':'ALL3.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]}, #SSS SST
    'ALL4':{'data':'ALL4.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]}, #SSS OHC
    'ALL5':{'data':'ALL5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]}, #SST OHC
    'ALL6':{'data':'ALL6.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]}, #SST OISSS && freq:d 4d && time:2011.08.28~2022.12.15

}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_UniOcean

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}_cdf{}_sf{}_ms{}_pe{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des,args.multi_task ,args.conv_dff, args.scale_factor, args.scales, args.patembed, ii)
    # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
    #             args.seq_len, args.label_len, args.pred_len,
    #             args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
    #             args.embed, args.distil, args.mix, args.des,args.multi_task , ii)
    

    exp = Exp(args) # set experiments
    
    if args.get_prediction:
        exp.get_prediction(setting)
    else:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

    torch.cuda.empty_cache()
