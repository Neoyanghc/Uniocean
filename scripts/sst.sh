python -u run.py --model informer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --train_epoch 50

python -u run.py --model informer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --train_epoch 50

python -u run.py --model informer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --train_epoch 50

python -u run.py --model autoformer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  ---seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 24 --train_epoch 50

python -u run.py --model autoformer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --train_epoch 50

python -u run.py --model autoformer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --train_epoch 50

python -u run.py --model fedformer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --train_epoch 50

python -u run.py --model fedformer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --train_epoch 50

python -u run.py --model fedformer --data OISST4 --attn prob --freq w --root_path ./data/OISST/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --train_epoch 50

python -u run.py --model convlstm --data OISST4  --freq w --root_path ./data/OISST/  --seq_len 32 --label_len 0 --pred_len 32 --batch_size 16 --train_epoch 50 --lradj type4

python -u run.py --model convlstm --data OISST4  --freq w --root_path ./data/OISST/  --seq_len 64 --label_len 0 --pred_len 64 --batch_size 12 --train_epoch 50 --lradj type4

python -u run.py --model convlstm --data OISST4  --freq w --root_path ./data/OISST/  --seq_len 96 --label_len 0 --pred_len 96 --batch_size 4 --train_epoch 50 --lradj type4 

python -u run.py --model gru --data OISST4  --freq w --root_path ./data/OISST/  --seq_len 32 --label_len 0 --pred_len 32 --batch_size 32 --train_epoch 50 --lradj type4

python -u run.py --model gru --data OISST4  --freq w --root_path ./data/OISST/  --seq_len 64 --label_len 0 --pred_len 64 --batch_size 32 --train_epoch 50 --lradj type4

python -u run.py --model gru --data OISST4  --freq w --root_path ./data/OISST/  --seq_len 96 --label_len 0 --pred_len 96 --batch_size 32 --train_epoch 50 --lradj type4