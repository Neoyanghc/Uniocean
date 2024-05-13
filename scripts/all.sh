
python -u run.py --model informerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3  --use_multi_scale --train_epoch 50

python -u run.py --model autoformerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3  --use_multi_scale --train_epoch 50

python -u run.py --model fedformerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --use_multi_scale --train_epoch 50 

python -u run.py --model informerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --use_multi_scale --train_epoch 50

python -u run.py --model autoformerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --use_multi_scale --train_epoch 50

python -u run.py --model fedformerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/  --seq_len 64 --label_len 32 --pred_len 32  --batch_size 20 --use_multi_scale --train_epoch 50 

python -u run.py --model informerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --use_multi_scale --train_epoch 50

python -u run.py --model autoformerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/   --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --use_multi_scale --train_epoch 50

python -u run.py --model fedformerUniOcean --data ALL2 --attn prob --freq w --root_path ./data/ALL/   --seq_len 96 --label_len 48 --pred_len 48 --lradj type4 --batch_size 12  --use_multi_scale --train_epoch 50 
