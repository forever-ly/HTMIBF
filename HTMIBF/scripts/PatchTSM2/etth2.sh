if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=1024
model_name=PatchTSM2

root_path_name=.././dataset/ETT-small
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
device=1
res_path="./results/1"
random_seed=2021

for pred_len in 96 192 336 720
do

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 20\
      --itr 1 --batch_size 128 --learning_rate 0.0005 --gpu $device \
      --weight_decay 0.01 \
      --t_layers 2 \
      --type split \
      --K  2\
      --tran_first \
      --patience 10 \
      --res_path $res_path \
      --temp 1 \
      --beta 1 \
      --share \
      --IB
done
