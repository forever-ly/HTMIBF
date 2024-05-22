if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
device=1
res_path="./results/3"
seq_len=1024
model_name=PatchTSM2

root_path_name=.././dataset/ETT-small
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

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
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.5\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 --batch_size 128 --learning_rate 0.001 --gpu $device \
      --weight_decay 0.01 \
      --t_layers 2 \
      --type split \
      --K 2 \
      --tran_first \
      --res_path $res_path \
      --share \
      --temp 1.0 \
      --beta 0.1 \
      --IB
done