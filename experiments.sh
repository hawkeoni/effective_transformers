python train.py --serialization_dir custom_transformer --d_model 512 --num_layers 6 --num_heads 8 --ff_dim 2048 \
--dropout 0.2 --model_type transformer --max_lengths 2010 --warmup_steps 1000 --batch_size 32 --base_lr 0.00002 \
--weight_decay 0.99


python train.py --serialization_dir pytorch_transformer --d_model 512 --num_layers 6 --num_heads 8 --ff_dim 2048 \
--dropout 0.2 --model_type pytorch_transformer --max_lengths 2010 --warmup_steps 1000 --batch_size 32 --base_lr 0.00002 \
--weight_decay 0.99

python train.py --serialization_dir lstm --d_model 512 --num_layers 2 \
--dropout 0.2 --model_type lstm --warmup_steps 100 --batch_size 32 --base_lr 0.001
