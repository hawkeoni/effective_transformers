# Effective Transformers
This is a comparison of transformers on listops task.

## How to run
Optional arguments - `--dataset_dir dataset --neptune`

**LSTM**
```bash
python train.py --serialization_dir test --d_model 512 --num_layers 2 \
--num_heads 8 --ff_dim 1024 --dropout 0.2 --model_type lstm \
--max_length 2010 --warmup_steps 1 --batch_size 8 \
--base_lr 0.001 --gpus 1 
```

**Transformer**
```bash
python train.py --serialization_dir custom_transformer --d_model 512 --num_layers 6 --num_heads 8 --ff_dim 2048 \
--dropout 0.2 --model_type transformer --max_length 2010 --warmup_steps 1000 --batch_size 32 --base_lr 0.00002 \
--weight_decay 0.99 --gpus 1

```