python data_preparation/grpo.py \
    --data_source medqa \
    --split train \
    --data_path data/processed/medqa_train.csv \
    --save_path data/verl/medqa_full/train.parquet

python data_preparation/grpo.py \
    --data_source medqa \
    --split dev \
    --data_path data/processed/medqa_dev.csv \
    --save_path data/verl/medqa_full/dev.parquet

python data_preparation/grpo.py \
    --data_source medqa \
    --split train \
    --data_path data/processed/medqa_train_grpo.csv \
    --save_path data/verl/medqa_grpo/train.parquet

python data_preparation/grpo.py \
    --data_source medqa \
    --split dev \
    --data_path data/processed/medqa_dev.csv \
    --save_path data/verl/medqa_grpo/dev.parquet