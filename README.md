# GPT Model - Generative Pre-trained Transformer Model
Design a chatbot system using GPT model.

## I. Architecture
<figure align="center">
    <img src="./assets/gpt_model.png">
</figure>

GPT Model uses multiples `Decoder Blocks` of `Transformer Architecuture`
Components inside Decoder Layer
- `Embedding Layer`
- `Positional Encoding`
- `Masked Multi-Head Attention`
- `Position Wise Feed-Forward Networks`
- `Residual Connection`

## II. Setup Environment
1. Make Sure you have installed Python
2. `Python version` in this Project: `3.10.9` 
3. `cd {project_folder}`
4. Install needed packages: `pip install requirements.txt`

## III. Parameters
- `vocab_size` `(token_size)`: Number of tokens.
- `n`: Number of Decoder Blocks.
- `embedding_dim`: Dimension of Word2Vec.
- `heads`: Number of heads in Multi-head attention.
- `d_ff`: Number of hidden neutron in Position Wise Feed Forward Networks.
- `dropout_rate`: Rate for Dropout Layer.
- `eps`: A value added to the denominator for numerical stability.
- `activation`: Activatin function in Position Wise Feed Forward Networks.
- `learning_rate`: Learning Rate of Optimizer.
- `checkpoint`: Folder path of trained model.

## IV. Dataset Setup

## V. Two Stages Training
In GPT Model has 2 stages of training:
- `Pretrain Stage`: Tranin model with large corpus to make model can see the next word if have a previous context (previous words).
- `Fine-tune Stage`: Train model for Specific Task with pretrained model.
1. Pre-training Stage
- For pretrain model: `python pretrain.py --data {path_clean_data} --tokenizer {path_tokenizer} --epochs {epochs} --batch_size {batch_size} --mini_batch {mini_batch} --checkpoint {path_checkpoint_folder}`
- Example: `python pretrain.py --data ./clean/pretrain_data.pkl --tokenizer ./pretrain/pretrain.pkl --epochs 5 --batch_size 64 --mini_batch 32 --checkpoint ./saved_models/pretrained.pt`

2. Fine-tune Stage
- For Fine-Tune model: `python train.py --data {folder_clean_data} --tokenizer {path_tokenizer} --epochs {epochs} --batch_size {batch_size} --mini_batch {mini_batch} --pretrained_model {checkpoint_pretrained_path} --checkpoint {checkpoint_path}`