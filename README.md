# GPT Model - Generative Pre-trained Transformer Model
Design a chatbot system using GPT model.

## 1. Architecture
<figure align="center">
    <img src="./assets/gpt_model.png">
    <figcaption>GPT Architecture</figcaption>
</figure>

## 2. Setup
- pip install requirements.txt
- Pretraining Stage
- Fine-tunning Stage

## 3. Parameters
- vocab_size (token_size): Number of tokens.
- n: Number of Decoder Blocks.
- embedding_dim: Dimention of Word2Vec.
- heads: Number of heads in Multi-head attention.
- d_ff: Number of hidden neutron in Position Wise Feed Forward Networks.
- dropout_rate: Rate for Dropout Layer.
- eps: Epsilon for Layer Norm.
- activation: Activatin function in Position Wise Feed Forward Networks.
- learning_rate: Learning Rate of Optimizer ADam.
- checkpoint: Folder path of trained model.