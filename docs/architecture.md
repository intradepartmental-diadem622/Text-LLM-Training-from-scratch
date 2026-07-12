# The model, and why it's built this way

The transformer here is small enough to read end to end, but it uses the same building
blocks as current open models rather than the 2017 originals. This page explains each
choice so the code isn't a black box.

## Rotary position embeddings (RoPE)

The original transformer added a learned vector to each token to encode its position. RoPE
instead *rotates* the query and key vectors by an angle proportional to their position
(`model/rope.py`). Because a rotation is applied to both sides of the attention dot product,
the score between a query at position *m* and a key at position *n* ends up depending only
on the offset *m − n*. That's a relative signal, and it lets the model handle sequences
longer than it trained on far more gracefully than a fixed lookup table.

## RMSNorm

LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm
(`model/norm.py`) drops the mean-centering and the bias, keeping only the rescale by
root-mean-square. It is cheaper and, in practice, just as stable, which is why the Llama
family adopted it. The normalization is done in float32 even under mixed precision so the
reduction doesn't lose precision.

## SwiGLU feed-forward

A plain MLP is `Linear → ReLU → Linear`. SwiGLU (`model/mlp.py`) splits the first
projection in two: one half is passed through SiLU and used as a gate on the other. Gated
activations consistently outperform ReLU at a matched parameter count, so the hidden width
is trimmed to about 2/3 of the classic 4× to keep the parameter budget the same.

## Grouped-query attention and the KV-cache

During generation, every past token's key and value vectors are cached so they're computed
once, not re-derived at each step (`model/attention.py`). That cache is the main memory
cost of decoding. Grouped-query attention lets several query heads share one key/value
head, which shrinks the cache and speeds up decoding with almost no quality loss. Set
`n_kv_head` equal to `n_head` for ordinary multi-head attention, or smaller for GQA.

## One attention kernel, everywhere

The fast path is `torch.nn.functional.scaled_dot_product_attention`. It runs on CPU, MPS,
and CUDA, and selects a FlashAttention kernel on capable GPUs, so there is one
implementation and no per-backend branching. A `naive` path (explicit scores, mask,
softmax, weighted sum) sits alongside it, selected by `ModelConfig.naive_attention`, so
the mechanics remain visible. A test checks that the two agree numerically.

## Weight tying and initialization

The input embedding and the output (LM head) share one weight matrix. They are inverse
operations over the same vocabulary, so tying them saves parameters and tends to help.
Residual output projections are initialized with a smaller standard deviation
(scaled by `1/sqrt(2 · n_blocks)`) so the variance of the residual stream stays roughly
constant as it flows through a deep stack. This is the GPT-2 initialization.

## Where hardware lives

Exactly one file, `device.py`, knows about CUDA, MPS, or CPU and picks the autocast dtype
(bf16 on modern CUDA, fp16 with a gradient scaler on older CUDA, fp32 elsewhere). Nothing
else in the codebase references `torch.cuda`. That's what keeps the same code correct on a
laptop and on a datacenter GPU.
