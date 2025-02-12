import pytest

from transformers import is_torch_available
from transformers.models.deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RotaryEmbedding,
)


if is_torch_available():
    import torch


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


@pytest.mark.parametrize(
    "batch_size, num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, q_lora_rank",
    [
        (2, 8, 12, 8, 12, 36, 48),
        (2, 1, 64, 8, 48, 32, 128),
        (1, 8, 12, 6, 16, 36, 48),
        (2, 8, 8, 16, 12, 2, 64),
    ]
)
def test_compare_training_vs_inference_mode(
    batch_size,
    num_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    kv_lora_rank,
    q_lora_rank,
):
    seq_len = 20
    dtype = torch.float32
    attn_implementation = "eager"
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        attn_implementation=attn_implementation,
        attention_bias=False,
    )

    attn=DeepseekV3Attention(config, layer_idx=0, weights_dtype=dtype)
    attn.apply(init_weights)
    rotary_emb = DeepseekV3RotaryEmbedding(config)

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size),
        dtype=dtype,
    )
    position_ids = torch.arange(seq_len, dtype=torch.int64).expand(batch_size, -1)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    debug_res1 = dict()
    output1, weights1 = attn._forward_training(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        debug_res=debug_res1,
    )
    debug_res2 = dict()
    output2, weights2 = attn._forward_inference(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        debug_res=debug_res2,
    )
    if attn_implementation == "eager":
        assert weights1 is not None and weights2 is not None
    for name, val1 in debug_res1.items():
        val2 = debug_res2.get(name)
        if val2 is not None:
            print(f"{name}: {val1.shape}")
            torch.testing.assert_close(val1, val2)
    torch.testing.assert_close(output1, output2)
    if weights1 is not None:
        torch.testing.assert_close(weights1, weights2)
