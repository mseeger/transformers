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


def test_compare_training_vs_inference_mode():
    batch_size = 2
    seq_len = 20
    dtype = torch.float32
    attn_implementation = "eager"
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=12,
        kv_lora_rank=36,
        q_lora_rank=48,
        attn_implementation=attn_implementation,
    )

    attn=DeepseekV3Attention(config, layer_idx=0)
    attn.apply(init_weights)
    rotary_emb = DeepseekV3RotaryEmbedding(config)

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size),
        dtype=dtype,
    )
    position_ids = torch.arange(seq_len, dtype=torch.int64).expand(batch_size, -1)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    output1, weights1 = attn._forward_training(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
    )
    output2, weights2 = attn._forward_inference(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
    )
    if attn_implementation == "eager":
        assert weights1 is not None and weights2 is not None
    torch.testing.assert_close(output1, output2)
    if weights1 is not None:
        torch.testing.assert_close(weights1, weights2)
