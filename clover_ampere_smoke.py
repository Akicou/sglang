import torch

results = {}

def report(name, ok, detail=""):
    results[name] = (ok, detail)
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {detail}")

device = "cuda"
cap = torch.cuda.get_device_capability(device)
print("compute_capability:", cap)

# ---- 1. conv kernel ----
try:
    from sglang.kernels.ops.mamba.clover_conv_triton import (
        supports_clover_local_memory_conv,
        clover_local_memory_conv,
    )
    channels = 256
    width = 4
    block_size = 4
    batch = 3
    num_tokens = batch * block_size
    mixed_qkv = torch.randn(num_tokens, channels, dtype=torch.bfloat16, device=device)
    conv_weights = torch.randn(channels, width, dtype=torch.bfloat16, device=device)
    conv_right_weight = torch.randn(channels, 1, width - 1, dtype=torch.bfloat16, device=device)
    bias = torch.randn(channels, dtype=torch.bfloat16, device=device)
    clean_conv_states = torch.randn(batch, channels, width - 1, dtype=torch.bfloat16, device=device)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=device)
    active_valid_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    supported = supports_clover_local_memory_conv(
        mixed_qkv, conv_weights=conv_weights, block_size=block_size, activation="silu"
    )
    if not supported:
        report("conv_gate", False, "supports_clover_local_memory_conv returned False on sm80")
    else:
        out = clover_local_memory_conv(
            mixed_qkv,
            clean_conv_states=clean_conv_states,
            cache_indices=cache_indices,
            conv_weights=conv_weights,
            bias=bias,
            activation="silu",
            block_size=block_size,
            active_valid_mask=active_valid_mask,
            conv_mode="independent_symmetric",
            conv_right_weight=conv_right_weight,
        )
        torch.cuda.synchronize()
        ok = torch.isfinite(out).all().item() and out.shape == mixed_qkv.shape
        report("conv_gate", True)
        report("conv_exec", ok, f"shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
except Exception as e:
    report("conv", False, f"{type(e).__name__}: {e}")

# ---- 2. local-memory attention kernel ----
try:
    from sglang.kernels.ops.attention.clover_gdn_triton import (
        supports_clover_local_memory_triton,
        clover_local_memory_triton,
    )
    block_size = 4
    batch = 3
    seq_len = batch * block_size
    num_q_heads = 4
    num_k_heads = 4
    num_v_heads = 8
    head_dim = 128
    query = torch.randn(1, seq_len, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(1, seq_len, num_k_heads, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(1, seq_len, num_v_heads, head_dim, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, seq_len, num_v_heads, dtype=torch.float32, device=device) * 0.01
    clean_ssm_states = torch.randn(batch, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16, device=device)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=device)
    active_valid_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    softmax_logit_scale = torch.ones(num_v_heads, dtype=torch.float32, device=device)
    softmax_history_bias = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
    softmax_history_c = torch.zeros(num_v_heads, dtype=torch.float32, device=device)

    supported = supports_clover_local_memory_triton(query, key, value, block_size=block_size)
    if not supported:
        report("attn_gate", False, "supports_clover_local_memory_triton returned False on sm80")
    else:
        out = clover_local_memory_triton(
            query, key, value, g,
            clean_ssm_states=clean_ssm_states,
            cache_indices=cache_indices,
            active_valid_mask=active_valid_mask,
            scale=head_dim ** -0.5,
            softmax_logit_scale=softmax_logit_scale,
            softmax_history_bias=softmax_history_bias,
            softmax_history_c=softmax_history_c,
            softmax_reference_block_size=32.0,
        )
        torch.cuda.synchronize()
        ok = torch.isfinite(out).all().item()
        report("attn_gate", True)
        report("attn_exec", ok, f"shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
except Exception as e:
    report("attn", False, f"{type(e).__name__}: {e}")

# ---- 3. MoE coreset kernel ----
try:
    from sglang.kernels.ops.moe.clover_coreset_triton import (
        supports_clover_block_coreset_triton,
        _apply_clover_block_coreset_triton,
    )
    num_experts = 256
    block_size = 4
    batch = 3
    num_tokens = batch * block_size
    top_k = 8
    top_p = 0.5
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device)
    generation_block_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    coreset_eligible = torch.ones(num_tokens, dtype=torch.bool, device=device)

    supported = supports_clover_block_coreset_triton(router_logits, top_k=top_k, block_size=block_size)
    if not supported:
        report("moe_gate", False, "supports_clover_block_coreset_triton returned False on sm80")
    else:
        out = _apply_clover_block_coreset_triton(
            router_logits,
            generation_block_ids=generation_block_ids,
            coreset_eligible=coreset_eligible,
            top_k=top_k,
            top_p=top_p,
            block_size=block_size,
        )
        torch.cuda.synchronize()
        ok = torch.isfinite(out[torch.isfinite(out) | (out == float("-inf"))]).numel() >= 0
        finite_or_neg_inf = torch.isfinite(out) | (out == float("-inf"))
        ok = bool(finite_or_neg_inf.all().item())
        report("moe_gate", True)
        report("moe_exec", ok, f"shape={tuple(out.shape)} all_finite_or_-inf={ok}")
except Exception as e:
    report("moe", False, f"{type(e).__name__}: {e}")

print("\n=== SUMMARY ===")
all_ok = True
for name, (ok, detail) in results.items():
    print(f"{name}: {'PASS' if ok else 'FAIL'} {detail}")
    all_ok = all_ok and ok
print("OVERALL:", "PASS" if all_ok else "FAIL")
