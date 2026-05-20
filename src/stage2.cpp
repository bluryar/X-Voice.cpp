#include "x_voice_internal.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <memory>
#include <string>
#include <sstream>
#include <utility>

namespace x_voice {
namespace {

void emit_progress(
        const ProgressCallback & progress_callback,
        const std::string & phase,
        int64_t current,
        int64_t total,
        const std::string & detail = {}) {
    if (progress_callback) progress_callback({phase, current, total, detail});
}

struct GraphArena {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocr = nullptr;

    explicit GraphArena(ggml_backend_buffer_type_t buft, size_t mem_size = 64ull * 1024ull * 1024ull) {
        ggml_init_params params{};
        params.mem_size = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("failed to initialize Stage-2 GGML context");
        allocr = ggml_gallocr_new(buft);
        if (!allocr) throw std::runtime_error("failed to create Stage-2 GGML graph allocator");
    }

    ~GraphArena() {
        if (allocr) ggml_gallocr_free(allocr);
        if (ctx) ggml_free(ctx);
    }
};

ggml_tensor * named(const char * name, ggml_tensor * tensor) {
    ggml_set_name(tensor, name);
    return tensor;
}

ggml_tensor * w(const TensorStore & store, const std::string & name) {
    return store.get(name);
}

ggml_tensor * bias_2d(ggml_context * ctx, ggml_tensor * bias) {
    return ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
}

ggml_tensor * bias_3d(ggml_context * ctx, ggml_tensor * bias) {
    return ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
}

ggml_tensor * bias_rows_3d(ggml_context * ctx, ggml_tensor * bias) {
    return ggml_reshape_3d(ctx, bias, bias->ne[0], 1, 1);
}

ggml_tensor * channels_2d(ggml_context * ctx, ggml_tensor * values) {
    return ggml_reshape_2d(ctx, values, values->ne[0], 1);
}

ggml_tensor * linear(ggml_context * ctx, const TensorStore & store, ggml_tensor * x, const std::string & prefix) {
    ggml_tensor * out = ggml_mul_mat(ctx, w(store, prefix + ".weight"), x);
    return ggml_add(ctx, out, bias_2d(ctx, w(store, prefix + ".bias")));
}

ggml_tensor * linear_nd(ggml_context * ctx, const TensorStore & store, ggml_tensor * x, const std::string & prefix) {
    ggml_tensor * out = ggml_mul_mat(ctx, w(store, prefix + ".weight"), x);
    if (out->ne[2] > 1 || out->ne[3] > 1) {
        return ggml_add(ctx, out, bias_rows_3d(ctx, w(store, prefix + ".bias")));
    }
    return ggml_add(ctx, out, bias_2d(ctx, w(store, prefix + ".bias")));
}

ggml_tensor * layer_norm_affine_2d(ggml_context * ctx, const TensorStore & store, ggml_tensor * x, const std::string & prefix) {
    ggml_tensor * cur = ggml_norm(ctx, x, 1e-6f);
    cur = ggml_mul(ctx, cur, channels_2d(ctx, w(store, prefix + ".weight")));
    return ggml_add(ctx, cur, bias_2d(ctx, w(store, prefix + ".bias")));
}

ggml_tensor * mish(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * softplus = ggml_log(ctx, ggml_scale_bias(ctx, ggml_exp(ctx, x), 1.0f, 1.0f));
    return ggml_mul(ctx, x, ggml_tanh(ctx, softplus));
}

ggml_tensor * slice_1d(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_1d(ctx, tensor, length, static_cast<size_t>(start) * ggml_element_size(tensor));
    return ggml_cont_1d(ctx, view, length);
}

ggml_tensor * slice_2d_rows(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_2d(
        ctx,
        tensor,
        length,
        tensor->ne[1],
        tensor->nb[1],
        static_cast<size_t>(start) * ggml_element_size(tensor));
    return ggml_cont_2d(ctx, view, length, view->ne[1]);
}

ggml_tensor * slice_3d_channels(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_3d(
        ctx,
        tensor,
        tensor->ne[0],
        length,
        tensor->ne[2],
        tensor->nb[1],
        tensor->nb[2],
        static_cast<size_t>(start) * tensor->nb[1]);
    return ggml_cont_3d(ctx, view, view->ne[0], view->ne[1], view->ne[2]);
}

ggml_tensor * slice_3d_branch_to_2d(ggml_context * ctx, ggml_tensor * tensor, int64_t branch_index) {
    ggml_tensor * view = ggml_view_2d(
        ctx,
        tensor,
        tensor->ne[0],
        tensor->ne[1],
        tensor->nb[1],
        static_cast<size_t>(branch_index) * tensor->nb[2]);
    return ggml_cont_2d(ctx, view, view->ne[0], view->ne[1]);
}

ggml_tensor * slice_conv_out_channels(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_3d(
        ctx,
        tensor,
        tensor->ne[0],
        tensor->ne[1],
        length,
        tensor->nb[1],
        tensor->nb[2],
        static_cast<size_t>(start) * tensor->nb[2]);
    return ggml_cont_3d(ctx, view, view->ne[0], view->ne[1], view->ne[2]);
}

ggml_tensor * conv_1d_f32(
        ggml_context * ctx,
        ggml_tensor * kernel,
        ggml_tensor * inp,
        int stride,
        int padding,
        int dilation) {
    ggml_tensor * im2col = ggml_im2col(
        ctx,
        kernel,
        inp,
        stride,
        0,
        padding,
        0,
        dilation,
        0,
        false,
        GGML_TYPE_F32);
    ggml_tensor * lhs = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * rhs = ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
    ggml_tensor * out = ggml_mul_mat(ctx, lhs, rhs);
    return ggml_reshape_3d(ctx, out, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

ggml_tensor * conv_1d_dw_f32(
        ggml_context * ctx,
        ggml_tensor * kernel,
        ggml_tensor * inp,
        int stride,
        int padding,
        int dilation) {
    ggml_tensor * inp_4d = ggml_reshape_4d(ctx, inp, inp->ne[0], 1, inp->ne[1], inp->ne[2]);
    ggml_tensor * im2col = ggml_im2col(
        ctx,
        kernel,
        inp_4d,
        stride,
        0,
        padding,
        0,
        dilation,
        0,
        false,
        GGML_TYPE_F32);
    ggml_tensor * cur = ggml_mul_mat(ctx, im2col, kernel);
    return ggml_reshape_3d(ctx, cur, cur->ne[0], cur->ne[2], 1);
}

ggml_tensor * conv_1d_groups_f32(
        ggml_context * ctx,
        const TensorStore & store,
        const std::string & prefix,
        ggml_tensor * inp,
        int groups,
        int stride,
        int padding,
        int dilation) {
    ggml_tensor * kernel = w(store, prefix + ".weight");
    ggml_tensor * bias = w(store, prefix + ".bias");
    const int64_t channels = inp->ne[1];
    const int64_t out_channels = bias->ne[0];
    if (channels % groups != 0 || out_channels % groups != 0) {
        std::ostringstream msg;
        msg << "grouped conv channel mismatch for " << prefix << ": channels=" << channels
            << " out_channels=" << out_channels << " groups=" << groups;
        throw std::runtime_error(msg.str());
    }
    const int64_t channels_per_group = channels / groups;
    const int64_t out_per_group = out_channels / groups;

    ggml_tensor * out = nullptr;
    for (int group_id = 0; group_id < groups; ++group_id) {
        ggml_tensor * in_group = slice_3d_channels(ctx, inp, group_id * channels_per_group, channels_per_group);
        ggml_tensor * kernel_group = slice_conv_out_channels(ctx, kernel, group_id * out_per_group, out_per_group);
        ggml_tensor * bias_group = slice_1d(ctx, bias, group_id * out_per_group, out_per_group);
        ggml_tensor * cur = conv_1d_f32(ctx, kernel_group, in_group, stride, padding, dilation);
        cur = ggml_add(ctx, cur, bias_3d(ctx, bias_group));
        out = out == nullptr ? cur : ggml_concat(ctx, out, cur, 1);
    }
    return out;
}

ggml_tensor * conv_position_embedding(
        ggml_context * ctx,
        const TensorStore & store,
        ggml_tensor * x,
        const std::string & conv0_prefix,
        const std::string & conv2_prefix,
        int seq_len,
        int groups) {
    const int64_t dim = x->ne[0];
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, x));
    cur = ggml_reshape_3d(ctx, cur, seq_len, dim, 1);

    cur = conv_1d_groups_f32(ctx, store, conv0_prefix, cur, groups, 1, 15, 1);
    ggml_tensor * cur_2d = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, cur, seq_len, dim)));
    cur_2d = mish(ctx, cur_2d);

    cur = ggml_cont(ctx, ggml_transpose(ctx, cur_2d));
    cur = ggml_reshape_3d(ctx, cur, seq_len, dim, 1);
    cur = conv_1d_groups_f32(ctx, store, conv2_prefix, cur, groups, 1, 15, 1);
    cur_2d = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, cur, seq_len, dim)));
    return mish(ctx, cur_2d);
}

ggml_tensor * conv_position_embedding_branchwise_3d(
        ggml_context * ctx,
        const TensorStore & store,
        ggml_tensor * x,
        const std::string & conv0_prefix,
        const std::string & conv2_prefix,
        int seq_len,
        int groups) {
    const int64_t dim = x->ne[0];
    const int64_t branch_count = x->ne[2];
    ggml_tensor * out = nullptr;
    for (int64_t branch_index = 0; branch_index < branch_count; ++branch_index) {
        ggml_tensor * branch = slice_3d_branch_to_2d(ctx, x, branch_index);
        ggml_tensor * cur = conv_position_embedding(
            ctx,
            store,
            branch,
            conv0_prefix,
            conv2_prefix,
            seq_len,
            groups);
        cur = ggml_reshape_3d(ctx, cur, dim, seq_len, 1);
        out = out == nullptr ? cur : ggml_concat(ctx, out, cur, 2);
    }
    return out;
}

ggml_tensor * grn_2d(ggml_context * ctx, const TensorStore & store, ggml_tensor * x, const std::string & prefix) {
    const int64_t dim = x->ne[0];
    ggml_tensor * x_time_major = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor * gx = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, x_time_major)));
    ggml_tensor * channel_mean = ggml_scale_bias(ctx, ggml_scale(ctx, ggml_sum(ctx, gx), 1.0f / static_cast<float>(dim)), 1.0f, 1e-6f);
    ggml_tensor * nx = ggml_cont(ctx, ggml_transpose(ctx, ggml_div(ctx, gx, channel_mean)));
    ggml_tensor * x_scaled = ggml_mul(ctx, x, ggml_repeat(ctx, nx, x));
    ggml_tensor * gamma = channels_2d(ctx, w(store, prefix + ".grn.gamma"));
    ggml_tensor * beta = bias_2d(ctx, w(store, prefix + ".grn.beta"));
    return ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, x_scaled, gamma), beta), x);
}

ggml_tensor * convnext_v2_text_block_2d(
        ggml_context * ctx,
        const TensorStore & store,
        ggml_tensor * x,
        const std::string & prefix,
        int seq_len) {
    ggml_tensor * residual = x;
    const int64_t dim = x->ne[0];
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, x));
    cur = ggml_reshape_3d(ctx, cur, seq_len, dim, 1);
    cur = conv_1d_dw_f32(ctx, w(store, prefix + ".dwconv.weight"), cur, 1, 3, 1);
    cur = ggml_add(ctx, cur, bias_3d(ctx, w(store, prefix + ".dwconv.bias")));
    cur = ggml_reshape_2d(ctx, cur, seq_len, dim);
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
    cur = layer_norm_affine_2d(ctx, store, cur, prefix + ".norm");
    cur = linear(ctx, store, cur, prefix + ".pwconv1");
    cur = ggml_gelu_erf(ctx, cur);
    cur = grn_2d(ctx, store, cur, prefix);
    cur = linear(ctx, store, cur, prefix + ".pwconv2");
    return ggml_add(ctx, residual, cur);
}

ggml_tensor * gelu_tanh_approx(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * cubic = ggml_mul(ctx, x, ggml_scale_bias(ctx, ggml_sqr(ctx, x), 0.044715f, 1.0f));
    ggml_tensor * inner = ggml_scale(ctx, cubic, 0.7978845608028654f);
    return ggml_scale(ctx, ggml_mul(ctx, x, ggml_scale_bias(ctx, ggml_tanh(ctx, inner), 1.0f, 1.0f)), 0.5f);
}

ggml_tensor * feed_forward_tanh_gelu(
        ggml_context * ctx,
        const TensorStore & store,
        ggml_tensor * x,
        const std::string & prefix) {
    ggml_tensor * up = linear(ctx, store, x, prefix + ".up");
    return linear(ctx, store, gelu_tanh_approx(ctx, up), prefix + ".down");
}

ggml_tensor * feed_forward_tanh_gelu_nd(
        ggml_context * ctx,
        const TensorStore & store,
        ggml_tensor * x,
        const std::string & prefix) {
    ggml_tensor * up = linear_nd(ctx, store, x, prefix + ".up");
    return linear_nd(ctx, store, gelu_tanh_approx(ctx, up), prefix + ".down");
}

ggml_tensor * self_attention_2d(
        ggml_context * ctx,
        const TensorStore & store,
        ggml_tensor * x,
        const std::string & prefix,
        int head_count,
        int head_dim,
        ggml_tensor * positions) {
    const int64_t seq_len = x->ne[1];
    ggml_tensor * q = linear(ctx, store, x, prefix + ".q");
    ggml_tensor * k = linear(ctx, store, x, prefix + ".k");
    ggml_tensor * v = linear(ctx, store, x, prefix + ".v");
    q = ggml_reshape_3d(ctx, q, head_dim, head_count, seq_len);
    k = ggml_reshape_3d(ctx, k, head_dim, head_count, seq_len);
    v = ggml_reshape_3d(ctx, v, head_dim, head_count, seq_len);
    q = ggml_rope(ctx, q, positions, head_dim, 0);
    k = ggml_rope(ctx, k, positions, head_dim, 0);

    ggml_tensor * q4 = ggml_permute(ctx, ggml_reshape_4d(ctx, q, head_dim, head_count, seq_len, 1), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx, ggml_reshape_4d(ctx, k, head_dim, head_count, seq_len, 1), 0, 2, 1, 3);
    ggml_tensor * scores = ggml_mul_mat(ctx, k4, q4);
    ggml_tensor * mask = ggml_scale(ctx, scores, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

    ggml_tensor * v4 = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, head_dim, head_count, seq_len, 1), 1, 2, 0, 3));
    ggml_tensor * attn = ggml_mul_mat(ctx, v4, probs);
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
    attn = ggml_cont_2d(ctx, attn, head_dim * head_count, seq_len);
    return linear(ctx, store, attn, prefix + ".o");
}

ggml_tensor * self_attention_3d(
        ggml_context * ctx,
        const TensorStore & store,
        ggml_tensor * x,
        const std::string & prefix,
        int head_count,
        int head_dim,
        ggml_tensor * positions) {
    const int64_t seq_len = x->ne[1];
    const int64_t branch_count = x->ne[2];
    ggml_tensor * q = linear_nd(ctx, store, x, prefix + ".q");
    ggml_tensor * k = linear_nd(ctx, store, x, prefix + ".k");
    ggml_tensor * v = linear_nd(ctx, store, x, prefix + ".v");
    q = ggml_reshape_4d(ctx, q, head_dim, head_count, seq_len, branch_count);
    k = ggml_reshape_4d(ctx, k, head_dim, head_count, seq_len, branch_count);
    v = ggml_reshape_4d(ctx, v, head_dim, head_count, seq_len, branch_count);
    q = ggml_rope(ctx, q, positions, head_dim, 0);
    k = ggml_rope(ctx, k, positions, head_dim, 0);

    ggml_tensor * q4 = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * scores = ggml_mul_mat(ctx, k4, q4);
    ggml_tensor * mask = ggml_scale(ctx, scores, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

    ggml_tensor * v4 = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * attn = ggml_mul_mat(ctx, v4, probs);
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
    attn = ggml_cont_3d(ctx, attn, head_dim * head_count, seq_len, branch_count);
    return linear_nd(ctx, store, attn, prefix + ".o");
}

ggml_tensor * dit_block_2d(
        ggml_context * ctx,
        const TensorStore & store,
        const XVoiceSpec & spec,
        ggml_tensor * x,
        ggml_tensor * time_embed,
        int block_index,
        ggml_tensor * positions) {
    const std::string prefix = "stage2.blk." + std::to_string(block_index);
    const int64_t hidden_size = x->ne[0];
    ggml_tensor * ada = linear(ctx, store, ggml_silu(ctx, time_embed), prefix + ".attn_norm");
    ggml_tensor * shift_msa = slice_2d_rows(ctx, ada, 0, hidden_size);
    ggml_tensor * scale_msa = slice_2d_rows(ctx, ada, hidden_size, hidden_size);
    ggml_tensor * gate_msa = slice_2d_rows(ctx, ada, hidden_size * 2, hidden_size);
    ggml_tensor * shift_mlp = slice_2d_rows(ctx, ada, hidden_size * 3, hidden_size);
    ggml_tensor * scale_mlp = slice_2d_rows(ctx, ada, hidden_size * 4, hidden_size);
    ggml_tensor * gate_mlp = slice_2d_rows(ctx, ada, hidden_size * 5, hidden_size);

    ggml_tensor * norm = ggml_norm(ctx, x, 1e-6f);
    norm = ggml_add(ctx, ggml_mul(ctx, norm, ggml_scale_bias(ctx, scale_msa, 1.0f, 1.0f)), shift_msa);
    ggml_tensor * attn = self_attention_2d(
        ctx,
        store,
        norm,
        prefix + ".attn",
        spec.stage2.head_count,
        spec.stage2.head_dim,
        positions);
    ggml_tensor * cur = ggml_add(ctx, x, ggml_mul(ctx, attn, gate_msa));

    norm = ggml_norm(ctx, cur, 1e-6f);
    norm = ggml_add(ctx, ggml_mul(ctx, norm, ggml_scale_bias(ctx, scale_mlp, 1.0f, 1.0f)), shift_mlp);
    ggml_tensor * ff = feed_forward_tanh_gelu(ctx, store, norm, prefix + ".ff");
    return ggml_add(ctx, cur, ggml_mul(ctx, ff, gate_mlp));
}

ggml_tensor * dit_block_3d(
        ggml_context * ctx,
        const TensorStore & store,
        const XVoiceSpec & spec,
        ggml_tensor * x,
        ggml_tensor * time_embed,
        int block_index,
        ggml_tensor * positions) {
    const std::string prefix = "stage2.blk." + std::to_string(block_index);
    const int64_t hidden_size = x->ne[0];
    ggml_tensor * ada = linear(ctx, store, ggml_silu(ctx, time_embed), prefix + ".attn_norm");
    ggml_tensor * shift_msa = slice_2d_rows(ctx, ada, 0, hidden_size);
    ggml_tensor * scale_msa = slice_2d_rows(ctx, ada, hidden_size, hidden_size);
    ggml_tensor * gate_msa = slice_2d_rows(ctx, ada, hidden_size * 2, hidden_size);
    ggml_tensor * shift_mlp = slice_2d_rows(ctx, ada, hidden_size * 3, hidden_size);
    ggml_tensor * scale_mlp = slice_2d_rows(ctx, ada, hidden_size * 4, hidden_size);
    ggml_tensor * gate_mlp = slice_2d_rows(ctx, ada, hidden_size * 5, hidden_size);

    ggml_tensor * norm = ggml_norm(ctx, x, 1e-6f);
    norm = ggml_add(ctx, ggml_mul(ctx, norm, ggml_scale_bias(ctx, scale_msa, 1.0f, 1.0f)), shift_msa);
    ggml_tensor * attn = self_attention_3d(
        ctx,
        store,
        norm,
        prefix + ".attn",
        spec.stage2.head_count,
        spec.stage2.head_dim,
        positions);
    ggml_tensor * cur = ggml_add(ctx, x, ggml_mul(ctx, attn, gate_msa));

    norm = ggml_norm(ctx, cur, 1e-6f);
    norm = ggml_add(ctx, ggml_mul(ctx, norm, ggml_scale_bias(ctx, scale_mlp, 1.0f, 1.0f)), shift_mlp);
    ggml_tensor * ff = feed_forward_tanh_gelu_nd(ctx, store, norm, prefix + ".ff");
    return ggml_add(ctx, cur, ggml_mul(ctx, ff, gate_mlp));
}

ggml_tensor * stage2_output_mel(
        ggml_context * ctx,
        const TensorStore & store,
        const XVoiceSpec & spec,
        ggml_tensor * x,
        ggml_tensor * time_embed) {
    ggml_tensor * ada = linear(ctx, store, ggml_silu(ctx, time_embed), "stage2.out_norm");
    ggml_tensor * scale = slice_2d_rows(ctx, ada, 0, spec.stage2.hidden_size);
    ggml_tensor * shift = slice_2d_rows(ctx, ada, spec.stage2.hidden_size, spec.stage2.hidden_size);
    ggml_tensor * norm = ggml_norm(ctx, x, 1e-6f);
    norm = ggml_add(ctx, ggml_mul(ctx, norm, ggml_scale_bias(ctx, scale, 1.0f, 1.0f)), shift);
    return linear(ctx, store, norm, "stage2.out");
}

ggml_tensor * stage2_output_mel_3d(
        ggml_context * ctx,
        const TensorStore & store,
        const XVoiceSpec & spec,
        ggml_tensor * x,
        ggml_tensor * time_embed) {
    ggml_tensor * ada = linear(ctx, store, ggml_silu(ctx, time_embed), "stage2.out_norm");
    ggml_tensor * scale = slice_2d_rows(ctx, ada, 0, spec.stage2.hidden_size);
    ggml_tensor * shift = slice_2d_rows(ctx, ada, spec.stage2.hidden_size, spec.stage2.hidden_size);
    ggml_tensor * norm = ggml_norm(ctx, x, 1e-6f);
    norm = ggml_add(ctx, ggml_mul(ctx, norm, ggml_scale_bias(ctx, scale, 1.0f, 1.0f)), shift);
    return linear_nd(ctx, store, norm, "stage2.out");
}

ggml_tensor * stage2_conditioned_batch_mel(
        ggml_context * ctx,
        const TensorStore & store,
        const XVoiceSpec & spec,
        ggml_tensor * x,
        ggml_tensor * cond,
        ggml_tensor * text,
        ggml_tensor * time_embed,
        ggml_tensor * positions,
        int seq_len) {
    ggml_tensor * cur = ggml_concat(ctx, x, cond, 0);
    cur = ggml_concat(ctx, cur, text, 0);
    cur = linear_nd(ctx, store, cur, "stage2.inp.proj");
    ggml_tensor * pos_embed = conv_position_embedding_branchwise_3d(
        ctx,
        store,
        cur,
        "stage2.inp.pos.0",
        "stage2.inp.pos.2",
        seq_len,
        spec.stage2.input_conv_pos_groups);
    cur = ggml_add(ctx, cur, pos_embed);

    for (int index = 0; index < spec.stage2.block_count; ++index) {
        cur = dit_block_3d(ctx, store, spec, cur, time_embed, index, positions);
    }
    return stage2_output_mel_3d(ctx, store, spec, cur, time_embed);
}

std::vector<int32_t> positions(int seq_len) {
    std::vector<int32_t> out(static_cast<size_t>(seq_len));
    for (int i = 0; i < seq_len; ++i) out[static_cast<size_t>(i)] = i;
    return out;
}

void check_time_inputs(
        const XVoiceSpec & spec,
        const std::vector<float> & time_hidden_ggml_order,
        const std::vector<int32_t> & time_language_ids,
        int batch_size) {
    if (batch_size <= 0) throw std::runtime_error("Stage-2 time batch_size must be positive");
    const size_t expected_hidden = static_cast<size_t>(spec.stage2.time_freq_embedding_length) * static_cast<size_t>(batch_size);
    if (time_hidden_ggml_order.size() != expected_hidden) {
        std::ostringstream msg;
        msg << "Stage-2 time_hidden size mismatch: got " << time_hidden_ggml_order.size()
            << ", expected " << expected_hidden << " for GGML ne=("
            << spec.stage2.time_freq_embedding_length << ", " << batch_size << ")";
        throw std::runtime_error(msg.str());
    }
    if (time_language_ids.size() != static_cast<size_t>(batch_size)) {
        throw std::runtime_error("Stage-2 time_language_ids size must match batch_size");
    }
    for (int32_t id : time_language_ids) {
        if (id < 0 || id >= static_cast<int32_t>(spec.languages.size())) {
            throw std::runtime_error("Stage-2 time_language_ids contains an out-of-range language id");
        }
    }
}

void check_stage2_block_arrays(
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_index) {
    if (seq_len <= 0) throw std::runtime_error("Stage-2 block seq_len must be positive");
    if (block_index < 0 || block_index >= spec.stage2.block_count) {
        throw std::runtime_error("Stage-2 block_index is outside the model block range");
    }
    const size_t x_count = static_cast<size_t>(spec.stage2.hidden_size) * static_cast<size_t>(seq_len);
    if (x_ggml_order.size() != x_count) {
        throw std::runtime_error("Stage-2 block x size must match GGML ne=(hidden_size, seq_len)");
    }
    if (time_embed_ggml_order.size() != static_cast<size_t>(spec.stage2.hidden_size)) {
        throw std::runtime_error("Stage-2 block time_embed size must match GGML ne=(hidden_size, 1)");
    }
}

void check_stage2_stack_arrays(
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_count) {
    if (block_count <= 0 || block_count > spec.stage2.block_count) {
        throw std::runtime_error("Stage-2 stack block_count is outside the model block range");
    }
    check_stage2_block_arrays(spec, x_ggml_order, time_embed_ggml_order, seq_len, 0);
}

void check_stage2_text_arrays(
        const XVoiceSpec & spec,
        const std::vector<int32_t> & text_ids_plus_one,
        const std::vector<int32_t> & language_ids,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        const std::vector<float> & text_keep_mask_ggml_order,
        const std::vector<float> & text_pos_embed_ggml_order,
        int seq_len) {
    if (seq_len <= 0) throw std::runtime_error("Stage-2 text seq_len must be positive");
    if (text_ids_plus_one.size() != static_cast<size_t>(seq_len)) {
        throw std::runtime_error("Stage-2 text_ids_plus_one size must match seq_len");
    }
    if (language_ids.size() != static_cast<size_t>(seq_len)) {
        throw std::runtime_error("Stage-2 text language_ids size must match seq_len");
    }
    if (text_no_lang_fusion_mask_ggml_order.size() != static_cast<size_t>(seq_len) ||
            text_keep_mask_ggml_order.size() != static_cast<size_t>(seq_len)) {
        throw std::runtime_error("Stage-2 text masks must match GGML ne=(1, seq_len)");
    }
    const size_t pos_count = static_cast<size_t>(spec.stage2.text_dim) * static_cast<size_t>(seq_len);
    if (text_pos_embed_ggml_order.size() != pos_count) {
        throw std::runtime_error("Stage-2 text_pos_embed size must match GGML ne=(text_dim, seq_len)");
    }
    for (int32_t id : text_ids_plus_one) {
        if (id < 0 || id >= static_cast<int32_t>(spec.stage2.text_embedding_row_count)) {
            throw std::runtime_error("Stage-2 text_ids_plus_one contains an out-of-range embedding row");
        }
    }
    for (int32_t id : language_ids) {
        if (id < 0 || id > static_cast<int32_t>(spec.languages.size())) {
            throw std::runtime_error("Stage-2 text language_ids contains an out-of-range language id");
        }
    }
}

void check_stage2_input_arrays(
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        int seq_len) {
    if (seq_len <= 0) throw std::runtime_error("Stage-2 input seq_len must be positive");
    const size_t mel_count = static_cast<size_t>(spec.audio.mel_channel_count) * static_cast<size_t>(seq_len);
    const size_t text_count = static_cast<size_t>(spec.stage2.text_dim) * static_cast<size_t>(seq_len);
    if (x_ggml_order.size() != mel_count || cond_ggml_order.size() != mel_count) {
        throw std::runtime_error("Stage-2 x/cond input sizes must match GGML ne=(mel_channel_count, seq_len)");
    }
    if (text_embed_ggml_order.size() != text_count) {
        throw std::runtime_error("Stage-2 text_embed input size must match GGML ne=(text_dim, seq_len)");
    }
}

void check_stage2_conditioned_batch_arrays(
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int branch_count) {
    if (seq_len <= 0) throw std::runtime_error("Stage-2 conditioned batch seq_len must be positive");
    if (branch_count <= 0) throw std::runtime_error("Stage-2 conditioned batch branch_count must be positive");
    const size_t mel_count =
        static_cast<size_t>(spec.audio.mel_channel_count) *
        static_cast<size_t>(seq_len) *
        static_cast<size_t>(branch_count);
    const size_t text_count =
        static_cast<size_t>(spec.stage2.text_dim) *
        static_cast<size_t>(seq_len) *
        static_cast<size_t>(branch_count);
    if (x_ggml_order.size() != mel_count || cond_ggml_order.size() != mel_count) {
        throw std::runtime_error("Stage-2 conditioned batch x/cond sizes must match GGML ne=(mel_channel_count, seq_len, branch_count)");
    }
    if (text_embed_ggml_order.size() != text_count) {
        throw std::runtime_error("Stage-2 conditioned batch text_embed size must match GGML ne=(text_dim, seq_len, branch_count)");
    }
    if (time_embed_ggml_order.size() != static_cast<size_t>(spec.stage2.hidden_size)) {
        throw std::runtime_error("Stage-2 conditioned batch time_embed size must match GGML ne=(hidden_size, 1)");
    }
}

void check_stage2_input_batch_arrays(
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        int seq_len,
        int branch_count) {
    if (seq_len <= 0) throw std::runtime_error("Stage-2 input batch seq_len must be positive");
    if (branch_count <= 0) throw std::runtime_error("Stage-2 input batch branch_count must be positive");
    const size_t mel_count =
        static_cast<size_t>(spec.audio.mel_channel_count) *
        static_cast<size_t>(seq_len) *
        static_cast<size_t>(branch_count);
    const size_t text_count =
        static_cast<size_t>(spec.stage2.text_dim) *
        static_cast<size_t>(seq_len) *
        static_cast<size_t>(branch_count);
    if (x_ggml_order.size() != mel_count || cond_ggml_order.size() != mel_count) {
        throw std::runtime_error("Stage-2 input batch x/cond sizes must match GGML ne=(mel_channel_count, seq_len, branch_count)");
    }
    if (text_embed_ggml_order.size() != text_count) {
        throw std::runtime_error("Stage-2 input batch text_embed size must match GGML ne=(text_dim, seq_len, branch_count)");
    }
}

std::vector<float> sinus_position_embedding(float timestep, int dim, float scale = 1000.0f) {
    if (dim <= 0 || dim % 2 != 0) {
        throw std::runtime_error("Stage-2 sinus position embedding dim must be a positive even value");
    }
    const int half_dim = dim / 2;
    const float exponent = std::log(10000.0f) / static_cast<float>(half_dim - 1);
    std::vector<float> out(static_cast<size_t>(dim));
    for (int i = 0; i < half_dim; ++i) {
        const float frequency = std::exp(static_cast<float>(i) * -exponent);
        const float phase = scale * timestep * frequency;
        out[static_cast<size_t>(i)] = std::sin(phase);
        out[static_cast<size_t>(half_dim + i)] = std::cos(phase);
    }
    return out;
}

std::pair<float, float> cfg_schedule_values(
        float t,
        float cfg_strength,
        float cfg_strength2,
        const std::string & cfg_schedule,
        float cfg_decay_time) {
    float current_cfg = cfg_strength;
    float current_cfg2 = cfg_strength2;
    if (cfg_schedule == "square") {
        if (t > cfg_decay_time) {
            const float scale = (1.0f - t) * (1.0f - t);
            current_cfg *= scale;
            current_cfg2 *= scale;
        }
    } else if (cfg_schedule == "cosine") {
        if (t > cfg_decay_time) {
            const float normalized_t = (t - cfg_decay_time) / (1.0f - cfg_decay_time);
            const float pi = 3.14159265358979323846f;
            const float scale = std::cos(0.5f * pi * normalized_t);
            current_cfg *= scale;
            current_cfg2 *= scale;
        }
    } else if (!cfg_schedule.empty() && cfg_schedule != "none") {
        throw std::runtime_error("unsupported Stage-2 cfg_schedule: " + cfg_schedule);
    }
    return {current_cfg, current_cfg2};
}

std::vector<float> apply_conditioning_region(
        const XVoiceSpec & spec,
        const std::vector<float> & sampled,
        const std::vector<float> & cond,
        const std::vector<float> & cond_mask,
        int seq_len) {
    std::vector<float> out = sampled;
    for (int frame = 0; frame < seq_len; ++frame) {
        if (cond_mask[static_cast<size_t>(frame)] == 0.0f) continue;
        const size_t offset = static_cast<size_t>(frame) * static_cast<size_t>(spec.audio.mel_channel_count);
        std::copy(
            cond.begin() + static_cast<std::ptrdiff_t>(offset),
            cond.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(spec.audio.mel_channel_count)),
            out.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    return out;
}

std::vector<int32_t> cfg_null_text_language_ids(
        const XVoiceSpec & spec,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        int seq_len) {
    std::vector<int32_t> out(static_cast<size_t>(seq_len));
    const int32_t unknown_language_id = static_cast<int32_t>(spec.languages.size());
    for (int index = 0; index < seq_len; ++index) {
        out[static_cast<size_t>(index)] =
            text_no_lang_fusion_mask_ggml_order[static_cast<size_t>(index)] != 0.0f ? 0 : unknown_language_id;
    }
    return out;
}

std::vector<float> zero_like(const std::vector<float> & values) {
    return std::vector<float>(values.size(), 0.0f);
}

std::vector<float> pack_ggml_order_branches(const std::vector<const std::vector<float> *> & branches, size_t branch_size) {
    if (branches.empty()) throw std::runtime_error("cannot pack zero Stage-2 CFG branches");
    std::vector<float> out(branch_size * branches.size());
    for (size_t branch = 0; branch < branches.size(); ++branch) {
        if (branches[branch] == nullptr || branches[branch]->size() != branch_size) {
            throw std::runtime_error("Stage-2 CFG branch size mismatch while packing GGML ne2 batch");
        }
        std::copy(
            branches[branch]->begin(),
            branches[branch]->end(),
            out.begin() + static_cast<std::ptrdiff_t>(branch * branch_size));
    }
    return out;
}

void check_stage2_no_cfg_sampler_arrays(
        const XVoiceSpec & spec,
        const std::vector<int32_t> & text_ids_plus_one,
        const std::vector<int32_t> & text_language_ids,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        const std::vector<float> & text_keep_mask_ggml_order,
        const std::vector<float> & text_pos_embed_ggml_order,
        const std::vector<int32_t> & time_language_ids,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & cond_mask_ggml_order,
        const std::vector<float> & fixed_noise_ggml_order,
        const std::vector<float> & sampler_timesteps,
        int seq_len,
        int step_count) {
    check_stage2_text_arrays(
        spec,
        text_ids_plus_one,
        text_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        seq_len);
    const size_t mel_count = static_cast<size_t>(spec.audio.mel_channel_count) * static_cast<size_t>(seq_len);
    if (cond_ggml_order.size() != mel_count || fixed_noise_ggml_order.size() != mel_count) {
        throw std::runtime_error("Stage-2 sampler cond/fixed_noise sizes must match GGML ne=(mel_channel_count, seq_len)");
    }
    if (cond_mask_ggml_order.size() != static_cast<size_t>(seq_len)) {
        throw std::runtime_error("Stage-2 sampler cond_mask size must match GGML ne=(1, seq_len)");
    }
    if (time_language_ids.size() != 1) {
        throw std::runtime_error("Stage-2 sampler currently supports one time language id");
    }
    if (time_language_ids[0] < 0 || time_language_ids[0] >= static_cast<int32_t>(spec.languages.size())) {
        throw std::runtime_error("Stage-2 sampler time language id is out of range");
    }
    if (step_count <= 0) {
        throw std::runtime_error("Stage-2 sampler step_count must be positive");
    }
    if (sampler_timesteps.size() < static_cast<size_t>(step_count + 1)) {
        throw std::runtime_error("Stage-2 sampler_timesteps must contain step_count + 1 values");
    }
}

class Stage2DitForwardRunner {
public:
    Stage2DitForwardRunner(
            ggml_backend_t backend,
            ggml_backend_buffer_type_t buft,
            const TensorStore & store,
            const XVoiceSpec & spec,
            int seq_len,
            int block_count)
        : backend_(backend),
          spec_(spec),
          seq_len_(seq_len),
          block_count_(block_count),
          arena_(buft, 1536ull * 1024ull * 1024ull) {
        if (seq_len_ <= 0) throw std::runtime_error("Stage-2 forward runner seq_len must be positive");
        if (block_count_ <= 0 || block_count_ > spec_.stage2.block_count) {
            throw std::runtime_error("Stage-2 forward runner block_count is outside the model block range");
        }

        x_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.stage2.hidden_size, seq_len_);
        ggml_set_input(named("xvoice.stage2.forward.x", x_));
        time_embed_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.stage2.hidden_size, 1);
        ggml_set_input(named("xvoice.stage2.forward.time_embed", time_embed_));
        pos_ = ggml_new_tensor_1d(arena_.ctx, GGML_TYPE_I32, seq_len_);
        ggml_set_input(named("xvoice.stage2.forward.positions", pos_));

        ggml_tensor * out = x_;
        for (int index = 0; index < block_count_; ++index) {
            out = dit_block_2d(arena_.ctx, store, spec_, out, time_embed_, index, pos_);
        }
        mel_ = named("xvoice.stage2.forward.mel", stage2_output_mel(arena_.ctx, store, spec_, out, time_embed_));
        ggml_set_output(mel_);

        graph_ = ggml_new_graph_custom(arena_.ctx, 70000, false);
        ggml_build_forward_expand(graph_, mel_);
        if (!ggml_gallocr_reserve(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to reserve Stage-2 DiT forward graph");
        }
        ggml_gallocr_alloc_graph(arena_.allocr, graph_);

        pos_data_ = positions(seq_len_);
        ggml_backend_tensor_set(pos_, pos_data_.data(), 0, ggml_nbytes(pos_));
    }

    FloatTensor run(
            const std::vector<float> & x_ggml_order,
            const std::vector<float> & time_embed_ggml_order) {
        check_stage2_stack_arrays(spec_, x_ggml_order, time_embed_ggml_order, seq_len_, block_count_);

        if (!ggml_gallocr_alloc_graph(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to allocate Stage-2 DiT forward graph");
        }
        ggml_backend_tensor_set(pos_, pos_data_.data(), 0, ggml_nbytes(pos_));
        ggml_backend_tensor_set(x_, x_ggml_order.data(), 0, ggml_nbytes(x_));
        ggml_backend_tensor_set(time_embed_, time_embed_ggml_order.data(), 0, ggml_nbytes(time_embed_));

        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 DiT forward graph");

        FloatTensor result;
        result.ne = {mel_->ne[0], mel_->ne[1], mel_->ne[2], mel_->ne[3]};
        result.data.resize(static_cast<size_t>(ggml_nelements(mel_)));
        ggml_backend_tensor_get(mel_, result.data.data(), 0, ggml_nbytes(mel_));
        return result;
    }

private:
    ggml_backend_t backend_ = nullptr;
    const XVoiceSpec & spec_;
    int seq_len_ = 0;
    int block_count_ = 0;
    GraphArena arena_;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * time_embed_ = nullptr;
    ggml_tensor * pos_ = nullptr;
    ggml_tensor * mel_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::vector<int32_t> pos_data_;
};

class Stage2TimeEmbeddingRunner {
public:
    Stage2TimeEmbeddingRunner(
            ggml_backend_t backend,
            ggml_backend_buffer_type_t buft,
            const TensorStore & store,
            const XVoiceSpec & spec,
            int batch_size)
        : backend_(backend),
          spec_(spec),
          batch_size_(batch_size),
          arena_(buft) {
        if (batch_size_ <= 0) throw std::runtime_error("Stage-2 time runner batch_size must be positive");
        time_hidden_ = ggml_new_tensor_2d(
            arena_.ctx,
            GGML_TYPE_F32,
            spec_.stage2.time_freq_embedding_length,
            batch_size_);
        ggml_set_input(named("xvoice.stage2.time_hidden", time_hidden_));
        lang_ids_ = ggml_new_tensor_1d(arena_.ctx, GGML_TYPE_I32, batch_size_);
        ggml_set_input(named("xvoice.stage2.time_language_ids", lang_ids_));

        ggml_tensor * cur = linear(arena_.ctx, store, time_hidden_, "stage2.time.mlp.0");
        cur = ggml_silu(arena_.ctx, cur);
        cur = linear(arena_.ctx, store, cur, "stage2.time.mlp.2");
        ggml_tensor * lang = ggml_get_rows(arena_.ctx, w(store, "stage2.lang_emb.weight"), lang_ids_);
        cur = ggml_concat(arena_.ctx, cur, lang, 0);
        cur = linear(arena_.ctx, store, cur, "stage2.cond_fusion.0");
        cur = ggml_silu(arena_.ctx, cur);
        out_ = named("xvoice.stage2.time_embed", linear(arena_.ctx, store, cur, "stage2.cond_fusion.2"));
        ggml_set_output(out_);

        graph_ = ggml_new_graph_custom(arena_.ctx, 128, false);
        ggml_build_forward_expand(graph_, out_);
        if (!ggml_gallocr_reserve(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to reserve Stage-2 time graph");
        }
    }

    FloatTensor run(
            const std::vector<float> & time_hidden_ggml_order,
            const std::vector<int32_t> & time_language_ids) {
        check_time_inputs(spec_, time_hidden_ggml_order, time_language_ids, batch_size_);

        if (!ggml_gallocr_alloc_graph(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to allocate Stage-2 time graph");
        }
        ggml_backend_tensor_set(time_hidden_, time_hidden_ggml_order.data(), 0, ggml_nbytes(time_hidden_));
        ggml_backend_tensor_set(lang_ids_, time_language_ids.data(), 0, ggml_nbytes(lang_ids_));

        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 time graph");

        FloatTensor result;
        result.ne = {out_->ne[0], out_->ne[1], out_->ne[2], out_->ne[3]};
        result.data.resize(static_cast<size_t>(ggml_nelements(out_)));
        ggml_backend_tensor_get(out_, result.data.data(), 0, ggml_nbytes(out_));
        return result;
    }

private:
    ggml_backend_t backend_ = nullptr;
    const XVoiceSpec & spec_;
    int batch_size_ = 0;
    GraphArena arena_;
    ggml_tensor * time_hidden_ = nullptr;
    ggml_tensor * lang_ids_ = nullptr;
    ggml_tensor * out_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

class Stage2InputEmbeddingRunner {
public:
    Stage2InputEmbeddingRunner(
            ggml_backend_t backend,
            ggml_backend_buffer_type_t buft,
            const TensorStore & store,
            const XVoiceSpec & spec,
            int seq_len)
        : backend_(backend),
          spec_(spec),
          seq_len_(seq_len),
          arena_(buft, 256ull * 1024ull * 1024ull) {
        if (seq_len_ <= 0) throw std::runtime_error("Stage-2 input runner seq_len must be positive");
        x_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.audio.mel_channel_count, seq_len_);
        ggml_set_input(named("xvoice.stage2.input.x", x_));
        cond_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.audio.mel_channel_count, seq_len_);
        ggml_set_input(named("xvoice.stage2.input.cond", cond_));
        text_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.stage2.text_dim, seq_len_);
        ggml_set_input(named("xvoice.stage2.input.text_embed", text_));

        ggml_tensor * cur = ggml_concat(arena_.ctx, x_, cond_, 0);
        cur = ggml_concat(arena_.ctx, cur, text_, 0);
        cur = linear(arena_.ctx, store, cur, "stage2.inp.proj");
        ggml_tensor * pos = conv_position_embedding(
            arena_.ctx,
            store,
            cur,
            "stage2.inp.pos.0",
            "stage2.inp.pos.2",
            seq_len_,
            spec_.stage2.input_conv_pos_groups);
        out_ = named("xvoice.stage2.input_embed", ggml_add(arena_.ctx, cur, pos));
        ggml_set_output(out_);

        graph_ = ggml_new_graph_custom(arena_.ctx, 2048, false);
        ggml_build_forward_expand(graph_, out_);
        if (!ggml_gallocr_reserve(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to reserve Stage-2 input graph");
        }
    }

    FloatTensor run(
            const std::vector<float> & x_ggml_order,
            const std::vector<float> & cond_ggml_order,
            const std::vector<float> & text_embed_ggml_order) {
        check_stage2_input_arrays(spec_, x_ggml_order, cond_ggml_order, text_embed_ggml_order, seq_len_);

        if (!ggml_gallocr_alloc_graph(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to allocate Stage-2 input graph");
        }
        ggml_backend_tensor_set(x_, x_ggml_order.data(), 0, ggml_nbytes(x_));
        ggml_backend_tensor_set(cond_, cond_ggml_order.data(), 0, ggml_nbytes(cond_));
        ggml_backend_tensor_set(text_, text_embed_ggml_order.data(), 0, ggml_nbytes(text_));

        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 input graph");

        FloatTensor result;
        result.ne = {out_->ne[0], out_->ne[1], out_->ne[2], out_->ne[3]};
        result.data.resize(static_cast<size_t>(ggml_nelements(out_)));
        ggml_backend_tensor_get(out_, result.data.data(), 0, ggml_nbytes(out_));
        return result;
    }

private:
    ggml_backend_t backend_ = nullptr;
    const XVoiceSpec & spec_;
    int seq_len_ = 0;
    GraphArena arena_;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * cond_ = nullptr;
    ggml_tensor * text_ = nullptr;
    ggml_tensor * out_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

class Stage2InputEmbeddingBatchRunner {
public:
    Stage2InputEmbeddingBatchRunner(
            ggml_backend_t backend,
            ggml_backend_buffer_type_t buft,
            const TensorStore & store,
            const XVoiceSpec & spec,
            int seq_len,
            int branch_count)
        : backend_(backend),
          spec_(spec),
          seq_len_(seq_len),
          branch_count_(branch_count),
          arena_(buft, 384ull * 1024ull * 1024ull) {
        if (seq_len_ <= 0) throw std::runtime_error("Stage-2 input batch runner seq_len must be positive");
        if (branch_count_ <= 1) throw std::runtime_error("Stage-2 input batch runner branch_count must be greater than one");
        x_ = ggml_new_tensor_3d(arena_.ctx, GGML_TYPE_F32, spec_.audio.mel_channel_count, seq_len_, branch_count_);
        ggml_set_input(named("xvoice.stage2.input_batch.x", x_));
        cond_ = ggml_new_tensor_3d(arena_.ctx, GGML_TYPE_F32, spec_.audio.mel_channel_count, seq_len_, branch_count_);
        ggml_set_input(named("xvoice.stage2.input_batch.cond", cond_));
        text_ = ggml_new_tensor_3d(arena_.ctx, GGML_TYPE_F32, spec_.stage2.text_dim, seq_len_, branch_count_);
        ggml_set_input(named("xvoice.stage2.input_batch.text_embed", text_));

        ggml_tensor * cur = ggml_concat(arena_.ctx, x_, cond_, 0);
        cur = ggml_concat(arena_.ctx, cur, text_, 0);
        cur = linear_nd(arena_.ctx, store, cur, "stage2.inp.proj");
        ggml_tensor * pos = conv_position_embedding_branchwise_3d(
            arena_.ctx,
            store,
            cur,
            "stage2.inp.pos.0",
            "stage2.inp.pos.2",
            seq_len_,
            spec_.stage2.input_conv_pos_groups);
        out_ = named("xvoice.stage2.input_batch_embed", ggml_add(arena_.ctx, cur, pos));
        ggml_set_output(out_);

        graph_ = ggml_new_graph_custom(arena_.ctx, 4096, false);
        ggml_build_forward_expand(graph_, out_);
        if (!ggml_gallocr_reserve(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to reserve Stage-2 input batch graph");
        }
    }

    FloatTensor run(
            const std::vector<float> & x_ggml_order,
            const std::vector<float> & cond_ggml_order,
            const std::vector<float> & text_embed_ggml_order) {
        check_stage2_input_batch_arrays(
            spec_,
            x_ggml_order,
            cond_ggml_order,
            text_embed_ggml_order,
            seq_len_,
            branch_count_);

        if (!ggml_gallocr_alloc_graph(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to allocate Stage-2 input batch graph");
        }
        ggml_backend_tensor_set(x_, x_ggml_order.data(), 0, ggml_nbytes(x_));
        ggml_backend_tensor_set(cond_, cond_ggml_order.data(), 0, ggml_nbytes(cond_));
        ggml_backend_tensor_set(text_, text_embed_ggml_order.data(), 0, ggml_nbytes(text_));

        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 input batch graph");

        FloatTensor result;
        result.ne = {out_->ne[0], out_->ne[1], out_->ne[2], out_->ne[3]};
        result.data.resize(static_cast<size_t>(ggml_nelements(out_)));
        ggml_backend_tensor_get(out_, result.data.data(), 0, ggml_nbytes(out_));
        return result;
    }

private:
    ggml_backend_t backend_ = nullptr;
    const XVoiceSpec & spec_;
    int seq_len_ = 0;
    int branch_count_ = 0;
    GraphArena arena_;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * cond_ = nullptr;
    ggml_tensor * text_ = nullptr;
    ggml_tensor * out_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

class Stage2ConditionedDitRunner {
public:
    Stage2ConditionedDitRunner(
            ggml_backend_t backend,
            ggml_backend_buffer_type_t buft,
            const TensorStore & store,
            const XVoiceSpec & spec,
            int seq_len)
        : backend_(backend),
          spec_(spec),
          seq_len_(seq_len),
          arena_(buft, 2048ull * 1024ull * 1024ull) {
        if (seq_len_ <= 0) throw std::runtime_error("Stage-2 conditioned DiT runner seq_len must be positive");

        x_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.audio.mel_channel_count, seq_len_);
        ggml_set_input(named("xvoice.stage2.conditioned.x", x_));
        cond_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.audio.mel_channel_count, seq_len_);
        ggml_set_input(named("xvoice.stage2.conditioned.cond", cond_));
        text_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.stage2.text_dim, seq_len_);
        ggml_set_input(named("xvoice.stage2.conditioned.text_embed", text_));
        time_embed_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.stage2.hidden_size, 1);
        ggml_set_input(named("xvoice.stage2.conditioned.time_embed", time_embed_));
        pos_ = ggml_new_tensor_1d(arena_.ctx, GGML_TYPE_I32, seq_len_);
        ggml_set_input(named("xvoice.stage2.conditioned.positions", pos_));

        ggml_tensor * cur = ggml_concat(arena_.ctx, x_, cond_, 0);
        cur = ggml_concat(arena_.ctx, cur, text_, 0);
        cur = linear(arena_.ctx, store, cur, "stage2.inp.proj");
        ggml_tensor * pos_embed = conv_position_embedding(
            arena_.ctx,
            store,
            cur,
            "stage2.inp.pos.0",
            "stage2.inp.pos.2",
            seq_len_,
            spec_.stage2.input_conv_pos_groups);
        cur = ggml_add(arena_.ctx, cur, pos_embed);

        for (int index = 0; index < spec_.stage2.block_count; ++index) {
            cur = dit_block_2d(arena_.ctx, store, spec_, cur, time_embed_, index, pos_);
        }
        mel_ = named("xvoice.stage2.conditioned.mel", stage2_output_mel(arena_.ctx, store, spec_, cur, time_embed_));
        ggml_set_output(mel_);

        graph_ = ggml_new_graph_custom(arena_.ctx, 72000, false);
        ggml_build_forward_expand(graph_, mel_);
        if (!ggml_gallocr_reserve(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to reserve Stage-2 conditioned DiT graph");
        }

        pos_data_ = positions(seq_len_);
    }

    FloatTensor run(
            const std::vector<float> & x_ggml_order,
            const std::vector<float> & cond_ggml_order,
            const std::vector<float> & text_embed_ggml_order,
            const std::vector<float> & time_embed_ggml_order) {
        check_stage2_input_arrays(spec_, x_ggml_order, cond_ggml_order, text_embed_ggml_order, seq_len_);
        if (time_embed_ggml_order.size() != static_cast<size_t>(spec_.stage2.hidden_size)) {
            throw std::runtime_error("Stage-2 conditioned DiT time_embed size must match GGML ne=(hidden_size, 1)");
        }

        if (!ggml_gallocr_alloc_graph(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to allocate Stage-2 conditioned DiT graph");
        }
        ggml_backend_tensor_set(x_, x_ggml_order.data(), 0, ggml_nbytes(x_));
        ggml_backend_tensor_set(cond_, cond_ggml_order.data(), 0, ggml_nbytes(cond_));
        ggml_backend_tensor_set(text_, text_embed_ggml_order.data(), 0, ggml_nbytes(text_));
        ggml_backend_tensor_set(time_embed_, time_embed_ggml_order.data(), 0, ggml_nbytes(time_embed_));
        ggml_backend_tensor_set(pos_, pos_data_.data(), 0, ggml_nbytes(pos_));

        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 conditioned DiT graph");
        }

        FloatTensor result;
        result.ne = {mel_->ne[0], mel_->ne[1], mel_->ne[2], mel_->ne[3]};
        result.data.resize(static_cast<size_t>(ggml_nelements(mel_)));
        ggml_backend_tensor_get(mel_, result.data.data(), 0, ggml_nbytes(mel_));
        return result;
    }

private:
    ggml_backend_t backend_ = nullptr;
    const XVoiceSpec & spec_;
    int seq_len_ = 0;
    GraphArena arena_;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * cond_ = nullptr;
    ggml_tensor * text_ = nullptr;
    ggml_tensor * time_embed_ = nullptr;
    ggml_tensor * pos_ = nullptr;
    ggml_tensor * mel_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::vector<int32_t> pos_data_;
};

class Stage2CfgBatchStepRunner {
public:
    Stage2CfgBatchStepRunner(
            ggml_backend_t backend,
            ggml_backend_buffer_type_t buft,
            const TensorStore & store,
            const XVoiceSpec & spec,
            int seq_len,
            int branch_count)
        : backend_(backend),
          spec_(spec),
          seq_len_(seq_len),
          branch_count_(branch_count),
          arena_(buft, 3072ull * 1024ull * 1024ull) {
        if (seq_len_ <= 0) throw std::runtime_error("Stage-2 CFG batch step runner seq_len must be positive");
        if (branch_count_ != 2 && branch_count_ != 3) {
            throw std::runtime_error("Stage-2 CFG batch step runner supports branch_count 2 or 3");
        }

        x_ = ggml_new_tensor_3d(arena_.ctx, GGML_TYPE_F32, spec_.audio.mel_channel_count, seq_len_, branch_count_);
        ggml_set_input(named("xvoice.stage2.cfg_batch_step.x", x_));
        cond_ = ggml_new_tensor_3d(arena_.ctx, GGML_TYPE_F32, spec_.audio.mel_channel_count, seq_len_, branch_count_);
        ggml_set_input(named("xvoice.stage2.cfg_batch_step.cond", cond_));
        text_ = ggml_new_tensor_3d(arena_.ctx, GGML_TYPE_F32, spec_.stage2.text_dim, seq_len_, branch_count_);
        ggml_set_input(named("xvoice.stage2.cfg_batch_step.text_embed", text_));
        time_embed_ = ggml_new_tensor_2d(arena_.ctx, GGML_TYPE_F32, spec_.stage2.hidden_size, 1);
        ggml_set_input(named("xvoice.stage2.cfg_batch_step.time_embed", time_embed_));
        cfg_ = ggml_new_tensor_1d(arena_.ctx, GGML_TYPE_F32, 1);
        ggml_set_input(named("xvoice.stage2.cfg_batch_step.cfg", cfg_));
        cfg2_ = ggml_new_tensor_1d(arena_.ctx, GGML_TYPE_F32, 1);
        ggml_set_input(named("xvoice.stage2.cfg_batch_step.cfg2", cfg2_));
        dt_ = ggml_new_tensor_1d(arena_.ctx, GGML_TYPE_F32, 1);
        ggml_set_input(named("xvoice.stage2.cfg_batch_step.dt", dt_));
        pos_ = ggml_new_tensor_1d(arena_.ctx, GGML_TYPE_I32, seq_len_);
        ggml_set_input(named("xvoice.stage2.cfg_batch_step.positions", pos_));

        ggml_tensor * pred_branches = stage2_conditioned_batch_mel(
            arena_.ctx,
            store,
            spec_,
            x_,
            cond_,
            text_,
            time_embed_,
            pos_,
            seq_len_);
        ggml_tensor * current = slice_3d_branch_to_2d(arena_.ctx, x_, 0);
        if (branch_count_ == 2) {
            ggml_tensor * full = slice_3d_branch_to_2d(arena_.ctx, pred_branches, 0);
            ggml_tensor * null = slice_3d_branch_to_2d(arena_.ctx, pred_branches, 1);
            ggml_tensor * pred = ggml_add(arena_.ctx, full, ggml_mul(arena_.ctx, ggml_sub(arena_.ctx, full, null), cfg_));
            next_ = named("xvoice.stage2.cfg_batch_step.next", ggml_add(arena_.ctx, current, ggml_mul(arena_.ctx, pred, dt_)));
        } else {
            ggml_tensor * full = slice_3d_branch_to_2d(arena_.ctx, pred_branches, 0);
            ggml_tensor * text = slice_3d_branch_to_2d(arena_.ctx, pred_branches, 1);
            ggml_tensor * null = slice_3d_branch_to_2d(arena_.ctx, pred_branches, 2);
            ggml_tensor * delta_content = ggml_sub(arena_.ctx, text, null);
            ggml_tensor * delta_audio = ggml_sub(arena_.ctx, full, text);
            ggml_tensor * pred = ggml_add(
                arena_.ctx,
                null,
                ggml_add(
                    arena_.ctx,
                    ggml_mul(arena_.ctx, delta_content, cfg2_),
                    ggml_mul(arena_.ctx, delta_audio, cfg_)));
            next_ = named("xvoice.stage2.cfg_batch_step.next", ggml_add(arena_.ctx, current, ggml_mul(arena_.ctx, pred, dt_)));
        }
        ggml_set_output(next_);

        graph_ = ggml_new_graph_custom(arena_.ctx, 72000, false);
        ggml_build_forward_expand(graph_, next_);
        if (!ggml_gallocr_reserve(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to reserve Stage-2 CFG batch step graph");
        }

        pos_data_ = positions(seq_len_);
    }

    FloatTensor run(
            const std::vector<float> & x_ggml_order,
            const std::vector<float> & cond_ggml_order,
            const std::vector<float> & text_embed_ggml_order,
            const std::vector<float> & time_embed_ggml_order,
            float cfg,
            float cfg2,
            float dt) {
        check_stage2_conditioned_batch_arrays(
            spec_,
            x_ggml_order,
            cond_ggml_order,
            text_embed_ggml_order,
            time_embed_ggml_order,
            seq_len_,
            branch_count_);

        if (!ggml_gallocr_alloc_graph(arena_.allocr, graph_)) {
            throw std::runtime_error("failed to allocate Stage-2 CFG batch step graph");
        }
        ggml_backend_tensor_set(x_, x_ggml_order.data(), 0, ggml_nbytes(x_));
        ggml_backend_tensor_set(cond_, cond_ggml_order.data(), 0, ggml_nbytes(cond_));
        ggml_backend_tensor_set(text_, text_embed_ggml_order.data(), 0, ggml_nbytes(text_));
        ggml_backend_tensor_set(time_embed_, time_embed_ggml_order.data(), 0, ggml_nbytes(time_embed_));
        ggml_backend_tensor_set(cfg_, &cfg, 0, ggml_nbytes(cfg_));
        if (branch_count_ == 3) ggml_backend_tensor_set(cfg2_, &cfg2, 0, ggml_nbytes(cfg2_));
        ggml_backend_tensor_set(dt_, &dt, 0, ggml_nbytes(dt_));
        ggml_backend_tensor_set(pos_, pos_data_.data(), 0, ggml_nbytes(pos_));

        const ggml_status status = ggml_backend_graph_compute(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 CFG batch step graph");
        }

        FloatTensor result;
        result.ne = {next_->ne[0], next_->ne[1], next_->ne[2], next_->ne[3]};
        result.data.resize(static_cast<size_t>(ggml_nelements(next_)));
        ggml_backend_tensor_get(next_, result.data.data(), 0, ggml_nbytes(next_));
        return result;
    }

private:
    ggml_backend_t backend_ = nullptr;
    const XVoiceSpec & spec_;
    int seq_len_ = 0;
    int branch_count_ = 0;
    GraphArena arena_;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * cond_ = nullptr;
    ggml_tensor * text_ = nullptr;
    ggml_tensor * time_embed_ = nullptr;
    ggml_tensor * cfg_ = nullptr;
    ggml_tensor * cfg2_ = nullptr;
    ggml_tensor * dt_ = nullptr;
    ggml_tensor * pos_ = nullptr;
    ggml_tensor * next_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::vector<int32_t> pos_data_;
};

} // namespace

FloatTensor run_stage2_time_embedding_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & time_hidden_ggml_order,
        const std::vector<int32_t> & time_language_ids,
        int batch_size) {
    check_time_inputs(spec, time_hidden_ggml_order, time_language_ids, batch_size);
    Stage2TimeEmbeddingRunner runner(backend, buft, store, spec, batch_size);
    return runner.run(time_hidden_ggml_order, time_language_ids);
}

FloatTensor run_stage2_dit_block_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_index) {
    check_stage2_block_arrays(spec, x_ggml_order, time_embed_ggml_order, seq_len, block_index);

    GraphArena arena(buft, 1024ull * 1024ull * 1024ull);
    ggml_tensor * x = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, spec.stage2.hidden_size, seq_len);
    ggml_set_input(named("xvoice.stage2.block.x", x));
    ggml_tensor * time_embed = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, spec.stage2.hidden_size, 1);
    ggml_set_input(named("xvoice.stage2.block.time_embed", time_embed));
    ggml_tensor * pos = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_input(named("xvoice.stage2.block.positions", pos));

    ggml_tensor * out = named("xvoice.stage2.block.out", dit_block_2d(arena.ctx, store, spec, x, time_embed, block_index, pos));
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(arena.ctx, 8192, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_reserve(arena.allocr, graph);
    ggml_gallocr_alloc_graph(arena.allocr, graph);

    const std::vector<int32_t> pos_data = positions(seq_len);
    ggml_backend_tensor_set(x, x_ggml_order.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(time_embed, time_embed_ggml_order.data(), 0, ggml_nbytes(time_embed));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, ggml_nbytes(pos));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 DiT block graph");

    FloatTensor result;
    result.ne = {out->ne[0], out->ne[1], out->ne[2], out->ne[3]};
    result.data.resize(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data.data(), 0, ggml_nbytes(out));
    return result;
}

FloatTensor run_stage2_dit_stack_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_count) {
    check_stage2_stack_arrays(spec, x_ggml_order, time_embed_ggml_order, seq_len, block_count);

    GraphArena arena(buft, 1536ull * 1024ull * 1024ull);
    ggml_tensor * x = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, spec.stage2.hidden_size, seq_len);
    ggml_set_input(named("xvoice.stage2.stack.x", x));
    ggml_tensor * time_embed = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, spec.stage2.hidden_size, 1);
    ggml_set_input(named("xvoice.stage2.stack.time_embed", time_embed));
    ggml_tensor * pos = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_input(named("xvoice.stage2.stack.positions", pos));

    ggml_tensor * out = x;
    for (int index = 0; index < block_count; ++index) {
        out = dit_block_2d(arena.ctx, store, spec, out, time_embed, index, pos);
    }
    out = named("xvoice.stage2.stack.out", out);
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(arena.ctx, 65536, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_reserve(arena.allocr, graph);
    ggml_gallocr_alloc_graph(arena.allocr, graph);

    const std::vector<int32_t> pos_data = positions(seq_len);
    ggml_backend_tensor_set(x, x_ggml_order.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(time_embed, time_embed_ggml_order.data(), 0, ggml_nbytes(time_embed));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, ggml_nbytes(pos));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 DiT stack graph");

    FloatTensor result;
    result.ne = {out->ne[0], out->ne[1], out->ne[2], out->ne[3]};
    result.data.resize(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data.data(), 0, ggml_nbytes(out));
    return result;
}

FloatTensor run_stage2_output_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len) {
    check_stage2_block_arrays(spec, x_ggml_order, time_embed_ggml_order, seq_len, 0);

    GraphArena arena(buft, 256ull * 1024ull * 1024ull);
    ggml_tensor * x = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, spec.stage2.hidden_size, seq_len);
    ggml_set_input(named("xvoice.stage2.output.x", x));
    ggml_tensor * time_embed = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, spec.stage2.hidden_size, 1);
    ggml_set_input(named("xvoice.stage2.output.time_embed", time_embed));

    ggml_tensor * mel = named("xvoice.stage2.output.mel", stage2_output_mel(arena.ctx, store, spec, x, time_embed));
    ggml_set_output(mel);

    ggml_cgraph * graph = ggml_new_graph_custom(arena.ctx, 512, false);
    ggml_build_forward_expand(graph, mel);
    ggml_gallocr_reserve(arena.allocr, graph);
    ggml_gallocr_alloc_graph(arena.allocr, graph);

    ggml_backend_tensor_set(x, x_ggml_order.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(time_embed, time_embed_ggml_order.data(), 0, ggml_nbytes(time_embed));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 output graph");

    FloatTensor result;
    result.ne = {mel->ne[0], mel->ne[1], mel->ne[2], mel->ne[3]};
    result.data.resize(static_cast<size_t>(ggml_nelements(mel)));
    ggml_backend_tensor_get(mel, result.data.data(), 0, ggml_nbytes(mel));
    return result;
}

FloatTensor run_stage2_dit_forward_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_count) {
    check_stage2_stack_arrays(spec, x_ggml_order, time_embed_ggml_order, seq_len, block_count);
    Stage2DitForwardRunner runner(backend, buft, store, spec, seq_len, block_count);
    return runner.run(x_ggml_order, time_embed_ggml_order);
}

FloatTensor run_stage2_text_embedding_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<int32_t> & text_ids_plus_one,
        const std::vector<int32_t> & language_ids,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        const std::vector<float> & text_keep_mask_ggml_order,
        const std::vector<float> & text_pos_embed_ggml_order,
        int seq_len) {
    check_stage2_text_arrays(
        spec,
        text_ids_plus_one,
        language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        seq_len);

    GraphArena arena(buft, 512ull * 1024ull * 1024ull);
    ggml_tensor * text_ids = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_input(named("xvoice.stage2.text_ids_plus_one", text_ids));
    ggml_tensor * lang_ids = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, seq_len);
    ggml_set_input(named("xvoice.stage2.text_language_ids", lang_ids));
    ggml_tensor * no_lang_mask = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, 1, seq_len);
    ggml_set_input(named("xvoice.stage2.text_no_lang_fusion_mask", no_lang_mask));
    ggml_tensor * keep_mask_in = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, 1, seq_len);
    ggml_set_input(named("xvoice.stage2.text_keep_mask", keep_mask_in));
    ggml_tensor * text_pos_embed = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, spec.stage2.text_dim, seq_len);
    ggml_set_input(named("xvoice.stage2.text_pos_embed", text_pos_embed));

    ggml_tensor * text = ggml_get_rows(arena.ctx, w(store, "stage2.txt.emb.weight"), text_ids);
    ggml_tensor * text_lookup = text;
    ggml_tensor * no_lang_full = ggml_repeat(arena.ctx, no_lang_mask, text);
    ggml_tensor * keep_mask = ggml_repeat(arena.ctx, keep_mask_in, text);

    ggml_tensor * lang = ggml_get_rows(arena.ctx, w(store, "stage2.txt.lang_emb.weight"), lang_ids);
    ggml_tensor * adaln = linear(arena.ctx, store, lang, "stage2.txt.lang_ada");
    ggml_tensor * scale = slice_2d_rows(arena.ctx, adaln, 0, spec.stage2.text_dim);
    ggml_tensor * shift = slice_2d_rows(arena.ctx, adaln, spec.stage2.text_dim, spec.stage2.text_dim);
    ggml_tensor * fused_text = ggml_add(
        arena.ctx,
        ggml_mul(arena.ctx, text, ggml_scale_bias(arena.ctx, scale, 1.0f, 1.0f)),
        shift);
    text = ggml_add(
        arena.ctx,
        ggml_mul(arena.ctx, text_lookup, no_lang_full),
        ggml_mul(arena.ctx, fused_text, ggml_scale_bias(arena.ctx, no_lang_full, -1.0f, 1.0f)));

    text = ggml_add(arena.ctx, text, text_pos_embed);
    text = ggml_mul(arena.ctx, text, keep_mask);
    for (int index = 0; index < spec.stage2.text_conv_layer_count; ++index) {
        text = convnext_v2_text_block_2d(arena.ctx, store, text, "stage2.txt.blk." + std::to_string(index), seq_len);
        text = ggml_mul(arena.ctx, text, keep_mask);
    }
    text = named("xvoice.stage2.text_embed", text);
    ggml_set_output(text);

    ggml_cgraph * graph = ggml_new_graph_custom(arena.ctx, 4096, false);
    ggml_build_forward_expand(graph, text);
    ggml_gallocr_reserve(arena.allocr, graph);
    ggml_gallocr_alloc_graph(arena.allocr, graph);

    ggml_backend_tensor_set(text_ids, text_ids_plus_one.data(), 0, ggml_nbytes(text_ids));
    ggml_backend_tensor_set(lang_ids, language_ids.data(), 0, ggml_nbytes(lang_ids));
    ggml_backend_tensor_set(no_lang_mask, text_no_lang_fusion_mask_ggml_order.data(), 0, ggml_nbytes(no_lang_mask));
    ggml_backend_tensor_set(keep_mask_in, text_keep_mask_ggml_order.data(), 0, ggml_nbytes(keep_mask_in));
    ggml_backend_tensor_set(text_pos_embed, text_pos_embed_ggml_order.data(), 0, ggml_nbytes(text_pos_embed));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Stage-2 text graph");

    FloatTensor result;
    result.ne = {text->ne[0], text->ne[1], text->ne[2], text->ne[3]};
    result.data.resize(static_cast<size_t>(ggml_nelements(text)));
    ggml_backend_tensor_get(text, result.data.data(), 0, ggml_nbytes(text));
    return result;
}

FloatTensor run_stage2_input_embedding_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        int seq_len) {
    check_stage2_input_arrays(spec, x_ggml_order, cond_ggml_order, text_embed_ggml_order, seq_len);
    Stage2InputEmbeddingRunner runner(backend, buft, store, spec, seq_len);
    return runner.run(x_ggml_order, cond_ggml_order, text_embed_ggml_order);
}

FloatTensor run_stage2_input_embedding_batch_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        int seq_len,
        int branch_count) {
    Stage2InputEmbeddingBatchRunner runner(backend, buft, store, spec, seq_len, branch_count);
    return runner.run(x_ggml_order, cond_ggml_order, text_embed_ggml_order);
}

Stage2NoCfgSamplerResult run_stage2_no_cfg_sampler_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<int32_t> & text_ids_plus_one,
        const std::vector<int32_t> & text_language_ids,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        const std::vector<float> & text_keep_mask_ggml_order,
        const std::vector<float> & text_pos_embed_ggml_order,
        const std::vector<int32_t> & time_language_ids,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & cond_mask_ggml_order,
        const std::vector<float> & fixed_noise_ggml_order,
        const std::vector<float> & sampler_timesteps,
        int seq_len,
        int step_count,
        const ProgressCallback & progress_callback,
        bool profile_stage2_branches) {
    check_stage2_no_cfg_sampler_arrays(
        spec,
        text_ids_plus_one,
        text_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        time_language_ids,
        cond_ggml_order,
        cond_mask_ggml_order,
        fixed_noise_ggml_order,
        sampler_timesteps,
        seq_len,
        step_count);

    FloatTensor text_embed = run_stage2_text_embedding_graph(
        backend,
        buft,
        store,
        spec,
        text_ids_plus_one,
        text_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        seq_len);

    Stage2TimeEmbeddingRunner time_runner(backend, buft, store, spec, 1);
    Stage2ConditionedDitRunner conditioned_runner(backend, buft, store, spec, seq_len);
    std::vector<float> current = fixed_noise_ggml_order;
    emit_progress(progress_callback, "stage2 sampler", 0, step_count, "no_cfg");
    for (int index = 0; index < step_count; ++index) {
        const std::vector<float> time_hidden = sinus_position_embedding(
            sampler_timesteps[static_cast<size_t>(index)],
            spec.stage2.time_freq_embedding_length);
        FloatTensor time_embed = time_runner.run(time_hidden, time_language_ids);
        if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/no_cfg", 0, 1, "conditioned_dit");
        FloatTensor pred = conditioned_runner.run(current, cond_ggml_order, text_embed.data, time_embed.data);
        if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/no_cfg", 1, 1, "conditioned_dit");
        const float dt = sampler_timesteps[static_cast<size_t>(index + 1)] - sampler_timesteps[static_cast<size_t>(index)];
        for (size_t i = 0; i < current.size(); ++i) {
            current[i] = current[i] + dt * pred.data[i];
        }
        emit_progress(progress_callback, "stage2 sampler", index + 1, step_count, "no_cfg");
    }

    Stage2NoCfgSamplerResult result;
    result.sampled.ne = {spec.audio.mel_channel_count, seq_len, 1, 1};
    result.sampled.data = current;
    result.mel.ne = result.sampled.ne;
    result.mel.data = apply_conditioning_region(spec, current, cond_ggml_order, cond_mask_ggml_order, seq_len);
    return result;
}

Stage2NonLayeredCfgSamplerResult run_stage2_nonlayered_cfg_sampler_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<int32_t> & text_ids_plus_one,
        const std::vector<int32_t> & text_language_ids,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        const std::vector<float> & text_keep_mask_ggml_order,
        const std::vector<float> & text_pos_embed_ggml_order,
        const std::vector<int32_t> & time_language_ids,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & cond_mask_ggml_order,
        const std::vector<float> & fixed_noise_ggml_order,
        const std::vector<float> & sampler_timesteps,
        int seq_len,
        int step_count,
        const ProgressCallback & progress_callback,
        bool profile_stage2_branches,
        bool batch_dit_forward) {
    check_stage2_no_cfg_sampler_arrays(
        spec,
        text_ids_plus_one,
        text_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        time_language_ids,
        cond_ggml_order,
        cond_mask_ggml_order,
        fixed_noise_ggml_order,
        sampler_timesteps,
        seq_len,
        step_count);

    FloatTensor full_text_embed = run_stage2_text_embedding_graph(
        backend,
        buft,
        store,
        spec,
        text_ids_plus_one,
        text_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        seq_len);

    const std::vector<int32_t> null_text_ids(static_cast<size_t>(seq_len), 0);
    const std::vector<int32_t> null_language_ids = cfg_null_text_language_ids(
        spec,
        text_no_lang_fusion_mask_ggml_order,
        seq_len);
    FloatTensor null_text_embed = run_stage2_text_embedding_graph(
        backend,
        buft,
        store,
        spec,
        null_text_ids,
        null_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        seq_len);

    const std::vector<float> zero_cond = zero_like(cond_ggml_order);
    Stage2TimeEmbeddingRunner time_runner(backend, buft, store, spec, 1);
    std::unique_ptr<Stage2CfgBatchStepRunner> cfg_batch_step_runner;
    std::unique_ptr<Stage2ConditionedDitRunner> conditioned_runner;
    if (batch_dit_forward) {
        cfg_batch_step_runner = std::make_unique<Stage2CfgBatchStepRunner>(backend, buft, store, spec, seq_len, 2);
    } else {
        conditioned_runner = std::make_unique<Stage2ConditionedDitRunner>(backend, buft, store, spec, seq_len);
    }
    std::vector<float> current = fixed_noise_ggml_order;
    const size_t mel_branch_size = static_cast<size_t>(spec.audio.mel_channel_count) * static_cast<size_t>(seq_len);
    const size_t text_branch_size = static_cast<size_t>(spec.stage2.text_dim) * static_cast<size_t>(seq_len);
    emit_progress(progress_callback, "stage2 sampler", 0, step_count, "cfg_nonlayered");
    for (int index = 0; index < step_count; ++index) {
        const float time_value = sampler_timesteps[static_cast<size_t>(index)];
        const std::vector<float> time_hidden = sinus_position_embedding(
            time_value,
            spec.stage2.time_freq_embedding_length);
        FloatTensor time_embed = time_runner.run(time_hidden, time_language_ids);

        if (batch_dit_forward) {
            const std::vector<float> x_branches = pack_ggml_order_branches({&current, &current}, mel_branch_size);
            const std::vector<float> cond_branches = pack_ggml_order_branches({&cond_ggml_order, &zero_cond}, mel_branch_size);
            const std::vector<float> text_branches = pack_ggml_order_branches({&full_text_embed.data, &null_text_embed.data}, text_branch_size);
            const float cfg = cfg_schedule_values(
                time_value,
                spec.sampler.cfg_strength,
                0.0f,
                spec.sampler.cfg_schedule,
                spec.sampler.cfg_decay_time).first;
            const float dt = sampler_timesteps[static_cast<size_t>(index + 1)] - sampler_timesteps[static_cast<size_t>(index)];
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/full|null", 0, 1, "cfg_step_batch2");
            FloatTensor next = cfg_batch_step_runner->run(x_branches, cond_branches, text_branches, time_embed.data, cfg, 0.0f, dt);
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/full|null", 1, 1, "cfg_step_batch2");
            current = std::move(next.data);
        } else {
            std::vector<float> full_pred;
            std::vector<float> null_pred;
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/full", 0, 1, "conditioned_dit");
            FloatTensor full = conditioned_runner->run(current, cond_ggml_order, full_text_embed.data, time_embed.data);
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/full", 1, 1, "conditioned_dit");
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/null", 0, 1, "conditioned_dit");
            FloatTensor null = conditioned_runner->run(current, zero_cond, null_text_embed.data, time_embed.data);
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/null", 1, 1, "conditioned_dit");
            full_pred = std::move(full.data);
            null_pred = std::move(null.data);

            const float cfg = cfg_schedule_values(
                time_value,
                spec.sampler.cfg_strength,
                0.0f,
                spec.sampler.cfg_schedule,
                spec.sampler.cfg_decay_time).first;
            const float dt = sampler_timesteps[static_cast<size_t>(index + 1)] - sampler_timesteps[static_cast<size_t>(index)];
            for (size_t i = 0; i < current.size(); ++i) {
                const float pred = full_pred[i] + (full_pred[i] - null_pred[i]) * cfg;
                current[i] = current[i] + dt * pred;
            }
        }
        emit_progress(progress_callback, "stage2 sampler", index + 1, step_count, "cfg_nonlayered");
    }

    Stage2NonLayeredCfgSamplerResult result;
    result.sampled.ne = {spec.audio.mel_channel_count, seq_len, 1, 1};
    result.sampled.data = current;
    result.mel.ne = result.sampled.ne;
    result.mel.data = apply_conditioning_region(spec, current, cond_ggml_order, cond_mask_ggml_order, seq_len);
    return result;
}

Stage2LayeredCfgSamplerResult run_stage2_layered_cfg_sampler_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<int32_t> & text_ids_plus_one,
        const std::vector<int32_t> & text_language_ids,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        const std::vector<float> & text_keep_mask_ggml_order,
        const std::vector<float> & text_pos_embed_ggml_order,
        const std::vector<int32_t> & time_language_ids,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & cond_mask_ggml_order,
        const std::vector<float> & fixed_noise_ggml_order,
        const std::vector<float> & sampler_timesteps,
        int seq_len,
        int step_count,
        const ProgressCallback & progress_callback,
        bool profile_stage2_branches,
        bool batch_dit_forward) {
    check_stage2_no_cfg_sampler_arrays(
        spec,
        text_ids_plus_one,
        text_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        time_language_ids,
        cond_ggml_order,
        cond_mask_ggml_order,
        fixed_noise_ggml_order,
        sampler_timesteps,
        seq_len,
        step_count);

    FloatTensor full_text_embed = run_stage2_text_embedding_graph(
        backend,
        buft,
        store,
        spec,
        text_ids_plus_one,
        text_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        seq_len);

    const std::vector<int32_t> null_text_ids(static_cast<size_t>(seq_len), 0);
    const std::vector<int32_t> null_language_ids = cfg_null_text_language_ids(
        spec,
        text_no_lang_fusion_mask_ggml_order,
        seq_len);
    FloatTensor null_text_embed = run_stage2_text_embedding_graph(
        backend,
        buft,
        store,
        spec,
        null_text_ids,
        null_language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        seq_len);

    const std::vector<float> zero_cond = zero_like(cond_ggml_order);
    Stage2TimeEmbeddingRunner time_runner(backend, buft, store, spec, 1);
    std::unique_ptr<Stage2CfgBatchStepRunner> cfg_batch_step_runner;
    std::unique_ptr<Stage2ConditionedDitRunner> conditioned_runner;
    if (batch_dit_forward) {
        cfg_batch_step_runner = std::make_unique<Stage2CfgBatchStepRunner>(backend, buft, store, spec, seq_len, 3);
    } else {
        conditioned_runner = std::make_unique<Stage2ConditionedDitRunner>(backend, buft, store, spec, seq_len);
    }
    std::vector<float> current = fixed_noise_ggml_order;
    const size_t mel_branch_size = static_cast<size_t>(spec.audio.mel_channel_count) * static_cast<size_t>(seq_len);
    const size_t text_branch_size = static_cast<size_t>(spec.stage2.text_dim) * static_cast<size_t>(seq_len);
    emit_progress(progress_callback, "stage2 sampler", 0, step_count, "cfg_layered");
    for (int index = 0; index < step_count; ++index) {
        const float time_value = sampler_timesteps[static_cast<size_t>(index)];
        const std::vector<float> time_hidden = sinus_position_embedding(
            time_value,
            spec.stage2.time_freq_embedding_length);
        FloatTensor time_embed = time_runner.run(time_hidden, time_language_ids);

        if (batch_dit_forward) {
            const std::vector<float> x_branches = pack_ggml_order_branches({&current, &current, &current}, mel_branch_size);
            const std::vector<float> cond_branches = pack_ggml_order_branches({&cond_ggml_order, &zero_cond, &zero_cond}, mel_branch_size);
            const std::vector<float> text_branches = pack_ggml_order_branches({&full_text_embed.data, &full_text_embed.data, &null_text_embed.data}, text_branch_size);
            const auto cfg_values = cfg_schedule_values(
                time_value,
                spec.sampler.cfg_strength,
                spec.sampler.cfg_strength2,
                spec.sampler.cfg_schedule,
                spec.sampler.cfg_decay_time);
            const float warmup_gate = std::min(time_value / 0.01f, 1.0f);
            const float dt = sampler_timesteps[static_cast<size_t>(index + 1)] - sampler_timesteps[static_cast<size_t>(index)];
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/full|text|null", 0, 1, "cfg_step_batch3");
            FloatTensor next = cfg_batch_step_runner->run(
                x_branches,
                cond_branches,
                text_branches,
                time_embed.data,
                1.0f + cfg_values.first,
                1.0f + cfg_values.second * warmup_gate,
                dt);
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/full|text|null", 1, 1, "cfg_step_batch3");
            current = std::move(next.data);
        } else {
            std::vector<float> full_pred;
            std::vector<float> text_pred;
            std::vector<float> null_pred;
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/full", 0, 1, "conditioned_dit");
            FloatTensor full = conditioned_runner->run(current, cond_ggml_order, full_text_embed.data, time_embed.data);
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/full", 1, 1, "conditioned_dit");

            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/text", 0, 1, "conditioned_dit");
            FloatTensor text = conditioned_runner->run(current, zero_cond, full_text_embed.data, time_embed.data);
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/text", 1, 1, "conditioned_dit");

            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/null", 0, 1, "conditioned_dit");
            FloatTensor null = conditioned_runner->run(current, zero_cond, null_text_embed.data, time_embed.data);
            if (profile_stage2_branches) emit_progress(progress_callback, "stage2 branch/null", 1, 1, "conditioned_dit");

            full_pred = std::move(full.data);
            text_pred = std::move(text.data);
            null_pred = std::move(null.data);

            const auto cfg_values = cfg_schedule_values(
                time_value,
                spec.sampler.cfg_strength,
                spec.sampler.cfg_strength2,
                spec.sampler.cfg_schedule,
                spec.sampler.cfg_decay_time);
            const float cfg = cfg_values.first;
            const float cfg2 = cfg_values.second;
            const float warmup_gate = std::min(time_value / 0.01f, 1.0f);
            const float dt = sampler_timesteps[static_cast<size_t>(index + 1)] - sampler_timesteps[static_cast<size_t>(index)];
            for (size_t i = 0; i < current.size(); ++i) {
                const float delta_audio = full_pred[i] - text_pred[i];
                const float delta_content = text_pred[i] - null_pred[i];
                const float pred = null_pred[i] + (1.0f + cfg2 * warmup_gate) * delta_content + (1.0f + cfg) * delta_audio;
                current[i] = current[i] + dt * pred;
            }
        }
        emit_progress(progress_callback, "stage2 sampler", index + 1, step_count, "cfg_layered");
    }

    Stage2LayeredCfgSamplerResult result;
    result.sampled.ne = {spec.audio.mel_channel_count, seq_len, 1, 1};
    result.sampled.data = current;
    result.mel.ne = result.sampled.ne;
    result.mel.data = apply_conditioning_region(spec, current, cond_ggml_order, cond_mask_ggml_order, seq_len);
    return result;
}

} // namespace x_voice
