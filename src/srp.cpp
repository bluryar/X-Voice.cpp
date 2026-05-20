#include "x_voice_internal.h"

#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace x_voice {
namespace {

struct GraphArena {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocr = nullptr;

    explicit GraphArena(ggml_backend_buffer_type_t buft, size_t mem_size = 256ull * 1024ull * 1024ull) {
        ggml_init_params params{};
        params.mem_size = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("failed to initialize SRP GGML context");
        allocr = ggml_gallocr_new(buft);
        if (!allocr) throw std::runtime_error("failed to create SRP GGML graph allocator");
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

ggml_tensor * channels_2d(ggml_context * ctx, ggml_tensor * values) {
    return ggml_reshape_2d(ctx, values, values->ne[0], 1);
}

ggml_tensor * linear(ggml_context * ctx, const TensorStore & store, ggml_tensor * x, const std::string & prefix) {
    ggml_tensor * out = ggml_mul_mat(ctx, w(store, prefix + ".weight"), x);
    return ggml_add(ctx, out, bias_2d(ctx, w(store, prefix + ".bias")));
}

ggml_tensor * layer_norm(ggml_context * ctx, const TensorStore & store, ggml_tensor * x, const std::string & prefix) {
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
        const XVoiceSpec & spec,
        int seq_len) {
    const int64_t dim = x->ne[0];
    ggml_tensor * cur = ggml_cont(ctx, ggml_transpose(ctx, x));
    cur = ggml_reshape_3d(ctx, cur, seq_len, dim, 1);

    cur = conv_1d_groups_f32(ctx, store, "srp.conv.0", cur, spec.srp.conv_pos_groups, 1, 15, 1);
    ggml_tensor * cur_2d = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, cur, seq_len, dim)));
    cur_2d = mish(ctx, cur_2d);

    cur = ggml_cont(ctx, ggml_transpose(ctx, cur_2d));
    cur = ggml_reshape_3d(ctx, cur, seq_len, dim, 1);
    cur = conv_1d_groups_f32(ctx, store, "srp.conv.2", cur, spec.srp.conv_pos_groups, 1, 15, 1);
    cur_2d = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, cur, seq_len, dim)));
    return mish(ctx, cur_2d);
}

ggml_tensor * srp_input_embedding(
        ggml_context * ctx,
        const TensorStore & store,
        const XVoiceSpec & spec,
        ggml_tensor * mel,
        int seq_len) {
    ggml_tensor * cur = linear(ctx, store, mel, "srp.mel_proj");
    cur = conv_position_embedding(ctx, store, cur, spec, seq_len);
    return named("xvoice.srp.input_embed", cur);
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

ggml_tensor * srp_block(
        ggml_context * ctx,
        const TensorStore & store,
        const XVoiceSpec & spec,
        ggml_tensor * x,
        int index,
        ggml_tensor * positions) {
    const std::string prefix = "srp.blk." + std::to_string(index);
    ggml_tensor * attn_in = layer_norm(ctx, store, x, prefix + ".ln1");
    ggml_tensor * attn = self_attention_2d(
        ctx,
        store,
        attn_in,
        prefix + ".attn",
        spec.srp.head_count,
        spec.srp.head_dim,
        positions);
    x = ggml_add(ctx, x, attn);
    ggml_tensor * ff_in = layer_norm(ctx, store, x, prefix + ".ln2");
    return ggml_add(ctx, x, feed_forward_tanh_gelu(ctx, store, ff_in, prefix + ".ff"));
}

ggml_tensor * srp_pool(ggml_context * ctx, const TensorStore & store, ggml_tensor * x) {
    const int64_t seq_len = x->ne[1];
    ggml_tensor * weights_logits = linear(ctx, store, ggml_tanh(ctx, linear(ctx, store, x, "srp.pool.0")), "srp.pool.2");
    ggml_tensor * weights_seq = ggml_cont_2d(ctx, ggml_transpose(ctx, weights_logits), seq_len, 1);
    ggml_tensor * weights_prob = ggml_soft_max(ctx, weights_seq);
    ggml_tensor * x_time_major = ggml_cont_2d(ctx, ggml_transpose(ctx, x), seq_len, x->ne[0]);
    return ggml_mul_mat(ctx, x_time_major, weights_prob);
}

ggml_tensor * srp_classifier(ggml_context * ctx, const TensorStore & store, ggml_tensor * pooled) {
    ggml_tensor * cur = layer_norm(ctx, store, pooled, "srp.cls.0");
    cur = ggml_gelu_erf(ctx, linear(ctx, store, cur, "srp.cls.1"));
    return linear(ctx, store, cur, "srp.cls.3");
}

std::vector<int32_t> positions(int seq_len) {
    std::vector<int32_t> out(static_cast<size_t>(seq_len));
    for (int i = 0; i < seq_len; ++i) out[static_cast<size_t>(i)] = i;
    return out;
}

void check_mel_size(const XVoiceSpec & spec, const std::vector<float> & mel_ggml_order, int seq_len) {
    if (seq_len <= 0) throw std::runtime_error("SRP seq_len must be positive");
    const size_t expected = static_cast<size_t>(spec.audio.mel_channel_count) * static_cast<size_t>(seq_len);
    if (mel_ggml_order.size() != expected) {
        std::ostringstream msg;
        msg << "SRP mel input size mismatch: got " << mel_ggml_order.size()
            << ", expected " << expected << " for GGML ne=("
            << spec.audio.mel_channel_count << ", " << seq_len << ")";
        throw std::runtime_error(msg.str());
    }
}

FloatTensor compute_srp_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & mel_ggml_order,
        int seq_len,
        bool full_logits) {
    check_mel_size(spec, mel_ggml_order, seq_len);
    GraphArena arena(buft);
    ggml_tensor * mel = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, spec.audio.mel_channel_count, seq_len);
    ggml_set_input(named(full_logits ? "xvoice.srp.logits.mel" : "xvoice.srp.input.mel", mel));

    ggml_tensor * out = srp_input_embedding(arena.ctx, store, spec, mel, seq_len);
    ggml_tensor * pos = nullptr;
    std::vector<int32_t> pos_data;
    if (full_logits) {
        pos = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, seq_len);
        ggml_set_input(named("xvoice.srp.logits.positions", pos));
        pos_data = positions(seq_len);
        for (int i = 0; i < spec.srp.block_count; ++i) out = srp_block(arena.ctx, store, spec, out, i, pos);
        out = named("xvoice.srp.logits", srp_classifier(arena.ctx, store, srp_pool(arena.ctx, store, out)));
    }
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(arena.ctx, full_logits ? 16384 : 2048, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_reserve(arena.allocr, graph);
    ggml_gallocr_alloc_graph(arena.allocr, graph);

    ggml_backend_tensor_set(mel, mel_ggml_order.data(), 0, ggml_nbytes(mel));
    if (pos) ggml_backend_tensor_set(pos, pos_data.data(), 0, ggml_nbytes(pos));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for SRP graph");

    FloatTensor result;
    result.ne = {out->ne[0], out->ne[1], out->ne[2], out->ne[3]};
    result.data.resize(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data.data(), 0, ggml_nbytes(out));
    return result;
}

} // namespace

FloatTensor run_srp_input_embedding_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & mel_ggml_order,
        int seq_len) {
    return compute_srp_graph(backend, buft, store, spec, mel_ggml_order, seq_len, false);
}

FloatTensor run_srp_logits_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & mel_ggml_order,
        int seq_len) {
    return compute_srp_graph(backend, buft, store, spec, mel_ggml_order, seq_len, true);
}

} // namespace x_voice
