#include "x_voice_internal.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace x_voice {
namespace {

struct GraphArena {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocr = nullptr;

    explicit GraphArena(ggml_backend_buffer_type_t buft, size_t mem_size = 512ull * 1024ull * 1024ull) {
        ggml_init_params params{};
        params.mem_size = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("failed to initialize Vocos GGML context");
        allocr = ggml_gallocr_new(buft);
        if (!allocr) throw std::runtime_error("failed to create Vocos GGML graph allocator");
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

ggml_tensor * layer_norm_affine_2d(
        ggml_context * ctx,
        const TensorStore & store,
        ggml_tensor * x,
        const std::string & prefix,
        float eps) {
    ggml_tensor * cur = ggml_norm(ctx, x, eps);
    cur = ggml_mul(ctx, cur, channels_2d(ctx, w(store, prefix + ".weight")));
    return ggml_add(ctx, cur, bias_2d(ctx, w(store, prefix + ".bias")));
}

ggml_tensor * slice_1d(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_1d(
        ctx,
        tensor,
        length,
        static_cast<size_t>(start) * ggml_element_size(tensor));
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

ggml_tensor * channel_time_to_conv1d_in(
        ggml_context * ctx,
        ggml_tensor * x,
        int frames,
        int channels) {
    ggml_tensor * cur = ggml_cont_2d(ctx, ggml_transpose(ctx, x), frames, channels);
    return ggml_reshape_3d(ctx, cur, frames, channels, 1);
}

ggml_tensor * conv1d_out_to_channel_time_2d(
        ggml_context * ctx,
        ggml_tensor * x,
        int frames,
        int channels) {
    ggml_tensor * cur = ggml_reshape_2d(ctx, x, frames, channels);
    return ggml_cont_2d(ctx, ggml_transpose(ctx, cur), channels, frames);
}

ggml_tensor * vocos_block(
        ggml_context * ctx,
        const TensorStore & store,
        const XVoiceSpec & spec,
        ggml_tensor * x,
        int frames,
        int block_index) {
    const std::string prefix = "vocos.backbone.convnext." + std::to_string(block_index);
    ggml_tensor * residual = x;
    ggml_tensor * cur = channel_time_to_conv1d_in(ctx, x, frames, spec.vocos.dim);
    cur = conv_1d_dw_f32(ctx, w(store, prefix + ".dwconv.weight"), cur, 1, spec.vocos.conv_kernel / 2, 1);
    cur = ggml_add(ctx, cur, bias_3d(ctx, w(store, prefix + ".dwconv.bias")));
    cur = conv1d_out_to_channel_time_2d(ctx, cur, frames, spec.vocos.dim);
    cur = layer_norm_affine_2d(ctx, store, cur, prefix + ".norm", 1e-6f);
    cur = linear(ctx, store, cur, prefix + ".pwconv1");
    cur = ggml_gelu_erf(ctx, cur);
    cur = linear(ctx, store, cur, prefix + ".pwconv2");
    cur = ggml_mul(ctx, cur, channels_2d(ctx, w(store, prefix + ".gamma")));
    return ggml_add(ctx, residual, cur);
}

void check_vocos_inputs(
        const XVoiceSpec & spec,
        const std::vector<float> & mel_source_order,
        int frames) {
    if (frames <= 0) throw std::runtime_error("Vocos frames must be positive");
    if (spec.vocos.conv_kernel % 2 != 1) {
        throw std::runtime_error("Vocos Conv1d same-padding graph expects an odd conv_kernel");
    }
    const size_t expected = static_cast<size_t>(frames) * static_cast<size_t>(spec.audio.mel_channel_count);
    if (mel_source_order.size() != expected) {
        std::ostringstream msg;
        msg << "Vocos mel size mismatch: got " << mel_source_order.size()
            << ", expected " << expected << " for source-order shape (frames, mel_channel_count)";
        throw std::runtime_error(msg.str());
    }
}

FloatTensor read_output_tensor(ggml_tensor * tensor) {
    FloatTensor out;
    out.ne = {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
    out.data.resize(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, out.data.data(), 0, ggml_nbytes(tensor));
    return out;
}

std::vector<float> read_store_f32(const TensorStore & store, const std::string & name) {
    ggml_tensor * tensor = w(store, name);
    if (tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("expected F32 tensor for " + name);
    }
    std::vector<float> out(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
    return out;
}

void inverse_fft(std::vector<std::complex<double>> & values) {
    const size_t n = values.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("Vocos ISTFT inverse FFT length must be a non-zero power of two");
    }
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }

    constexpr double pi = 3.141592653589793238462643383279502884;
    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = 2.0 * pi / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> wcur(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                const std::complex<double> u = values[i + j];
                const std::complex<double> v = values[i + j + len / 2] * wcur;
                values[i + j] = u + v;
                values[i + j + len / 2] = u - v;
                wcur *= wlen;
            }
        }
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    for (std::complex<double> & value : values) value *= inv_n;
}

FloatTensor istft_from_reim(
        const XVoiceSpec & spec,
        const std::vector<float> & window_source_order,
        const std::vector<float> & real_source_order,
        const std::vector<float> & imag_source_order,
        int frames) {
    const int n_fft = spec.vocos.n_fft;
    const int hop = spec.vocos.hop_length;
    const int win_length = spec.vocos.win_length;
    const int bins = n_fft / 2 + 1;
    if (n_fft <= 0 || hop <= 0 || win_length <= 0) {
        throw std::runtime_error("Vocos ISTFT requires positive n_fft, hop_length, and win_length");
    }
    if (frames <= 0) throw std::runtime_error("Vocos ISTFT frames must be positive");
    const size_t expected_bins = static_cast<size_t>(bins) * static_cast<size_t>(frames);
    if (real_source_order.size() != expected_bins || imag_source_order.size() != expected_bins) {
        throw std::runtime_error("Vocos ISTFT real/imag sizes must match source-order shape (bins, frames)");
    }
    if (window_source_order.size() != static_cast<size_t>(win_length)) {
        throw std::runtime_error("Vocos ISTFT window size must match win_length");
    }
    if (win_length > n_fft) {
        throw std::runtime_error("Vocos ISTFT win_length must be <= n_fft");
    }

    std::vector<float> window(static_cast<size_t>(n_fft), 0.0f);
    const int left = (n_fft - win_length) / 2;
    std::copy(window_source_order.begin(), window_source_order.end(), window.begin() + left);

    const size_t total_len = static_cast<size_t>(n_fft) + static_cast<size_t>(hop) * static_cast<size_t>(frames - 1);
    std::vector<float> waveform(total_len, 0.0f);
    std::vector<float> window_sum(total_len, 0.0f);
    std::vector<float> window_sqr(window.size());
    for (size_t i = 0; i < window.size(); ++i) window_sqr[i] = window[i] * window[i];

    std::vector<std::complex<double>> spectrum(static_cast<size_t>(n_fft));
    for (int frame = 0; frame < frames; ++frame) {
        std::fill(spectrum.begin(), spectrum.end(), std::complex<double>(0.0, 0.0));
        for (int bin = 0; bin < bins; ++bin) {
            const size_t index = static_cast<size_t>(bin) * static_cast<size_t>(frames) + static_cast<size_t>(frame);
            spectrum[static_cast<size_t>(bin)] = std::complex<double>(real_source_order[index], imag_source_order[index]);
        }
        for (int bin = 1; bin < bins - 1; ++bin) {
            spectrum[static_cast<size_t>(n_fft - bin)] = std::conj(spectrum[static_cast<size_t>(bin)]);
        }
        inverse_fft(spectrum);

        const size_t start = static_cast<size_t>(frame) * static_cast<size_t>(hop);
        for (int i = 0; i < n_fft; ++i) {
            const size_t index = start + static_cast<size_t>(i);
            waveform[index] += static_cast<float>(spectrum[static_cast<size_t>(i)].real()) * window[static_cast<size_t>(i)];
            window_sum[index] += window_sqr[static_cast<size_t>(i)];
        }
    }

    for (size_t i = 0; i < waveform.size(); ++i) {
        if (window_sum[i] > 1e-8f) waveform[i] /= window_sum[i];
    }

    const size_t pad = static_cast<size_t>(n_fft / 2);
    const size_t start = total_len > 2 * pad ? pad : 0;
    const size_t end = total_len > 2 * pad ? total_len - pad : total_len;
    FloatTensor out;
    out.ne = {static_cast<int64_t>(end - start), 1, 1, 1};
    out.data.assign(waveform.begin() + static_cast<std::ptrdiff_t>(start), waveform.begin() + static_cast<std::ptrdiff_t>(end));
    return out;
}

} // namespace

VocosBackboneHeadResult run_vocos_backbone_head_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & mel_source_order,
        int frames) {
    check_vocos_inputs(spec, mel_source_order, frames);

    const int bins = spec.vocos.n_fft / 2 + 1;
    GraphArena arena(buft, 768ull * 1024ull * 1024ull);
    ggml_tensor * mel = ggml_new_tensor_2d(
        arena.ctx,
        GGML_TYPE_F32,
        spec.audio.mel_channel_count,
        frames);
    ggml_set_input(named("xvoice.vocos.mel", mel));

    ggml_tensor * cur = channel_time_to_conv1d_in(
        arena.ctx,
        mel,
        frames,
        spec.audio.mel_channel_count);
    cur = conv_1d_groups_f32(
        arena.ctx,
        store,
        "vocos.backbone.embed",
        cur,
        1,
        1,
        spec.vocos.conv_kernel / 2,
        1);
    cur = conv1d_out_to_channel_time_2d(arena.ctx, cur, frames, spec.vocos.dim);
    cur = layer_norm_affine_2d(arena.ctx, store, cur, "vocos.backbone.norm", 1e-6f);

    for (int index = 0; index < spec.vocos.layer_count; ++index) {
        cur = vocos_block(arena.ctx, store, spec, cur, frames, index);
    }

    ggml_tensor * hidden = named(
        "xvoice.vocos.final_layer_norm",
        layer_norm_affine_2d(arena.ctx, store, cur, "vocos.backbone.final_layer_norm", 1e-6f));
    ggml_tensor * head_logits = named(
        "xvoice.vocos.head.logits",
        linear(arena.ctx, store, hidden, "vocos.head.out"));
    ggml_tensor * log_mag = slice_2d_rows(arena.ctx, head_logits, 0, bins);
    ggml_tensor * phase = slice_2d_rows(arena.ctx, head_logits, bins, bins);
    ggml_tensor * mag = ggml_clamp(arena.ctx, ggml_exp(arena.ctx, log_mag), 0.0f, 100.0f);
    ggml_tensor * real_channel_time = ggml_mul(arena.ctx, mag, ggml_cos(arena.ctx, phase));
    ggml_tensor * imag_channel_time = ggml_mul(arena.ctx, mag, ggml_sin(arena.ctx, phase));
    ggml_tensor * real = named(
        "xvoice.vocos.head.real",
        ggml_cont_2d(arena.ctx, ggml_transpose(arena.ctx, real_channel_time), frames, bins));
    ggml_tensor * imag = named(
        "xvoice.vocos.head.imag",
        ggml_cont_2d(arena.ctx, ggml_transpose(arena.ctx, imag_channel_time), frames, bins));
    ggml_set_output(head_logits);
    ggml_set_output(real);
    ggml_set_output(imag);

    ggml_cgraph * graph = ggml_new_graph_custom(arena.ctx, 16384, false);
    ggml_build_forward_expand(graph, head_logits);
    ggml_build_forward_expand(graph, real);
    ggml_build_forward_expand(graph, imag);
    ggml_gallocr_reserve(arena.allocr, graph);
    ggml_gallocr_alloc_graph(arena.allocr, graph);

    ggml_backend_tensor_set(mel, mel_source_order.data(), 0, ggml_nbytes(mel));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml_backend_graph_compute failed for Vocos backbone/head graph");

    VocosBackboneHeadResult result;
    result.head_logits = read_output_tensor(head_logits);
    result.real = read_output_tensor(real);
    result.imag = read_output_tensor(imag);
    return result;
}

VocosDecodeResult run_vocos_decode_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const TensorStore & store,
        const XVoiceSpec & spec,
        const std::vector<float> & mel_source_order,
        int frames) {
    VocosDecodeResult result;
    result.neural = run_vocos_backbone_head_graph(backend, buft, store, spec, mel_source_order, frames);
    const std::vector<float> window = read_store_f32(store, "vocos.head.istft.window");
    result.waveform = istft_from_reim(spec, window, result.neural.real.data, result.neural.imag.data, frames);
    result.sample_rate = spec.audio.sample_rate;
    result.decoded_frames = frames;
    return result;
}

} // namespace x_voice
