#include "x_voice_internal.h"

#include "ggml-cpu.h"

#if defined(GGML_USE_CUDA) || defined(GGML_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

namespace x_voice {
namespace {

int64_t key_id(gguf_context * ctx, const char * key, bool required) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0 && required) throw std::runtime_error(std::string("missing GGUF metadata key: ") + key);
    return kid;
}

bool is_int_type(gguf_type type) {
    return type == GGUF_TYPE_UINT8 || type == GGUF_TYPE_INT8 ||
           type == GGUF_TYPE_UINT16 || type == GGUF_TYPE_INT16 ||
           type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32 ||
           type == GGUF_TYPE_UINT64 || type == GGUF_TYPE_INT64;
}

int64_t read_i(gguf_context * ctx, int64_t kid, const char * key) {
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx, kid);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(ctx, kid);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, kid);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(ctx, kid);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, kid);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx, kid);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx, kid));
        case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx, kid);
        default: throw std::runtime_error(std::string("metadata key is not integer: ") + key);
    }
}

float read_f(gguf_context * ctx, int64_t kid, const char * key) {
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, kid);
        case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(ctx, kid));
        case GGUF_TYPE_UINT32: return static_cast<float>(gguf_get_val_u32(ctx, kid));
        case GGUF_TYPE_INT32: return static_cast<float>(gguf_get_val_i32(ctx, kid));
        case GGUF_TYPE_UINT64: return static_cast<float>(gguf_get_val_u64(ctx, kid));
        case GGUF_TYPE_INT64: return static_cast<float>(gguf_get_val_i64(ctx, kid));
        default: throw std::runtime_error(std::string("metadata key is not numeric: ") + key);
    }
}

std::vector<std::pair<std::string, std::vector<int64_t>>> representative_tensor_contracts() {
    return {
        {"stage2.txt.emb.weight", {512, 822}},
        {"stage2.txt.lang_ada.weight", {512, 1024}},
        {"stage2.txt.blk.0.dwconv.weight", {7, 1, 512}},
        {"stage2.txt.blk.0.pwconv1.weight", {512, 1024}},
        {"stage2.txt.blk.0.pwconv2.weight", {1024, 512}},
        {"stage2.inp.proj.weight", {712, 1024}},
        {"stage2.inp.pos.0.weight", {31, 64, 1024}},
        {"stage2.blk.0.attn.q.weight", {1024, 1024}},
        {"stage2.blk.0.ff.up.weight", {1024, 2048}},
        {"stage2.out.weight", {1024, 100}},
        {"srp.mel_proj.weight", {100, 512}},
        {"srp.blk.0.attn.q.weight", {512, 504}},
        {"srp.cls.3.weight", {512, 32}},
        {"vocos.backbone.embed.weight", {7, 100, 512}},
        {"vocos.backbone.convnext.0.dwconv.weight", {7, 1, 512}},
        {"vocos.backbone.convnext.0.pwconv1.weight", {512, 1536}},
        {"vocos.backbone.convnext.0.pwconv2.weight", {1536, 512}},
        {"vocos.head.out.weight", {512, 1026}},
    };
}

bool has_prefix(const std::string & value, const std::string & prefix) {
    return value.rfind(prefix, 0) == 0;
}

struct BackendHandle {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    std::string name;

    explicit BackendHandle(const RuntimeOptions & options) : name(options.backend) {
        ggml_time_init();
        if (name == "cpu") {
            backend = ggml_backend_cpu_init();
            buft = ggml_backend_cpu_buffer_type();
            if (backend) ggml_backend_cpu_set_n_threads(backend, options.threads);
        } else if (name == "cuda") {
#if defined(GGML_USE_CUDA) || defined(GGML_CUDA)
            backend = ggml_backend_cuda_init(options.device);
            buft = ggml_backend_cuda_buffer_type(options.device);
#else
            throw std::runtime_error("CUDA backend is not available in this x-voice-ggml-cpp build");
#endif
        } else {
            throw std::runtime_error("unsupported backend: " + name);
        }
        if (!backend || !buft) throw std::runtime_error("failed to initialize backend: " + name);
    }

    ~BackendHandle() {
        if (backend) ggml_backend_free(backend);
    }
};

std::vector<std::string> read_vocab_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open X-Voice vocab file: " + path);
    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        tokens.push_back(line);
    }
    if (tokens.empty()) throw std::runtime_error("empty X-Voice vocab file: " + path);
    return tokens;
}

std::vector<std::string> load_tokenizer_tokens(gguf_context * gguf, const RuntimeOptions & options) {
    std::vector<std::string> tokens = load_string_array(gguf, "tokenizer.ggml.tokens", false);
    if (!tokens.empty()) return tokens;
    if (!options.vocab_path.empty()) return read_vocab_file(options.vocab_path);
    const std::string default_vocab = "/root/code/ggbond/models/X-Voice/XVoice_Base_Stage1/vocab.txt";
    return read_vocab_file(default_vocab);
}

} // namespace

std::vector<std::string> load_string_array(gguf_context * ctx, const char * key, bool required) {
    const int64_t kid = key_id(ctx, key, required);
    if (kid < 0) return {};
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, kid) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("metadata key is not a string array: ") + key);
    }
    std::vector<std::string> out;
    const size_t n = gguf_get_arr_n(ctx, kid);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(ctx, kid, i));
    return out;
}

std::vector<int> optional_i_arr(gguf_context * ctx, const char * key, const std::vector<int> & fallback) {
    const int64_t kid = key_id(ctx, key, false);
    if (kid < 0) return fallback;
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY || !is_int_type(gguf_get_arr_type(ctx, kid))) {
        return fallback;
    }
    const size_t n = gguf_get_arr_n(ctx, kid);
    const gguf_type ty = gguf_get_arr_type(ctx, kid);
    const void * raw = gguf_get_arr_data(ctx, kid);
    std::vector<int> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        switch (ty) {
            case GGUF_TYPE_UINT32: out.push_back(static_cast<const uint32_t *>(raw)[i]); break;
            case GGUF_TYPE_INT32: out.push_back(static_cast<const int32_t *>(raw)[i]); break;
            case GGUF_TYPE_UINT64: out.push_back(static_cast<int>(static_cast<const uint64_t *>(raw)[i])); break;
            case GGUF_TYPE_INT64: out.push_back(static_cast<int>(static_cast<const int64_t *>(raw)[i])); break;
            case GGUF_TYPE_UINT16: out.push_back(static_cast<const uint16_t *>(raw)[i]); break;
            case GGUF_TYPE_INT16: out.push_back(static_cast<const int16_t *>(raw)[i]); break;
            case GGUF_TYPE_UINT8: out.push_back(static_cast<const uint8_t *>(raw)[i]); break;
            case GGUF_TYPE_INT8: out.push_back(static_cast<const int8_t *>(raw)[i]); break;
            default: return fallback;
        }
    }
    return out;
}

int64_t require_i(gguf_context * ctx, const char * key) {
    return read_i(ctx, key_id(ctx, key, true), key);
}

int64_t optional_i(gguf_context * ctx, const char * key, int64_t fallback) {
    const int64_t kid = key_id(ctx, key, false);
    if (kid < 0) return fallback;
    try {
        return read_i(ctx, kid, key);
    } catch (...) {
        return fallback;
    }
}

float require_f(gguf_context * ctx, const char * key) {
    return read_f(ctx, key_id(ctx, key, true), key);
}

float optional_f(gguf_context * ctx, const char * key, float fallback) {
    const int64_t kid = key_id(ctx, key, false);
    if (kid < 0) return fallback;
    try {
        return read_f(ctx, kid, key);
    } catch (...) {
        return fallback;
    }
}

bool require_bool(gguf_context * ctx, const char * key) {
    const int64_t kid = key_id(ctx, key, true);
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_BOOL) throw std::runtime_error(std::string("metadata key is not bool: ") + key);
    return gguf_get_val_bool(ctx, kid);
}

bool optional_bool(gguf_context * ctx, const char * key, bool fallback) {
    const int64_t kid = key_id(ctx, key, false);
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_BOOL) return fallback;
    return gguf_get_val_bool(ctx, kid);
}

std::string require_s(gguf_context * ctx, const char * key) {
    const int64_t kid = key_id(ctx, key, true);
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_STRING) throw std::runtime_error(std::string("metadata key is not string: ") + key);
    return gguf_get_val_str(ctx, kid);
}

std::string optional_s(gguf_context * ctx, const char * key, const std::string & fallback) {
    const int64_t kid = key_id(ctx, key, false);
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_STRING) return fallback;
    return gguf_get_val_str(ctx, kid);
}

XVoiceSpec parse_xvoice_spec(gguf_context * ctx) {
    XVoiceSpec spec;
    spec.bundle_kind = require_s(ctx, "xvoice.bundle.kind");
    spec.bundle_filename = require_s(ctx, "xvoice.bundle.filename");
    spec.languages = load_string_array(ctx, "xvoice.languages", true);

    spec.audio.sample_rate = static_cast<int>(require_i(ctx, "xvoice.audio.sample_rate"));
    spec.audio.mel_channel_count = static_cast<int>(require_i(ctx, "xvoice.audio.mel_channel_count"));
    spec.audio.hop_length = static_cast<int>(require_i(ctx, "xvoice.audio.hop_length"));
    spec.audio.win_length = static_cast<int>(require_i(ctx, "xvoice.audio.win_length"));
    spec.audio.n_fft = static_cast<int>(require_i(ctx, "xvoice.audio.n_fft"));
    spec.audio.mel_spec_type = require_s(ctx, "xvoice.audio.mel_spec_type");

    Stage2Spec & s2 = spec.stage2;
    const int text_vocab_size = static_cast<int>(require_i(ctx, "xvoice.tokenizer.vocab_size"));
    s2.model_name = require_s(ctx, "xvoice.stage2.model_name");
    s2.tokenizer = require_s(ctx, "xvoice.stage2.tokenizer");
    s2.backbone = require_s(ctx, "xvoice.stage2.backbone");
    s2.sft = require_bool(ctx, "xvoice.stage2.sft");
    s2.use_total_text = require_bool(ctx, "xvoice.stage2.use_total_text");
    s2.frame_duration = require_bool(ctx, "xvoice.stage2.frame_duration");
    s2.hidden_size = static_cast<int>(require_i(ctx, "xvoice.stage2.hidden_size"));
    s2.block_count = static_cast<int>(require_i(ctx, "xvoice.stage2.block_count"));
    s2.head_count = static_cast<int>(require_i(ctx, "xvoice.stage2.attention.head_count"));
    s2.head_dim = static_cast<int>(require_i(ctx, "xvoice.stage2.attention.key_length"));
    s2.feed_forward_mult = static_cast<int>(require_i(ctx, "xvoice.stage2.feed_forward_mult"));
    s2.feed_forward_length = static_cast<int>(require_i(ctx, "xvoice.stage2.feed_forward_length"));
    s2.time_freq_embedding_length = static_cast<int>(require_i(ctx, "xvoice.stage2.time_freq_embedding_length"));
    s2.text_dim = static_cast<int>(require_i(ctx, "xvoice.stage2.text_dim"));
    s2.text_vocab_size = text_vocab_size;
    s2.text_embedding_row_count = static_cast<int>(require_i(ctx, "xvoice.stage2.text_embedding_row_count"));
    s2.text_num_embeds = static_cast<int>(require_i(ctx, "xvoice.stage2.text_num_embeds"));
    s2.input_embedding_width = static_cast<int>(require_i(ctx, "xvoice.stage2.input_embedding_width"));
    s2.text_conv_layer_count = static_cast<int>(require_i(ctx, "xvoice.stage2.text_conv_layer_count"));
    s2.text_conv_kernel = static_cast<int>(require_i(ctx, "xvoice.stage2.text_conv_kernel"));
    s2.lang_dim = static_cast<int>(require_i(ctx, "xvoice.stage2.lang_dim"));
    s2.lang_dim_in_t = static_cast<int>(require_i(ctx, "xvoice.stage2.lang_dim_in_t"));
    s2.input_conv_pos_kernel = static_cast<int>(require_i(ctx, "xvoice.stage2.input_conv_pos_kernel"));
    s2.input_conv_pos_groups = static_cast<int>(require_i(ctx, "xvoice.stage2.input_conv_pos_groups"));
    s2.text_infill_lang_type = require_s(ctx, "xvoice.stage2.text_infill_lang_type");
    s2.time_infill_lang_type = require_s(ctx, "xvoice.stage2.time_infill_lang_type");
    s2.share_lang_embed = require_bool(ctx, "xvoice.stage2.share_lang_embed");
    s2.text_mask_padding = optional_bool(ctx, "xvoice.stage2.text_mask_padding", true);
    s2.prefix_token_id = static_cast<int>(optional_i(ctx, "xvoice.stage2.prefix_token_id", text_vocab_size));
    s2.anchor_token_ids = optional_i_arr(ctx, "xvoice.stage2.anchor_token_ids", {7, 0});
    s2.max_duration = static_cast<int>(optional_i(ctx, "xvoice.stage2.max_duration", 8192));

    SRPSpec & srp = spec.srp;
    srp.model_name = require_s(ctx, "xvoice.srp.model_name");
    srp.loss = require_s(ctx, "xvoice.srp.loss");
    srp.hidden_size = static_cast<int>(require_i(ctx, "xvoice.srp.hidden_size"));
    srp.block_count = static_cast<int>(require_i(ctx, "xvoice.srp.block_count"));
    srp.head_count = static_cast<int>(require_i(ctx, "xvoice.srp.attention.head_count"));
    srp.head_dim = static_cast<int>(require_i(ctx, "xvoice.srp.attention.key_length"));
    srp.attention_inner_length = static_cast<int>(require_i(ctx, "xvoice.srp.attention.inner_length"));
    srp.feed_forward_mult = static_cast<int>(require_i(ctx, "xvoice.srp.feed_forward_mult"));
    srp.feed_forward_length = static_cast<int>(require_i(ctx, "xvoice.srp.feed_forward_length"));
    srp.class_count = static_cast<int>(require_i(ctx, "xvoice.srp.class_count"));
    srp.speed_delta = require_f(ctx, "xvoice.srp.speed_delta");
    srp.duration_min_seconds = optional_f(ctx, "xvoice.srp.duration.min_seconds", 2.0f);
    srp.duration_max_seconds = optional_f(ctx, "xvoice.srp.duration.max_seconds", 30.0f);
    srp.conv_pos_kernel = static_cast<int>(require_i(ctx, "xvoice.srp.conv_pos_kernel"));
    srp.conv_pos_groups = static_cast<int>(require_i(ctx, "xvoice.srp.conv_pos_groups"));

    VocosSpec & vocos = spec.vocos;
    vocos.layer_count = static_cast<int>(require_i(ctx, "xvoice.vocos.backbone.layer_count"));
    vocos.dim = static_cast<int>(require_i(ctx, "xvoice.vocos.backbone.dim"));
    vocos.intermediate_dim = static_cast<int>(require_i(ctx, "xvoice.vocos.backbone.intermediate_dim"));
    vocos.conv_kernel = static_cast<int>(require_i(ctx, "xvoice.vocos.backbone.conv_kernel"));
    vocos.n_fft = static_cast<int>(require_i(ctx, "xvoice.vocos.head.n_fft"));
    vocos.hop_length = static_cast<int>(require_i(ctx, "xvoice.vocos.head.hop_length"));
    vocos.win_length = static_cast<int>(require_i(ctx, "xvoice.vocos.head.win_length"));

    SamplerSpec & sampler = spec.sampler;
    sampler.nfe_step = static_cast<int>(require_i(ctx, "xvoice.sampler.nfe_step"));
    sampler.cfg_strength = require_f(ctx, "xvoice.sampler.cfg_strength");
    sampler.cfg_strength2 = optional_f(ctx, "xvoice.sampler.cfg_strength2", 4.0f);
    sampler.cfg_schedule = optional_s(ctx, "xvoice.sampler.cfg_schedule", "square");
    sampler.cfg_decay_time = optional_f(ctx, "xvoice.sampler.cfg_decay_time", 0.6f);
    sampler.layered_default = optional_bool(ctx, "xvoice.sampler.layered_default", false);
    sampler.sway_sampling_coef = require_f(ctx, "xvoice.sampler.sway_sampling_coef");
    return spec;
}

TensorInfo tensor_info_from_ggml(const ggml_tensor * tensor) {
    TensorInfo info;
    info.name = tensor->name;
    info.type_name = ggml_type_name(tensor->type);
    info.n_dims = ggml_n_dims(tensor);
    for (int i = 0; i < 4; ++i) info.ne[i] = tensor->ne[i];
    info.nbytes = ggml_nbytes(tensor);
    return info;
}

std::string format_ne(const TensorInfo & info) {
    std::ostringstream out;
    out << "(";
    const int dims = std::max(1, info.n_dims);
    for (int i = 0; i < dims; ++i) {
        if (i) out << ", ";
        out << info.ne[i];
    }
    out << ")";
    return out.str();
}

struct XVoiceRuntime::Impl {
    std::string model_path;
    RuntimeOptions options;
    ggml_context * tensor_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    std::unique_ptr<BackendHandle> backend;
    TensorStore store;
    XVoiceSpec spec;
    ModelInfo info;
    XVoiceTokenizer tokenizer;
    std::vector<TensorInfo> tensor_infos;

    Impl(std::string path, RuntimeOptions opts) : model_path(std::move(path)), options(opts) {
        load_headers();
    }

    ~Impl() {
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (tensor_ctx) ggml_free(tensor_ctx);
    }

    void load_headers() {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &tensor_ctx;
        std::unique_ptr<gguf_context, GgufDeleter> gguf(gguf_init_from_file(model_path.c_str(), params));
        if (!gguf || !tensor_ctx) throw std::runtime_error("failed to load GGUF headers: " + model_path);

        spec = parse_xvoice_spec(gguf.get());
        info.architecture = optional_s(gguf.get(), "general.architecture");
        info.name = optional_s(gguf.get(), "general.name", "X-Voice");
        info.bundle_kind = spec.bundle_kind;
        info.bundle_filename = spec.bundle_filename;
        info.tensor_count = gguf_get_n_tensors(gguf.get());
        info.kv_count = gguf_get_n_kv(gguf.get());

        tokenizer = XVoiceTokenizer(
            load_tokenizer_tokens(gguf.get(), options),
            options.pinyin_dict_path,
            options.jieba_dict_path,
            options.zh_override_path);
        info.tokenizer_token_count = tokenizer.token_count();
        if (info.tokenizer_token_count != spec.stage2.text_vocab_size) {
            throw std::runtime_error("tokenizer token count does not match xvoice.tokenizer.vocab_size");
        }

        tensor_infos.reserve(static_cast<size_t>(info.tensor_count));
        for (int64_t i = 0; i < info.tensor_count; ++i) {
            const char * name = gguf_get_tensor_name(gguf.get(), i);
            ggml_tensor * tensor = ggml_get_tensor(tensor_ctx, name);
            if (!tensor) throw std::runtime_error(std::string("GGUF tensor not in ggml context: ") + name);
            TensorInfo ti = tensor_info_from_ggml(tensor);
            if (has_prefix(ti.name, "stage2.")) ++info.stage2_tensor_count;
            if (has_prefix(ti.name, "srp.")) ++info.srp_tensor_count;
            if (has_prefix(ti.name, "vocos.")) ++info.vocos_tensor_count;
            tensor_infos.push_back(std::move(ti));
        }

        if (options.load_tensor_data) load_tensor_payloads(gguf.get());
    }

    void load_tensor_payloads(gguf_context * gguf) {
        backend.reset(new BackendHandle(options));
        info.backend_name = ggml_backend_name(backend->backend);
        weight_buf = ggml_backend_alloc_ctx_tensors(tensor_ctx, backend->backend);
        if (!weight_buf) throw std::runtime_error("failed to allocate GGUF weights on backend: " + options.backend);
        ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        info.weight_buffer_size = ggml_backend_buffer_get_size(weight_buf);

        std::ifstream f(model_path, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open GGUF tensor payloads: " + model_path);

        const size_t data_offset = gguf_get_data_offset(gguf);
        const int64_t n_tensors = gguf_get_n_tensors(gguf);
        int64_t total_payload_bytes = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            ggml_tensor * tensor = ggml_get_tensor(tensor_ctx, name);
            if (!tensor) throw std::runtime_error(std::string("GGUF tensor not in context: ") + name);
            total_payload_bytes += static_cast<int64_t>(ggml_nbytes(tensor));
        }
        int64_t loaded_payload_bytes = 0;
        store.ctx = tensor_ctx;
        store.tensors.reserve(static_cast<size_t>(n_tensors));
        if (options.progress_callback) {
            options.progress_callback({"load tensors", 0, total_payload_bytes, options.backend, "B"});
        }
        std::vector<uint8_t> bytes;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            ggml_tensor * tensor = ggml_get_tensor(tensor_ctx, name);
            if (!tensor) throw std::runtime_error(std::string("GGUF tensor not in context: ") + name);

            const size_t nbytes = ggml_nbytes(tensor);
            bytes.resize(nbytes);
            const size_t tensor_offset = data_offset + gguf_get_tensor_offset(gguf, i);
            f.seekg(static_cast<std::streamoff>(tensor_offset));
            f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            if (!f) throw std::runtime_error(std::string("failed to read GGUF tensor payload: ") + name);

            ggml_backend_tensor_set(tensor, bytes.data(), 0, nbytes);
            store.tensors.emplace(name, tensor);
            ++info.loaded_tensor_count;
            loaded_payload_bytes += static_cast<int64_t>(nbytes);
            if (options.progress_callback) {
                options.progress_callback({"load tensors", loaded_payload_bytes, total_payload_bytes, name, "B"});
            }
        }
    }

    const TensorInfo * find_tensor(const std::string & name) const {
        for (const TensorInfo & info : tensor_infos) {
            if (info.name == name) return &info;
        }
        return nullptr;
    }
};

XVoiceRuntime::XVoiceRuntime(const std::string & model_path, const RuntimeOptions & options)
    : impl_(new Impl(model_path, options)) {}

XVoiceRuntime::~XVoiceRuntime() = default;

const ModelInfo & XVoiceRuntime::info() const {
    return impl_->info;
}

const XVoiceSpec & XVoiceRuntime::spec() const {
    return impl_->spec;
}

std::vector<TensorInfo> XVoiceRuntime::tensors() const {
    return impl_->tensor_infos;
}

TextEncoding XVoiceRuntime::encode_text(const std::string & text, TextKind kind, const std::string & language) const {
    TextEncoding encoding;
    encoding.language = normalize_language_code(language);
    encoding.language_id = language_id_for_code(impl_->spec, encoding.language);
    if (kind == TextKind::Ipa) {
        encoding.tokens = impl_->tokenizer.split_ipa_v6(text);
    } else if (kind == TextKind::Tokens) {
        encoding.tokens = impl_->tokenizer.split_token_text(text);
    } else if (kind == TextKind::Plain) {
        const std::string ipa_text = impl_->tokenizer.plain_text_to_ipa_v6(text, encoding.language);
        encoding.tokens = impl_->tokenizer.split_ipa_v6(ipa_text);
    } else {
        throw std::runtime_error("unsupported X-Voice text kind");
    }
    encoding.token_ids = impl_->tokenizer.tokens_to_ids(encoding.tokens);
    int units = 0;
    for (const std::string & token : encoding.tokens) {
        if (!token.empty() && token != " ") ++units;
    }
    encoding.unit_count = std::max(units, 1);
    return encoding;
}

FloatTensor XVoiceRuntime::run_srp_input_embedding(const std::vector<float> & mel_ggml_order, int seq_len) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_srp_input_embedding requires RuntimeOptions::load_tensor_data=true");
    }
    return run_srp_input_embedding_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        mel_ggml_order,
        seq_len);
}

FloatTensor XVoiceRuntime::run_srp_logits(const std::vector<float> & mel_ggml_order, int seq_len) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_srp_logits requires RuntimeOptions::load_tensor_data=true");
    }
    return run_srp_logits_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        mel_ggml_order,
        seq_len);
}

FloatTensor XVoiceRuntime::run_stage2_time_embedding(
        const std::vector<float> & time_hidden_ggml_order,
        const std::vector<int32_t> & time_language_ids,
        int batch_size) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_time_embedding requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_time_embedding_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        time_hidden_ggml_order,
        time_language_ids,
        batch_size);
}

FloatTensor XVoiceRuntime::run_stage2_input_embedding(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        int seq_len) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_input_embedding requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_input_embedding_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        x_ggml_order,
        cond_ggml_order,
        text_embed_ggml_order,
        seq_len);
}

FloatTensor XVoiceRuntime::run_stage2_input_embedding_batch(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        int seq_len,
        int branch_count) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_input_embedding_batch requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_input_embedding_batch_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        x_ggml_order,
        cond_ggml_order,
        text_embed_ggml_order,
        seq_len,
        branch_count);
}

FloatTensor XVoiceRuntime::run_stage2_text_embedding(
        const std::vector<int32_t> & text_ids_plus_one,
        const std::vector<int32_t> & language_ids,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        const std::vector<float> & text_keep_mask_ggml_order,
        const std::vector<float> & text_pos_embed_ggml_order,
        int seq_len) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_text_embedding requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_text_embedding_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        text_ids_plus_one,
        language_ids,
        text_no_lang_fusion_mask_ggml_order,
        text_keep_mask_ggml_order,
        text_pos_embed_ggml_order,
        seq_len);
}

FloatTensor XVoiceRuntime::run_stage2_dit_block(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_index) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_dit_block requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_dit_block_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        x_ggml_order,
        time_embed_ggml_order,
        seq_len,
        block_index);
}

FloatTensor XVoiceRuntime::run_stage2_dit_stack(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_count) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_dit_stack requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_dit_stack_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        x_ggml_order,
        time_embed_ggml_order,
        seq_len,
        block_count);
}

FloatTensor XVoiceRuntime::run_stage2_output(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_output requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_output_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        x_ggml_order,
        time_embed_ggml_order,
        seq_len);
}

FloatTensor XVoiceRuntime::run_stage2_dit_forward(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_count) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_dit_forward requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_dit_forward_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        x_ggml_order,
        time_embed_ggml_order,
        seq_len,
        block_count);
}

Stage2NoCfgSamplerResult XVoiceRuntime::run_stage2_no_cfg_sampler(
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
        int step_count) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_no_cfg_sampler requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_no_cfg_sampler_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
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
        step_count,
        impl_->options.progress_callback,
        impl_->options.profile_stage2_branches);
}

Stage2NonLayeredCfgSamplerResult XVoiceRuntime::run_stage2_nonlayered_cfg_sampler(
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
        int step_count) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_nonlayered_cfg_sampler requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_nonlayered_cfg_sampler_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
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
        step_count,
        impl_->options.progress_callback,
        impl_->options.profile_stage2_branches,
        impl_->options.stage2_batch_dit_forward);
}

Stage2LayeredCfgSamplerResult XVoiceRuntime::run_stage2_layered_cfg_sampler(
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
        int step_count) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_stage2_layered_cfg_sampler requires RuntimeOptions::load_tensor_data=true");
    }
    return run_stage2_layered_cfg_sampler_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
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
        step_count,
        impl_->options.progress_callback,
        impl_->options.profile_stage2_branches,
        impl_->options.stage2_batch_dit_forward);
}

VocosBackboneHeadResult XVoiceRuntime::run_vocos_backbone_head(
        const std::vector<float> & mel_source_order,
        int frames) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_vocos_backbone_head requires RuntimeOptions::load_tensor_data=true");
    }
    return run_vocos_backbone_head_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        mel_source_order,
        frames);
}

VocosDecodeResult XVoiceRuntime::run_vocos_decode(
        const std::vector<float> & mel_source_order,
        int frames) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("run_vocos_decode requires RuntimeOptions::load_tensor_data=true");
    }
    return run_vocos_decode_graph(
        impl_->backend->backend,
        impl_->backend->buft,
        impl_->store,
        impl_->spec,
        mel_source_order,
        frames);
}

ReferenceMelResult XVoiceRuntime::reference_mel_from_wav(
        const std::string & wav_path,
        float target_rms) const {
    if (impl_->options.progress_callback) {
        impl_->options.progress_callback({"reference mel", 0, 1, wav_path});
    }
    ReferenceMelResult result = reference_mel_from_wav_impl(impl_->spec, wav_path, target_rms);
    if (impl_->options.progress_callback) {
        impl_->options.progress_callback({"reference mel", 1, 1, wav_path});
    }
    return result;
}

XVoiceSynthesisResult XVoiceRuntime::synthesize_from_ref_mel(
        const std::string & text,
        TextKind kind,
        const std::string & language,
        const std::vector<float> & ref_mel_source_order,
        int ref_frames,
        const std::string & sampler_mode,
        int step_count,
        float speed_value,
        int noise_seed,
        bool no_ref_audio,
        bool decode_full_mel) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("synthesize_from_ref_mel requires RuntimeOptions::load_tensor_data=true");
    }
    return synthesize_from_ref_mel_impl(
        *this,
        impl_->spec,
        text,
        kind,
        language,
        ref_mel_source_order,
        ref_frames,
        sampler_mode,
        step_count,
        speed_value,
        impl_->options.auto_speed_min,
        impl_->options.auto_speed_max,
        noise_seed,
        no_ref_audio,
        decode_full_mel,
        impl_->options.progress_callback);
}

XVoiceSynthesisResult XVoiceRuntime::synthesize_from_wav(
        const std::string & text,
        TextKind kind,
        const std::string & language,
        const std::string & wav_path,
        const std::string & sampler_mode,
        int step_count,
        float speed_value,
        int noise_seed,
        bool no_ref_audio,
        bool decode_full_mel,
        float target_rms) const {
    if (!impl_->backend || !impl_->weight_buf) {
        throw std::runtime_error("synthesize_from_wav requires RuntimeOptions::load_tensor_data=true");
    }
    return synthesize_from_wav_impl(
        *this,
        impl_->spec,
        text,
        kind,
        language,
        wav_path,
        sampler_mode,
        step_count,
        speed_value,
        impl_->options.auto_speed_min,
        impl_->options.auto_speed_max,
        noise_seed,
        no_ref_audio,
        decode_full_mel,
        target_rms,
        impl_->options.progress_callback);
}

void XVoiceRuntime::validate_tensor_contract() const {
    for (const auto & item : representative_tensor_contracts()) {
        const TensorInfo * tensor = impl_->find_tensor(item.first);
        if (!tensor) throw std::runtime_error("missing representative tensor: " + item.first);
        const std::vector<int64_t> & expected = item.second;
        if (tensor->n_dims != static_cast<int>(expected.size())) {
            std::ostringstream msg;
            msg << "tensor " << item.first << " rank mismatch: got " << tensor->n_dims
                << " ne=" << format_ne(*tensor) << ", expected rank " << expected.size();
            throw std::runtime_error(msg.str());
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (tensor->ne[i] != expected[i]) {
                std::ostringstream msg;
                msg << "tensor " << item.first << " ne" << i << " mismatch: got "
                    << tensor->ne[i] << " in ne=" << format_ne(*tensor)
                    << ", expected ne" << i << "=" << expected[i];
                throw std::runtime_error(msg.str());
            }
        }
    }
}

std::string XVoiceRuntime::inspect_report() const {
    const ModelInfo & info = impl_->info;
    const XVoiceSpec & spec = impl_->spec;
    std::ostringstream out;
    out << "architecture: " << info.architecture << "\n";
    out << "name: " << info.name << "\n";
    out << "bundle: " << info.bundle_kind << " (" << info.bundle_filename << ")\n";
    out << "kv_count: " << info.kv_count << "\n";
    out << "tensor_count: " << info.tensor_count << "\n";
    out << "component_tensors: stage2=" << info.stage2_tensor_count
        << " srp=" << info.srp_tensor_count
        << " vocos=" << info.vocos_tensor_count << "\n";
    out << "languages:";
    for (const auto & lang : spec.languages) out << " " << lang;
    out << "\n";
    out << "audio: sample_rate=" << spec.audio.sample_rate
        << " mel_channels=" << spec.audio.mel_channel_count
        << " hop=" << spec.audio.hop_length
        << " win=" << spec.audio.win_length
        << " n_fft=" << spec.audio.n_fft
        << " mel_spec_type=" << spec.audio.mel_spec_type << "\n";
    out << "stage2: hidden=" << spec.stage2.hidden_size
        << " blocks=" << spec.stage2.block_count
        << " heads=" << spec.stage2.head_count
        << " head_dim=" << spec.stage2.head_dim
        << " text_dim=" << spec.stage2.text_dim
        << " vocab=" << spec.stage2.text_vocab_size
        << " prefix_token_id=" << spec.stage2.prefix_token_id
        << " anchor_token_ids=";
    for (size_t i = 0; i < spec.stage2.anchor_token_ids.size(); ++i) {
        if (i) out << ",";
        out << spec.stage2.anchor_token_ids[i];
    }
    out << "\n";
    out << "srp: hidden=" << spec.srp.hidden_size
        << " blocks=" << spec.srp.block_count
        << " heads=" << spec.srp.head_count
        << " inner=" << spec.srp.attention_inner_length
        << " class_count=" << spec.srp.class_count
        << " speed_delta=" << spec.srp.speed_delta << "\n";
    out << "vocos: layers=" << spec.vocos.layer_count
        << " dim=" << spec.vocos.dim
        << " intermediate=" << spec.vocos.intermediate_dim
        << " conv_kernel=" << spec.vocos.conv_kernel << "\n";
    out << "sampler: nfe_step=" << spec.sampler.nfe_step
        << " cfg_strength=" << spec.sampler.cfg_strength
        << " cfg_strength2=" << spec.sampler.cfg_strength2
        << " cfg_schedule=" << spec.sampler.cfg_schedule
        << " cfg_decay_time=" << spec.sampler.cfg_decay_time
        << " layered_default=" << (spec.sampler.layered_default ? "true" : "false")
        << " sway=" << spec.sampler.sway_sampling_coef << "\n";
    out << "tokenizer_tokens: " << info.tokenizer_token_count << "\n";
    if (info.loaded_tensor_count > 0) {
        out << "backend: " << info.backend_name
            << " loaded_tensors=" << info.loaded_tensor_count
            << " weight_buffer_bytes=" << info.weight_buffer_size << "\n";
    }
    return out.str();
}

} // namespace x_voice
