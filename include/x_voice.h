#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace x_voice {

enum class TextKind {
    Ipa,
    Tokens,
    Plain,
};

struct ProgressEvent {
    std::string phase;
    int64_t current = 0;
    int64_t total = 0;
    std::string detail;
    std::string unit = "it";
};

using ProgressCallback = std::function<void(const ProgressEvent &)>;

struct RuntimeOptions {
    // v0 defaults to metadata/tensor-header loading only. Graph milestones will
    // enable backend tensor payload loading explicitly.
    bool load_tensor_data = false;
    std::string backend = "cpu";
    int threads = 4;
    int device = 0;
    // Transitional fallback for existing GGUF files that predate
    // tokenizer.ggml.tokens. Future GGUF exports should be self-contained.
    std::string vocab_path;
    // Optional Chinese raw-text frontend resources. Empty values use the
    // build-time defaults when the frontend is enabled.
    std::string pinyin_dict_path;
    std::string jieba_dict_path;
    std::string zh_override_path;
    // Optional host-side progress callback. It reports coarse runtime phases and
    // sampler steps; it must not affect GGML graph construction or tensor data.
    ProgressCallback progress_callback;
    // Optional profiling-only sampler instrumentation. This emits aggregate
    // Stage-2 CFG branch timings through progress_callback without changing
    // GGML graph construction or tensor values.
    bool profile_stage2_branches = false;
    // Experimental: batch CFG branches in GGML ne2 and fuse CFG combine +
    // Euler update into the graph step. This preserves values in parity
    // fixtures, but is not the default because the current conditioned
    // single-branch path is faster on the local CUDA backend.
    bool stage2_batch_dit_forward = false;
    // Optional guard for automatically predicted SRP speaking speed. Explicit
    // --speed-value / speed_value still wins; these bounds only clamp SRP
    // output when speed is predicted from the reference audio.
    float auto_speed_min = 0.0f;
    float auto_speed_max = 0.0f;
};

struct ModelInfo {
    std::string architecture;
    std::string name;
    std::string bundle_kind;
    std::string bundle_filename;
    int64_t tensor_count = 0;
    int64_t kv_count = 0;
    int stage2_tensor_count = 0;
    int srp_tensor_count = 0;
    int vocos_tensor_count = 0;
    int tokenizer_token_count = 0;
    int loaded_tensor_count = 0;
    size_t weight_buffer_size = 0;
    std::string backend_name;
};

struct AudioSpec {
    int sample_rate = 24000;
    int mel_channel_count = 100;
    int hop_length = 256;
    int win_length = 1024;
    int n_fft = 1024;
    std::string mel_spec_type;
};

struct Stage2Spec {
    std::string model_name;
    std::string tokenizer;
    std::string backbone;
    bool sft = true;
    bool use_total_text = false;
    bool frame_duration = true;
    int hidden_size = 1024;
    int block_count = 22;
    int head_count = 16;
    int head_dim = 64;
    int feed_forward_mult = 2;
    int feed_forward_length = 2048;
    int time_freq_embedding_length = 256;
    int text_dim = 512;
    int text_vocab_size = 0;
    int text_embedding_row_count = 0;
    int text_num_embeds = 0;
    int input_embedding_width = 0;
    int text_conv_layer_count = 4;
    int text_conv_kernel = 7;
    int lang_dim = 512;
    int lang_dim_in_t = 512;
    int input_conv_pos_kernel = 31;
    int input_conv_pos_groups = 16;
    std::string text_infill_lang_type;
    std::string time_infill_lang_type;
    bool share_lang_embed = true;
    bool text_mask_padding = true;
    int prefix_token_id = 0;
    std::vector<int> anchor_token_ids;
    int max_duration = 8192;
};

struct SRPSpec {
    std::string model_name;
    std::string loss;
    int hidden_size = 512;
    int block_count = 16;
    int head_count = 12;
    int head_dim = 42;
    int attention_inner_length = 504;
    int feed_forward_mult = 2;
    int feed_forward_length = 1024;
    int class_count = 32;
    float speed_delta = 0.25f;
    float duration_min_seconds = 2.0f;
    float duration_max_seconds = 30.0f;
    int conv_pos_kernel = 31;
    int conv_pos_groups = 16;
};

struct VocosSpec {
    int layer_count = 8;
    int dim = 512;
    int intermediate_dim = 1536;
    int conv_kernel = 7;
    int n_fft = 1024;
    int hop_length = 256;
    int win_length = 1024;
};

struct SamplerSpec {
    int nfe_step = 32;
    float cfg_strength = 2.5f;
    float cfg_strength2 = 4.0f;
    std::string cfg_schedule = "square";
    float cfg_decay_time = 0.6f;
    bool layered_default = false;
    float sway_sampling_coef = -1.0f;
};

struct XVoiceSpec {
    std::string bundle_kind;
    std::string bundle_filename;
    std::vector<std::string> languages;
    AudioSpec audio;
    Stage2Spec stage2;
    SRPSpec srp;
    VocosSpec vocos;
    SamplerSpec sampler;
};

struct TensorInfo {
    std::string name;
    std::string type_name;
    int n_dims = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    size_t nbytes = 0;
};

struct TextEncoding {
    std::string language;
    int language_id = -1;
    std::vector<std::string> tokens;
    std::vector<int32_t> token_ids;
    int unit_count = 0;
};

struct FloatTensor {
    std::vector<float> data;
    std::vector<int64_t> ne;
};

struct Stage2NoCfgSamplerResult {
    FloatTensor sampled;
    FloatTensor mel;
};

struct Stage2NonLayeredCfgSamplerResult {
    FloatTensor sampled;
    FloatTensor mel;
};

struct Stage2LayeredCfgSamplerResult {
    FloatTensor sampled;
    FloatTensor mel;
};

struct VocosBackboneHeadResult {
    FloatTensor head_logits;
    FloatTensor real;
    FloatTensor imag;
};

struct VocosDecodeResult {
    VocosBackboneHeadResult neural;
    FloatTensor waveform;
    int sample_rate = 0;
    int decoded_frames = 0;
};

struct ReferenceMelResult {
    FloatTensor mel;
    int source_sample_rate = 0;
    float rms = 0.0f;
};

struct XVoiceSynthesisResult {
    TextEncoding text;
    FloatTensor mel;
    FloatTensor generated_mel;
    VocosDecodeResult vocos;
    std::string sampler_mode;
    int prompt_frames = 0;
    int generated_frames = 0;
    int duration = 0;
    int speed_class = -1;
    int raw_speed_class = -1;
    float raw_speed_value = 0.0f;
    float speed_value = 0.0f;
    float auto_speed_min = 0.0f;
    float auto_speed_max = 0.0f;
    std::string speed_policy = "auto";
    float predicted_seconds = 0.0f;
};

class XVoiceRuntime {
public:
    explicit XVoiceRuntime(const std::string & model_path, const RuntimeOptions & options = {});
    ~XVoiceRuntime();

    XVoiceRuntime(const XVoiceRuntime &) = delete;
    XVoiceRuntime & operator=(const XVoiceRuntime &) = delete;

    const ModelInfo & info() const;
    const XVoiceSpec & spec() const;
    std::vector<TensorInfo> tensors() const;
    std::string inspect_report() const;

    void validate_tensor_contract() const;

    TextEncoding encode_text(const std::string & text, TextKind kind, const std::string & language) const;

    FloatTensor run_srp_input_embedding(const std::vector<float> & mel_ggml_order, int seq_len) const;
    FloatTensor run_srp_logits(const std::vector<float> & mel_ggml_order, int seq_len) const;
    FloatTensor run_stage2_time_embedding(
        const std::vector<float> & time_hidden_ggml_order,
        const std::vector<int32_t> & time_language_ids,
        int batch_size) const;
    FloatTensor run_stage2_input_embedding(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        int seq_len) const;
    FloatTensor run_stage2_input_embedding_batch(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & cond_ggml_order,
        const std::vector<float> & text_embed_ggml_order,
        int seq_len,
        int branch_count) const;
    FloatTensor run_stage2_text_embedding(
        const std::vector<int32_t> & text_ids_plus_one,
        const std::vector<int32_t> & language_ids,
        const std::vector<float> & text_no_lang_fusion_mask_ggml_order,
        const std::vector<float> & text_keep_mask_ggml_order,
        const std::vector<float> & text_pos_embed_ggml_order,
        int seq_len) const;
    FloatTensor run_stage2_dit_block(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_index) const;
    FloatTensor run_stage2_dit_stack(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_count) const;
    FloatTensor run_stage2_output(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len) const;
    FloatTensor run_stage2_dit_forward(
        const std::vector<float> & x_ggml_order,
        const std::vector<float> & time_embed_ggml_order,
        int seq_len,
        int block_count) const;
    Stage2NoCfgSamplerResult run_stage2_no_cfg_sampler(
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
        int step_count) const;
    Stage2NonLayeredCfgSamplerResult run_stage2_nonlayered_cfg_sampler(
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
        int step_count) const;
    Stage2LayeredCfgSamplerResult run_stage2_layered_cfg_sampler(
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
        int step_count) const;
    VocosBackboneHeadResult run_vocos_backbone_head(
        const std::vector<float> & mel_source_order,
        int frames) const;
    VocosDecodeResult run_vocos_decode(
        const std::vector<float> & mel_source_order,
        int frames) const;
    ReferenceMelResult reference_mel_from_wav(
        const std::string & wav_path,
        float target_rms = 0.1f) const;
    XVoiceSynthesisResult synthesize_from_ref_mel(
        const std::string & text,
        TextKind kind,
        const std::string & language,
        const std::vector<float> & ref_mel_source_order,
        int ref_frames,
        const std::string & sampler_mode = "default",
        int step_count = -1,
        float speed_value = 0.0f,
        int noise_seed = 173,
        bool no_ref_audio = false,
        bool decode_full_mel = false) const;
    XVoiceSynthesisResult synthesize_from_wav(
        const std::string & text,
        TextKind kind,
        const std::string & language,
        const std::string & wav_path,
        const std::string & sampler_mode = "default",
        int step_count = -1,
        float speed_value = 0.0f,
        int noise_seed = 173,
        bool no_ref_audio = false,
        bool decode_full_mel = false,
        float target_rms = 0.1f) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char * text_kind_name(TextKind kind);
TextKind parse_text_kind(const std::string & value);

} // namespace x_voice
