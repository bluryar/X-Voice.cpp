#pragma once

#include "x_voice.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace x_voice {

struct TensorStore {
    ggml_context * ctx = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;

    ggml_tensor * get(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second == nullptr) {
            throw std::runtime_error("missing GGUF tensor: " + name);
        }
        return it->second;
    }
};

struct GgufDeleter {
    void operator()(gguf_context * ctx) const {
        if (ctx) gguf_free(ctx);
    }
};

std::string normalize_language_code(const std::string & language);
int language_id_for_code(const XVoiceSpec & spec, const std::string & language);

class XVoiceTokenizer {
public:
    explicit XVoiceTokenizer(
        std::vector<std::string> tokens = {},
        std::string pinyin_dict_path = {},
        std::string jieba_dict_path = {},
        std::string zh_override_path = {});

    int token_count() const { return static_cast<int>(tokens_.size()); }
    std::vector<std::string> split_ipa_v6(const std::string & ipa_text) const;
    std::vector<std::string> split_token_text(const std::string & token_text) const;
    std::string plain_text_to_ipa_v6(const std::string & text, const std::string & language) const;
    std::vector<int32_t> tokens_to_ids(const std::vector<std::string> & tokens) const;

private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::string pinyin_dict_path_;
    std::string jieba_dict_path_;
    std::string zh_override_path_;
    int32_t pad_id_ = 0;
};

std::vector<std::string> load_string_array(gguf_context * ctx, const char * key, bool required);
std::vector<int> optional_i_arr(gguf_context * ctx, const char * key, const std::vector<int> & fallback);
int64_t require_i(gguf_context * ctx, const char * key);
int64_t optional_i(gguf_context * ctx, const char * key, int64_t fallback);
float require_f(gguf_context * ctx, const char * key);
float optional_f(gguf_context * ctx, const char * key, float fallback);
bool require_bool(gguf_context * ctx, const char * key);
bool optional_bool(gguf_context * ctx, const char * key, bool fallback);
std::string require_s(gguf_context * ctx, const char * key);
std::string optional_s(gguf_context * ctx, const char * key, const std::string & fallback = {});

XVoiceSpec parse_xvoice_spec(gguf_context * ctx);
TensorInfo tensor_info_from_ggml(const ggml_tensor * tensor);
std::string format_ne(const TensorInfo & info);

void validate_stage2_graph_contract();
void validate_srp_graph_contract();
void validate_vocos_graph_contract();

FloatTensor run_srp_input_embedding_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & mel_ggml_order,
    int seq_len);

FloatTensor run_srp_logits_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & mel_ggml_order,
    int seq_len);

FloatTensor run_stage2_time_embedding_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & time_hidden_ggml_order,
    const std::vector<int32_t> & time_language_ids,
    int batch_size);

FloatTensor run_stage2_input_embedding_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & x_ggml_order,
    const std::vector<float> & cond_ggml_order,
    const std::vector<float> & text_embed_ggml_order,
    int seq_len);

FloatTensor run_stage2_input_embedding_batch_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & x_ggml_order,
    const std::vector<float> & cond_ggml_order,
    const std::vector<float> & text_embed_ggml_order,
    int seq_len,
    int branch_count);

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
    int seq_len);

FloatTensor run_stage2_dit_block_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & x_ggml_order,
    const std::vector<float> & time_embed_ggml_order,
    int seq_len,
    int block_index);

FloatTensor run_stage2_dit_stack_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & x_ggml_order,
    const std::vector<float> & time_embed_ggml_order,
    int seq_len,
    int block_count);

FloatTensor run_stage2_output_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & x_ggml_order,
    const std::vector<float> & time_embed_ggml_order,
    int seq_len);

FloatTensor run_stage2_dit_forward_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & x_ggml_order,
    const std::vector<float> & time_embed_ggml_order,
    int seq_len,
    int block_count);

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
    bool profile_stage2_branches);

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
    bool batch_dit_forward);

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
    bool batch_dit_forward);

VocosBackboneHeadResult run_vocos_backbone_head_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & mel_source_order,
    int frames);

VocosDecodeResult run_vocos_decode_graph(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const TensorStore & store,
    const XVoiceSpec & spec,
    const std::vector<float> & mel_source_order,
    int frames);

ReferenceMelResult reference_mel_from_wav_impl(
    const XVoiceSpec & spec,
    const std::string & wav_path,
    float target_rms);

XVoiceSynthesisResult synthesize_from_ref_mel_impl(
    const XVoiceRuntime & runtime,
    const XVoiceSpec & spec,
    const std::string & text,
    TextKind kind,
    const std::string & language,
    const std::vector<float> & ref_mel_source_order,
    int ref_frames,
    const std::string & sampler_mode,
    int step_count,
    float speed_value,
    float auto_speed_min,
    float auto_speed_max,
    int noise_seed,
    bool no_ref_audio,
    bool decode_full_mel,
    const ProgressCallback & progress_callback);

XVoiceSynthesisResult synthesize_from_wav_impl(
    const XVoiceRuntime & runtime,
    const XVoiceSpec & spec,
    const std::string & text,
    TextKind kind,
    const std::string & language,
    const std::string & wav_path,
    const std::string & sampler_mode,
    int step_count,
    float speed_value,
    float auto_speed_min,
    float auto_speed_max,
    int noise_seed,
    bool no_ref_audio,
    bool decode_full_mel,
    float target_rms,
    const ProgressCallback & progress_callback);

} // namespace x_voice
