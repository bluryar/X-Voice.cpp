#include "x_voice_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace x_voice {
namespace {

constexpr float kPi = 3.14159265358979323846f;

void emit_progress(
        const ProgressCallback & progress_callback,
        const std::string & phase,
        int64_t current,
        int64_t total,
        const std::string & detail = {}) {
    if (progress_callback) progress_callback({phase, current, total, detail});
}

int argmax(const std::vector<float> & values) {
    if (values.empty()) throw std::runtime_error("SRP logits are empty");
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

float speed_value_from_class(const XVoiceSpec & spec, int speed_class) {
    if (speed_class < 0 || speed_class >= spec.srp.class_count) {
        throw std::runtime_error("SRP speed class is outside the model class range");
    }
    return static_cast<float>(speed_class + 1) * spec.srp.speed_delta;
}

float clamp_auto_speed(float speed, float min_value, float max_value) {
    if (min_value > 0.0f && max_value > 0.0f && min_value > max_value) {
        throw std::runtime_error("auto speed min must be <= auto speed max");
    }
    if (min_value > 0.0f) speed = std::max(speed, min_value);
    if (max_value > 0.0f) speed = std::min(speed, max_value);
    return speed;
}

std::pair<int, float> generated_frame_count_from_speed(
        const XVoiceSpec & spec,
        int target_unit_count,
        float speed_value) {
    if (target_unit_count <= 0) throw std::runtime_error("target unit count must be positive");
    if (speed_value <= 0.0f) throw std::runtime_error("speed value must be positive");
    float seconds = static_cast<float>(target_unit_count) / speed_value;
    seconds = std::max(spec.srp.duration_min_seconds, std::min(spec.srp.duration_max_seconds, seconds));
    const int frames = static_cast<int>((seconds * static_cast<float>(spec.audio.sample_rate)) /
                                        static_cast<float>(spec.audio.hop_length));
    return {frames, seconds};
}

std::vector<int32_t> pad_or_trim_i32(const std::vector<int32_t> & values, int length, int32_t pad_value) {
    if (length <= 0) throw std::runtime_error("pad_or_trim length must be positive");
    std::vector<int32_t> out(static_cast<size_t>(length), pad_value);
    const int copy = std::min<int>(length, static_cast<int>(values.size()));
    std::copy(values.begin(), values.begin() + copy, out.begin());
    return out;
}

std::pair<std::vector<int32_t>, int> prefix_text_padding(
        const std::vector<int32_t> & target_text_ids,
        int total_len,
        int prompt_len,
        int prefix_token,
        const std::vector<int> & anchor_token_ids) {
    if (prompt_len >= total_len) throw std::runtime_error("prompt_len must be less than total_len");
    if (prompt_len <= 0 || target_text_ids.empty()) return {target_text_ids, 0};
    const int target_mel_len = total_len - prompt_len;
    const int num_prefix = static_cast<int>(
        std::lround(static_cast<double>(prompt_len) / static_cast<double>(target_mel_len) *
                    static_cast<double>(target_text_ids.size())));
    if (num_prefix <= 0) return {target_text_ids, 0};
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(num_prefix + static_cast<int>(anchor_token_ids.size()) + target_text_ids.size()));
    out.insert(out.end(), static_cast<size_t>(num_prefix), static_cast<int32_t>(prefix_token));
    for (int id : anchor_token_ids) out.push_back(static_cast<int32_t>(id));
    out.insert(out.end(), target_text_ids.begin(), target_text_ids.end());
    return {out, num_prefix};
}

std::vector<int32_t> build_prefixed_language_ids(
        int target_token_count,
        int target_language_id,
        int total_len,
        int prompt_len,
        const std::vector<int> & anchor_token_ids,
        int unknown_language_id) {
    if (prompt_len >= total_len) throw std::runtime_error("prompt_len must be less than total_len");
    std::vector<int32_t> language_ids(static_cast<size_t>(target_token_count), static_cast<int32_t>(target_language_id));
    if (prompt_len <= 0 || target_token_count <= 0) return language_ids;
    const int target_mel_len = total_len - prompt_len;
    const int num_prefix = static_cast<int>(
        std::lround(static_cast<double>(prompt_len) / static_cast<double>(target_mel_len) *
                    static_cast<double>(target_token_count)));
    if (num_prefix <= 0) return language_ids;
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(num_prefix + static_cast<int>(anchor_token_ids.size()) + target_token_count));
    out.insert(out.end(), static_cast<size_t>(num_prefix), -1);
    out.insert(out.end(), anchor_token_ids.size(), static_cast<int32_t>(unknown_language_id));
    out.insert(out.end(), language_ids.begin(), language_ids.end());
    return out;
}

std::vector<float> text_positional_embedding(int seq_len, int dim, float theta = 10000.0f) {
    if (seq_len <= 0) throw std::runtime_error("text positional embedding seq_len must be positive");
    if (dim <= 0 || dim % 2 != 0) throw std::runtime_error("text positional embedding dim must be a positive even value");
    const int half = dim / 2;
    std::vector<float> out(static_cast<size_t>(seq_len) * static_cast<size_t>(dim));
    for (int frame = 0; frame < seq_len; ++frame) {
        for (int i = 0; i < half; ++i) {
            const float step = static_cast<float>(i * 2);
            const float freq = 1.0f / std::pow(theta, step / static_cast<float>(dim));
            const float phase = static_cast<float>(frame) * freq;
            out[static_cast<size_t>(frame) * static_cast<size_t>(dim) + static_cast<size_t>(i)] = std::cos(phase);
            out[static_cast<size_t>(frame) * static_cast<size_t>(dim) + static_cast<size_t>(half + i)] = std::sin(phase);
        }
    }
    return out;
}

std::vector<float> epss_timesteps(int steps) {
    const std::vector<int> * predefined = nullptr;
    const std::vector<int> s5{0, 2, 4, 8, 16, 32};
    const std::vector<int> s6{0, 2, 4, 6, 8, 16, 32};
    const std::vector<int> s7{0, 2, 4, 6, 8, 16, 24, 32};
    const std::vector<int> s10{0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
    const std::vector<int> s12{0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32};
    const std::vector<int> s16{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
    if (steps == 5) predefined = &s5;
    else if (steps == 6) predefined = &s6;
    else if (steps == 7) predefined = &s7;
    else if (steps == 10) predefined = &s10;
    else if (steps == 12) predefined = &s12;
    else if (steps == 16) predefined = &s16;
    if (predefined) {
        std::vector<float> out;
        out.reserve(predefined->size());
        for (int value : *predefined) out.push_back(static_cast<float>(value) / 32.0f);
        return out;
    }
    std::vector<float> out(static_cast<size_t>(steps + 1));
    for (int i = 0; i <= steps; ++i) out[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(steps);
    return out;
}

std::vector<float> sampler_timesteps(const XVoiceSpec & spec, int steps) {
    if (steps <= 0) throw std::runtime_error("sampler steps must be positive");
    std::vector<float> out = epss_timesteps(steps);
    if (spec.sampler.sway_sampling_coef != 0.0f) {
        for (float & value : out) {
            const float t = value;
            value = t + spec.sampler.sway_sampling_coef * (std::cos(kPi * 0.5f * t) - 1.0f + t);
        }
    }
    return out;
}

std::vector<float> normal_noise(int seed, size_t count) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(count);
    for (float & value : out) value = dist(rng);
    return out;
}

std::vector<float> slice_generated_mel(
        const std::vector<float> & mel,
        int prompt_frames,
        int duration,
        int channels,
        const std::vector<float> & cond_mask,
        bool decode_full_mel) {
    if (decode_full_mel || prompt_frames <= 0) return mel;
    const bool prompt_at_start = !cond_mask.empty() && cond_mask[0] > 0.5f;
    const int begin = prompt_at_start ? prompt_frames : 0;
    const int end = prompt_at_start ? duration : duration - prompt_frames;
    if (begin < 0 || end < begin || end > duration) throw std::runtime_error("invalid generated mel slice");
    std::vector<float> out(static_cast<size_t>(end - begin) * static_cast<size_t>(channels));
    std::copy(
        mel.begin() + static_cast<std::ptrdiff_t>(begin * channels),
        mel.begin() + static_cast<std::ptrdiff_t>(end * channels),
        out.begin());
    return out;
}

} // namespace

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
        float requested_speed_value,
        float auto_speed_min,
        float auto_speed_max,
        int noise_seed,
        bool no_ref_audio,
        bool decode_full_mel,
        const ProgressCallback & progress_callback) {
    if (ref_frames <= 0) throw std::runtime_error("ref_frames must be positive");
    const int channels = spec.audio.mel_channel_count;
    if (ref_mel_source_order.size() != static_cast<size_t>(ref_frames) * static_cast<size_t>(channels)) {
        throw std::runtime_error("ref_mel size must match source-order shape (ref_frames, mel_channel_count)");
    }

    XVoiceSynthesisResult result;
    emit_progress(progress_callback, "text frontend", 0, 1, text_kind_name(kind));
    result.text = runtime.encode_text(text, kind, language);
    emit_progress(progress_callback, "text frontend", 1, 1, text_kind_name(kind));

    int speed_class = -1;
    float speed = requested_speed_value;
    const bool predict_speed = speed <= 0.0f;
    emit_progress(progress_callback, "srp duration", 0, 1, predict_speed ? "predict" : "fixed");
    if (speed <= 0.0f) {
        FloatTensor logits = runtime.run_srp_logits(ref_mel_source_order, ref_frames);
        speed_class = argmax(logits.data);
        speed = speed_value_from_class(spec, speed_class);
    }
    const float raw_speed = speed;
    const int raw_speed_class = speed_class;
    std::string speed_policy = predict_speed ? "auto" : "fixed";
    if (predict_speed && (auto_speed_min > 0.0f || auto_speed_max > 0.0f)) {
        speed = clamp_auto_speed(speed, auto_speed_min, auto_speed_max);
        speed_policy = speed == raw_speed ? "auto_in_range" : "auto_clamped";
    }
    emit_progress(progress_callback, "srp duration", 1, 1, predict_speed ? "predict" : "fixed");

    emit_progress(progress_callback, "stage2 prep", 0, 1, "host tensors");
    const auto frame_info = generated_frame_count_from_speed(spec, result.text.unit_count, speed);
    const int generated_frames = frame_info.first;
    const int requested_duration = ref_frames + generated_frames;
    const int unknown_language_id = static_cast<int>(spec.languages.size());
    const auto prefixed = prefix_text_padding(
        result.text.token_ids,
        requested_duration,
        ref_frames,
        spec.stage2.prefix_token_id,
        spec.stage2.anchor_token_ids);
    const std::vector<int32_t> prefixed_lang = build_prefixed_language_ids(
        static_cast<int>(result.text.token_ids.size()),
        result.text.language_id,
        requested_duration,
        ref_frames,
        spec.stage2.anchor_token_ids,
        unknown_language_id);
    const int non_pad_text = static_cast<int>(
        std::count_if(prefixed.first.begin(), prefixed.first.end(), [](int32_t value) { return value != -1; }));
    const int duration = std::min(
        spec.stage2.max_duration,
        std::max(std::max(non_pad_text, ref_frames) + 1, requested_duration));

    std::vector<int32_t> text_raw = pad_or_trim_i32(prefixed.first, duration, -1);
    std::vector<int32_t> text_ids_plus_one(text_raw.size());
    for (size_t i = 0; i < text_raw.size(); ++i) text_ids_plus_one[i] = text_raw[i] + 1;

    const std::vector<int32_t> lang_raw = pad_or_trim_i32(prefixed_lang, duration, unknown_language_id);
    std::vector<int32_t> lang_safe(lang_raw.size());
    std::vector<float> no_lang_mask(lang_raw.size());
    for (size_t i = 0; i < lang_raw.size(); ++i) {
        no_lang_mask[i] = lang_raw[i] == -1 ? 1.0f : 0.0f;
        lang_safe[i] = lang_raw[i] == -1 ? 0 : lang_raw[i];
    }

    std::vector<float> keep_mask(text_ids_plus_one.size());
    for (size_t i = 0; i < text_ids_plus_one.size(); ++i) keep_mask[i] = text_ids_plus_one[i] != 0 ? 1.0f : 0.0f;

    std::vector<float> cond(static_cast<size_t>(duration) * static_cast<size_t>(channels), 0.0f);
    std::vector<float> cond_mask(static_cast<size_t>(duration), 0.0f);
    for (int frame = 0; frame < ref_frames; ++frame) {
        if (!no_ref_audio) {
            std::copy(
                ref_mel_source_order.begin() + static_cast<std::ptrdiff_t>(frame * channels),
                ref_mel_source_order.begin() + static_cast<std::ptrdiff_t>((frame + 1) * channels),
                cond.begin() + static_cast<std::ptrdiff_t>(frame * channels));
        }
        cond_mask[static_cast<size_t>(frame)] = 1.0f;
    }

    const std::vector<float> text_pos = text_positional_embedding(duration, spec.stage2.text_dim);
    const std::vector<int32_t> time_language_ids{static_cast<int32_t>(result.text.language_id)};
    const std::vector<float> fixed_noise = normal_noise(noise_seed, cond.size());
    const int resolved_steps = step_count > 0 ? step_count : spec.sampler.nfe_step;
    if (resolved_steps <= 0) throw std::runtime_error("Stage-2 sampler step_count must be positive");
    const std::vector<float> timesteps = sampler_timesteps(spec, resolved_steps);

    std::string mode = sampler_mode == "default"
        ? (spec.sampler.layered_default ? "cfg_layered" : "cfg_nonlayered")
        : sampler_mode;
    emit_progress(progress_callback, "stage2 prep", 1, 1, mode);

    if (mode == "no_cfg") {
        Stage2NoCfgSamplerResult sampled = runtime.run_stage2_no_cfg_sampler(
            text_ids_plus_one,
            lang_safe,
            no_lang_mask,
            keep_mask,
            text_pos,
            time_language_ids,
            cond,
            cond_mask,
            fixed_noise,
            timesteps,
            duration,
            resolved_steps);
        result.mel = sampled.mel;
    } else if (mode == "cfg_nonlayered") {
        Stage2NonLayeredCfgSamplerResult sampled = runtime.run_stage2_nonlayered_cfg_sampler(
            text_ids_plus_one,
            lang_safe,
            no_lang_mask,
            keep_mask,
            text_pos,
            time_language_ids,
            cond,
            cond_mask,
            fixed_noise,
            timesteps,
            duration,
            resolved_steps);
        result.mel = sampled.mel;
    } else if (mode == "cfg_layered") {
        Stage2LayeredCfgSamplerResult sampled = runtime.run_stage2_layered_cfg_sampler(
            text_ids_plus_one,
            lang_safe,
            no_lang_mask,
            keep_mask,
            text_pos,
            time_language_ids,
            cond,
            cond_mask,
            fixed_noise,
            timesteps,
            duration,
            resolved_steps);
        result.mel = sampled.mel;
    } else {
        throw std::runtime_error("unsupported synthesize sampler mode: " + mode);
    }

    const std::vector<float> decode_mel = slice_generated_mel(
        result.mel.data,
        ref_frames,
        duration,
        channels,
        cond_mask,
        decode_full_mel);
    const int decode_frames = static_cast<int>(decode_mel.size() / static_cast<size_t>(channels));
    result.generated_mel.ne = {channels, decode_frames, 1, 1};
    result.generated_mel.data = decode_mel;
    emit_progress(progress_callback, "vocos decode", 0, 1, std::to_string(decode_frames) + " frames");
    result.vocos = runtime.run_vocos_decode(result.generated_mel.data, decode_frames);
    emit_progress(progress_callback, "vocos decode", 1, 1, std::to_string(decode_frames) + " frames");
    result.sampler_mode = mode;
    result.prompt_frames = ref_frames;
    result.generated_frames = generated_frames;
    result.duration = duration;
    result.speed_class = speed_class;
    result.raw_speed_class = raw_speed_class;
    result.raw_speed_value = raw_speed;
    result.speed_value = speed;
    result.auto_speed_min = auto_speed_min;
    result.auto_speed_max = auto_speed_max;
    result.speed_policy = speed_policy;
    result.predicted_seconds = frame_info.second;
    return result;
}

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
        const ProgressCallback & progress_callback) {
    emit_progress(progress_callback, "reference mel", 0, 1, wav_path);
    ReferenceMelResult ref = reference_mel_from_wav_impl(spec, wav_path, target_rms);
    emit_progress(progress_callback, "reference mel", 1, 1, wav_path);
    return synthesize_from_ref_mel_impl(
        runtime,
        spec,
        text,
        kind,
        language,
        ref.mel.data,
        static_cast<int>(ref.mel.ne[1]),
        sampler_mode,
        step_count,
        speed_value,
        auto_speed_min,
        auto_speed_max,
        noise_seed,
        no_ref_audio,
        decode_full_mel,
        progress_callback);
}

} // namespace x_voice
