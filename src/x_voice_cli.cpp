#include "x_voice.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0 << " --model PATH [--vocab PATH] [--inspect] [--validate-tensors]\n"
        << "       " << argv0 << " --model PATH [--vocab PATH] --text TEXT --text-kind ipa|tokens|plain --language CODE --print-tokens [--pinyin-dict DIR] [--jieba-dict-dir DIR] [--zh-override TSV]\n"
        << "       " << argv0 << " --model PATH --load-tensors --synthesize --text TEXT --text-kind ipa|tokens|plain --language CODE --ref-wav WAV --output-wav WAV [--metadata-json JSON] [--output-generated-mel F32] [--preset product] [--ref-auto-trim] [--ref-start-frame N] [--ref-max-frames N] [--sampler-mode default|no_cfg|cfg_nonlayered|cfg_layered] [--step-count N] [--speed-value F] [--auto-speed-min F] [--auto-speed-max F] [--profile-stage2-branches] [--stage2-batch-dit-forward] [--progress|--no-progress] [--pinyin-dict DIR] [--jieba-dict-dir DIR] [--zh-override TSV]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-time --stage2-time-hidden F32 --stage2-time-language-ids I32 --stage2-time-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-text --seq-len N --stage2-text-ids I32 --stage2-text-language-ids I32 --stage2-text-no-lang-mask F32 --stage2-text-keep-mask F32 --stage2-text-pos-embed F32 --stage2-text-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-input --seq-len N --stage2-x F32 --stage2-cond F32 --stage2-input-text-embed F32 --stage2-input-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-block --seq-len N --block-index N --stage2-block-input F32 --stage2-block-time-embed F32 --stage2-block-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-stack --seq-len N --stack-blocks N --stage2-stack-input F32 --stage2-stack-time-embed F32 --stage2-stack-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-output --seq-len N --stage2-output-input F32 --stage2-output-time-embed F32 --stage2-output-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-forward --seq-len N --stack-blocks N --stage2-forward-input F32 --stage2-forward-time-embed F32 --stage2-forward-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-no-cfg-sampler --seq-len N --step-count N --stage2-sampler-text-ids I32 --stage2-sampler-text-language-ids I32 --stage2-sampler-text-no-lang-mask F32 --stage2-sampler-text-keep-mask F32 --stage2-sampler-text-pos-embed F32 --stage2-sampler-time-language-ids I32 --stage2-sampler-cond F32 --stage2-sampler-cond-mask F32 --stage2-sampler-fixed-noise F32 --stage2-sampler-timesteps F32 --stage2-sampler-sampled-ref F32 [--stage2-sampler-out-ref F32] [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-nonlayered-cfg-sampler --seq-len N --step-count N --stage2-sampler-text-ids I32 --stage2-sampler-text-language-ids I32 --stage2-sampler-text-no-lang-mask F32 --stage2-sampler-text-keep-mask F32 --stage2-sampler-text-pos-embed F32 --stage2-sampler-time-language-ids I32 --stage2-sampler-cond F32 --stage2-sampler-cond-mask F32 --stage2-sampler-fixed-noise F32 --stage2-sampler-timesteps F32 --stage2-sampler-sampled-ref F32 [--stage2-sampler-out-ref F32] [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-stage2-layered-cfg-sampler --seq-len N --step-count N --stage2-sampler-text-ids I32 --stage2-sampler-text-language-ids I32 --stage2-sampler-text-no-lang-mask F32 --stage2-sampler-text-keep-mask F32 --stage2-sampler-text-pos-embed F32 --stage2-sampler-time-language-ids I32 --stage2-sampler-cond F32 --stage2-sampler-cond-mask F32 --stage2-sampler-fixed-noise F32 --stage2-sampler-timesteps F32 --stage2-sampler-sampled-ref F32 [--stage2-sampler-out-ref F32] [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-srp-input --seq-len N --srp-mel F32 --srp-input-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-srp-logits --seq-len N --srp-mel F32 --srp-logits-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-vocos-head --seq-len FRAMES --vocos-mel F32 --vocos-head-logits-ref F32 --vocos-real-ref F32 --vocos-imag-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --check-vocos-waveform --seq-len FRAMES --vocos-mel F32 --vocos-waveform-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --synthesize-vocos-mel --seq-len FRAMES --vocos-mel F32 --output-wav WAV\n"
        << "       " << argv0 << " --model PATH --check-ref-mel --ref-wav WAV --ref-mel-ref F32 [--threshold F]\n"
        << "       " << argv0 << " --model PATH --load-tensors --synthesize-ref-mel --text TEXT --text-kind ipa|tokens|plain --language CODE --ref-mel F32 --ref-mel-frames N --output-wav WAV [--metadata-json JSON] [--output-generated-mel F32] [--sampler-mode default|no_cfg|cfg_nonlayered|cfg_layered] [--step-count N] [--speed-value F] [--auto-speed-min F] [--auto-speed-max F] [--progress|--no-progress]\n"
        << "       " << argv0 << " --model PATH --load-tensors --synthesize-ref-wav --text TEXT --text-kind ipa|tokens|plain --language CODE --ref-wav WAV --output-wav WAV [--metadata-json JSON] [--output-generated-mel F32] [--preset product] [--ref-auto-trim] [--ref-start-frame N] [--ref-max-frames N] [--sampler-mode default|no_cfg|cfg_nonlayered|cfg_layered] [--step-count N] [--speed-value F] [--auto-speed-min F] [--auto-speed-max F] [--profile-stage2-branches] [--stage2-batch-dit-forward] [--progress|--no-progress]\n";
}

std::string require_value(int argc, char ** argv, int & i, const std::string & flag) {
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
    return argv[++i];
}

bool stderr_is_tty() {
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(STDERR_FILENO) != 0;
#endif
}

std::string format_duration(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    const int total = static_cast<int>(seconds + 0.5);
    const int minutes = total / 60;
    const int secs = total % 60;
    std::ostringstream out;
    out << std::setw(2) << std::setfill('0') << minutes
        << ":" << std::setw(2) << std::setfill('0') << secs;
    return out.str();
}

std::string format_progress_value(double value, const std::string & unit) {
    std::ostringstream out;
    if (unit == "B") {
        out << std::fixed << std::setprecision(1) << (value / (1024.0 * 1024.0)) << "MiB";
    } else {
        out << static_cast<int64_t>(value);
    }
    return out.str();
}

std::string format_progress_rate(double value, const std::string & unit) {
    std::ostringstream out;
    if (unit == "B") {
        out << std::fixed << std::setprecision(1) << (value / (1024.0 * 1024.0)) << "MiB/s";
    } else {
        out << std::fixed << std::setprecision(2) << value << "it/s";
    }
    return out.str();
}

class ProgressPrinter {
public:
    void update(const x_voice::ProgressEvent & event) {
        if (event.phase.empty() || event.total <= 0) return;
        const auto now = Clock::now();
        const bool new_phase = !active_ || event.phase != phase_;
        const int64_t total = std::max<int64_t>(event.total, 1);
        const int64_t current = std::max<int64_t>(0, std::min<int64_t>(event.current, total));
        if (new_phase) {
            if (active_) std::cerr << "\n";
            active_ = true;
            phase_ = event.phase;
            start_ = now;
            last_render_ = now - std::chrono::milliseconds(1000);
        }
        const bool finished = current >= total;
        if (!new_phase && !finished && now - last_render_ < std::chrono::milliseconds(100)) return;
        last_render_ = now;

        const double elapsed = std::chrono::duration<double>(now - start_).count();
        const double ratio = static_cast<double>(current) / static_cast<double>(total);
        const int pct = static_cast<int>(ratio * 100.0 + 0.5);
        const int width = 28;
        const int filled = std::max(0, std::min(width, static_cast<int>(ratio * width + 0.5)));
        const double rate = elapsed > 1e-6 ? static_cast<double>(current) / elapsed : 0.0;
        const double remaining = rate > 1e-6 ? static_cast<double>(total - current) / rate : 0.0;
        const std::string unit = event.unit.empty() ? "it" : event.unit;

        std::ostringstream bar;
        bar << "\r" << phase_ << ": "
            << std::setw(3) << pct << "%|";
        for (int i = 0; i < width; ++i) bar << (i < filled ? '#' : '-');
        bar << "| " << format_progress_value(static_cast<double>(current), unit)
            << "/" << format_progress_value(static_cast<double>(total), unit)
            << " [" << format_duration(elapsed)
            << "<" << format_duration(remaining)
            << ", " << format_progress_rate(rate, unit) << "]";
        if (!event.detail.empty()) bar << " " << event.detail;
        std::cerr << bar.str() << std::flush;
        if (finished) {
            std::cerr << "\n";
            active_ = false;
        }
    }

    void finish() {
        if (active_) {
            std::cerr << "\n";
            active_ = false;
        }
    }

    ~ProgressPrinter() {
        finish();
    }

private:
    using Clock = std::chrono::steady_clock;
    bool active_ = false;
    std::string phase_;
    Clock::time_point start_ = Clock::now();
    Clock::time_point last_render_ = Clock::now();
};

struct ProgressPhaseTiming {
    std::string phase;
    int64_t current = 0;
    int64_t total = 0;
    std::string detail;
    std::string unit = "it";
    double elapsed_seconds = 0.0;
    bool finished = false;
};

class ProgressRecorder {
public:
    void update(const x_voice::ProgressEvent & event) {
        if (event.phase.empty() || event.total <= 0) return;
        const auto now = Clock::now();
        Entry & entry = entry_for_phase(event.phase);
        if (!entry.started || event.current <= 0) {
            entry.started = true;
            entry.active_segment = true;
            entry.segment_start = now;
        }
        entry.timing.phase = event.phase;
        entry.timing.current = event.current;
        entry.timing.total = event.total;
        entry.timing.detail = event.detail;
        entry.timing.unit = event.unit.empty() ? "it" : event.unit;
        if (event.current >= event.total) {
            if (entry.active_segment) {
                entry.elapsed_seconds += std::chrono::duration<double>(now - entry.segment_start).count();
                entry.active_segment = false;
            }
            entry.timing.finished = true;
        } else {
            entry.timing.finished = false;
        }
        entry.timing.elapsed_seconds = entry.elapsed_seconds;
        if (entry.active_segment) {
            entry.timing.elapsed_seconds += std::chrono::duration<double>(now - entry.segment_start).count();
        }
    }

    std::vector<ProgressPhaseTiming> timings() const {
        const auto now = Clock::now();
        std::vector<ProgressPhaseTiming> out;
        out.reserve(entries_.size());
        for (const Entry & entry : entries_) {
            ProgressPhaseTiming timing = entry.timing;
            if (entry.started && entry.active_segment) {
                timing.elapsed_seconds = entry.elapsed_seconds +
                    std::chrono::duration<double>(now - entry.segment_start).count();
            }
            out.push_back(std::move(timing));
        }
        return out;
    }

private:
    using Clock = std::chrono::steady_clock;

    struct Entry {
        ProgressPhaseTiming timing;
        Clock::time_point segment_start = Clock::now();
        double elapsed_seconds = 0.0;
        bool started = false;
        bool active_segment = false;
    };

    Entry & entry_for_phase(const std::string & phase) {
        const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry & entry) {
            return entry.timing.phase == phase;
        });
        if (it != entries_.end()) return *it;
        entries_.push_back({});
        entries_.back().timing.phase = phase;
        return entries_.back();
    }

    std::vector<Entry> entries_;
};

std::vector<float> read_f32_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("failed to open F32 file: " + path);
    const std::streamsize size = in.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("F32 file byte size is not a multiple of four: " + path);
    }
    std::vector<float> data(static_cast<size_t>(size) / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(data.data()), size);
    if (!in) throw std::runtime_error("failed to read F32 file: " + path);
    return data;
}

std::vector<int32_t> read_i32_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("failed to open I32 file: " + path);
    const std::streamsize size = in.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(int32_t)) != 0) {
        throw std::runtime_error("I32 file byte size is not a multiple of four: " + path);
    }
    std::vector<int32_t> data(static_cast<size_t>(size) / sizeof(int32_t));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(data.data()), size);
    if (!in) throw std::runtime_error("failed to read I32 file: " + path);
    return data;
}

std::vector<float> repeat_float_branch(const std::vector<float> & branch, int branch_count) {
    if (branch_count <= 0) throw std::runtime_error("branch_count must be positive");
    std::vector<float> out;
    out.reserve(branch.size() * static_cast<size_t>(branch_count));
    for (int index = 0; index < branch_count; ++index) {
        out.insert(out.end(), branch.begin(), branch.end());
    }
    return out;
}

struct Diff {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

Diff compare_f32(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("reference size mismatch: actual=" + std::to_string(actual.size()) +
                                 " expected=" + std::to_string(expected.size()));
    }
    Diff diff;
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float value = std::fabs(actual[i] - expected[i]);
        diff.max_abs = std::max(diff.max_abs, value);
        sum += value;
    }
    diff.mean_abs = actual.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(actual.size()));
    return diff;
}

int argmax(const std::vector<float> & values) {
    if (values.empty()) return -1;
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

float xvoice_srp_speed_value(int speed_class) {
    return speed_class < 0 ? 0.0f : static_cast<float>(speed_class + 1) * 0.25f;
}

void warn_if_slow_auto_speed(
        const x_voice::XVoiceSynthesisResult & result,
        float requested_speed_value,
        const std::string & text_kind,
        const std::string & language) {
    if (requested_speed_value > 0.0f) return;
    if (result.speed_class < 0) return;
    if (text_kind != "plain" || language != "zh") return;
    const float diagnostic_speed = result.raw_speed_value > 0.0f ? result.raw_speed_value : result.speed_value;
    if (diagnostic_speed > 4.5f) return;
    if (result.speed_policy == "auto_clamped") return;
    std::cerr
        << "warning: SRP predicted a slow zh plain-text speed_value="
        << diagnostic_speed
        << " (about " << result.predicted_seconds
        << "s for this text); use --speed-value 8..12 or --auto-speed-min 8 --auto-speed-max 12 for tighter narration pacing.\n";
}

void write_u16_le(std::ostream & out, uint16_t value) {
    const char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

void write_u32_le(std::ostream & out, uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

void write_i16_le(std::ostream & out, int16_t value) {
    write_u16_le(out, static_cast<uint16_t>(value));
}

void write_wav_mono_i16(const std::string & path, const std::vector<float> & waveform, int sample_rate) {
    if (sample_rate <= 0) throw std::runtime_error("WAV sample_rate must be positive");
    const std::filesystem::path output(path);
    if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open output WAV: " + path);

    const uint32_t data_bytes = static_cast<uint32_t>(waveform.size() * sizeof(int16_t));
    out.write("RIFF", 4);
    write_u32_le(out, 36u + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32_le(out, 16u);
    write_u16_le(out, 1u);
    write_u16_le(out, 1u);
    write_u32_le(out, static_cast<uint32_t>(sample_rate));
    write_u32_le(out, static_cast<uint32_t>(sample_rate * 2));
    write_u16_le(out, 2u);
    write_u16_le(out, 16u);
    out.write("data", 4);
    write_u32_le(out, data_bytes);

    for (float value : waveform) {
        const float clipped = std::max(-1.0f, std::min(1.0f, value));
        const int sample = static_cast<int>(std::lrint(clipped * 32767.0f));
        write_i16_le(out, static_cast<int16_t>(std::max(-32768, std::min(32767, sample))));
    }
    if (!out) throw std::runtime_error("failed to write output WAV: " + path);
}

void write_f32_file(const std::string & path, const std::vector<float> & values) {
    if (path.empty()) return;
    const std::filesystem::path output(path);
    if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open output F32 file: " + path);
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("failed to write output F32 file: " + path);
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(hex[(c >> 4) & 0x0f]);
                    out.push_back(hex[c & 0x0f]);
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

void write_string_array_json(std::ostream & out, const std::vector<std::string> & values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ", ";
        out << "\"" << json_escape(values[i]) << "\"";
    }
    out << "]";
}

void write_i32_array_json(std::ostream & out, const std::vector<int32_t> & values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ", ";
        out << values[i];
    }
    out << "]";
}

int detect_reference_start_frame_from_mel(
        const std::vector<float> & mel_source_order,
        int frames,
        int channels,
        float threshold_ratio,
        int margin_frames) {
    if (frames <= 0 || channels <= 0) return 0;
    const size_t expected = static_cast<size_t>(frames) * static_cast<size_t>(channels);
    if (mel_source_order.size() < expected) return 0;
    std::vector<float> scores(static_cast<size_t>(frames), 0.0f);
    for (int frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (int channel = 0; channel < channels; ++channel) {
            sum += static_cast<double>(mel_source_order[
                static_cast<size_t>(frame) * static_cast<size_t>(channels) + static_cast<size_t>(channel)]);
        }
        scores[static_cast<size_t>(frame)] = static_cast<float>(sum / static_cast<double>(channels));
    }
    const auto minmax = std::minmax_element(scores.begin(), scores.end());
    const float lo = *minmax.first;
    const float hi = *minmax.second;
    if (!(hi > lo)) return 0;
    const float ratio = std::max(0.0f, std::min(1.0f, threshold_ratio));
    const float threshold = lo + (hi - lo) * ratio;
    constexpr int kConsecutive = 3;
    for (int frame = 0; frame + kConsecutive <= frames; ++frame) {
        bool ok = true;
        for (int offset = 0; offset < kConsecutive; ++offset) {
            if (scores[static_cast<size_t>(frame + offset)] < threshold) {
                ok = false;
                break;
            }
        }
        if (ok) return std::max(0, frame - std::max(0, margin_frames));
    }
    return 0;
}

void write_synthesis_metadata_json(
        const std::string & path,
        const x_voice::XVoiceSynthesisResult & result,
        const std::string & model_path,
        const std::string & output_wav_path,
        const std::string & input_mode,
        const std::string & text_kind,
        const std::string & ref_path,
        int ref_start_frame,
        int ref_max_frames,
        int source_prompt_frames,
        int source_sample_rate,
        float ref_rms,
        float target_rms,
        const std::vector<ProgressPhaseTiming> & profile) {
    if (path.empty()) return;
    const std::filesystem::path output(path);
    if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open metadata JSON: " + path);

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"project\": \"x-voice-ggml-cpp\",\n";
    out << "  \"model\": \"" << json_escape(model_path) << "\",\n";
    out << "  \"input_mode\": \"" << json_escape(input_mode) << "\",\n";
    out << "  \"text_kind\": \"" << json_escape(text_kind) << "\",\n";
    out << "  \"language\": \"" << json_escape(result.text.language) << "\",\n";
    out << "  \"language_id\": " << result.text.language_id << ",\n";
    out << "  \"tokens\": ";
    write_string_array_json(out, result.text.tokens);
    out << ",\n";
    out << "  \"token_ids\": ";
    write_i32_array_json(out, result.text.token_ids);
    out << ",\n";
    out << "  \"unit_count\": " << result.text.unit_count << ",\n";
    out << "  \"reference\": {\n";
    out << "    \"path\": \"" << json_escape(ref_path) << "\",\n";
    out << "    \"target_rms\": " << target_rms << ",\n";
    out << "    \"ref_start_frame\": " << ref_start_frame << ",\n";
    out << "    \"ref_max_frames\": " << ref_max_frames << ",\n";
    out << "    \"source_prompt_frames\": " << source_prompt_frames << ",\n";
    out << "    \"source_sample_rate\": " << source_sample_rate << ",\n";
    out << "    \"source_rms\": " << ref_rms << "\n";
    out << "  },\n";
    out << "  \"sampler\": {\n";
    out << "    \"mode\": \"" << json_escape(result.sampler_mode) << "\",\n";
    out << "    \"duration\": " << result.duration << ",\n";
    out << "    \"speed_class\": " << result.speed_class << ",\n";
    out << "    \"raw_speed_class\": " << result.raw_speed_class << ",\n";
    out << "    \"raw_speed_value\": " << result.raw_speed_value << ",\n";
    out << "    \"speed_value\": " << result.speed_value << ",\n";
    out << "    \"speed_policy\": \"" << json_escape(result.speed_policy) << "\",\n";
    out << "    \"auto_speed_min\": " << result.auto_speed_min << ",\n";
    out << "    \"auto_speed_max\": " << result.auto_speed_max << ",\n";
    out << "    \"predicted_seconds\": " << result.predicted_seconds << "\n";
    out << "  },\n";
    out << "  \"audio\": {\n";
    out << "    \"output_wav\": \"" << json_escape(output_wav_path) << "\",\n";
    out << "    \"sample_rate\": " << result.vocos.sample_rate << ",\n";
    out << "    \"samples\": " << result.vocos.waveform.data.size() << ",\n";
    out << "    \"prompt_frames\": " << result.prompt_frames << ",\n";
    out << "    \"generated_frames\": " << result.generated_frames << ",\n";
    out << "    \"decode_frames\": " << (result.generated_mel.ne.size() > 1 ? result.generated_mel.ne[1] : 0) << "\n";
    out << "  },\n";
    out << "  \"profile\": [\n";
    for (size_t i = 0; i < profile.size(); ++i) {
        const ProgressPhaseTiming & timing = profile[i];
        out << "    {"
            << "\"phase\": \"" << json_escape(timing.phase) << "\", "
            << "\"current\": " << timing.current << ", "
            << "\"total\": " << timing.total << ", "
            << "\"unit\": \"" << json_escape(timing.unit) << "\", "
            << "\"detail\": \"" << json_escape(timing.detail) << "\", "
            << "\"elapsed_seconds\": " << std::fixed << std::setprecision(6) << timing.elapsed_seconds << std::defaultfloat << ", "
            << "\"finished\": " << (timing.finished ? "true" : "false")
            << "}";
        if (i + 1 < profile.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    if (!out) throw std::runtime_error("failed to write metadata JSON: " + path);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path = "/root/code/ggbond/models/x-voice-f32.gguf";
    std::string vocab_path;
    std::string pinyin_dict_path;
    std::string jieba_dict_path;
    std::string zh_override_path;
    std::string text;
    std::string text_kind = "ipa";
    std::string language = "en";
    std::string backend = "cpu";
    std::string stage2_time_hidden_path;
    std::string stage2_time_language_ids_path;
    std::string stage2_time_ref_path;
    std::string stage2_text_ids_path;
    std::string stage2_text_language_ids_path;
    std::string stage2_text_no_lang_mask_path;
    std::string stage2_text_keep_mask_path;
    std::string stage2_text_pos_embed_path;
    std::string stage2_text_ref_path;
    std::string stage2_x_path;
    std::string stage2_cond_path;
    std::string stage2_input_text_embed_path;
    std::string stage2_input_ref_path;
    std::string stage2_block_input_path;
    std::string stage2_block_time_embed_path;
    std::string stage2_block_ref_path;
    std::string stage2_stack_input_path;
    std::string stage2_stack_time_embed_path;
    std::string stage2_stack_ref_path;
    std::string stage2_output_input_path;
    std::string stage2_output_time_embed_path;
    std::string stage2_output_ref_path;
    std::string stage2_forward_input_path;
    std::string stage2_forward_time_embed_path;
    std::string stage2_forward_ref_path;
    std::string stage2_sampler_text_ids_path;
    std::string stage2_sampler_text_language_ids_path;
    std::string stage2_sampler_text_no_lang_mask_path;
    std::string stage2_sampler_text_keep_mask_path;
    std::string stage2_sampler_text_pos_embed_path;
    std::string stage2_sampler_time_language_ids_path;
    std::string stage2_sampler_cond_path;
    std::string stage2_sampler_cond_mask_path;
    std::string stage2_sampler_fixed_noise_path;
    std::string stage2_sampler_timesteps_path;
    std::string stage2_sampler_sampled_ref_path;
    std::string stage2_sampler_out_ref_path;
    std::string srp_mel_path;
    std::string srp_input_ref_path;
    std::string srp_logits_ref_path;
    std::string vocos_mel_path;
    std::string vocos_head_logits_ref_path;
    std::string vocos_real_ref_path;
    std::string vocos_imag_ref_path;
    std::string vocos_waveform_ref_path;
    std::string ref_mel_path;
    std::string ref_wav_path;
    std::string ref_mel_ref_path;
    std::string output_wav_path;
    std::string metadata_json_path;
    std::string output_generated_mel_path;
    std::string sampler_mode = "default";
    std::string preset;
    int threads = 4;
    int device = 0;
    int seq_len = 16;
    int ref_mel_frames = 0;
    int ref_max_frames = 0;
    int ref_start_frame = 0;
    int ref_auto_trim_margin_frames = 0;
    int block_index = 0;
    int stack_blocks = 22;
    int step_count = 32;
    int noise_seed = 173;
    float threshold = -1.0f;
    float speed_value = 0.0f;
    float auto_speed_min = 0.0f;
    float auto_speed_max = 0.0f;
    float target_rms = 0.1f;
    float ref_auto_trim_threshold_ratio = 0.80f;
    bool inspect = false;
    bool validate_tensors = false;
    bool print_tokens = false;
    bool load_tensors = false;
    bool check_stage2_time = false;
    bool check_stage2_text = false;
    bool check_stage2_input = false;
    bool check_stage2_block = false;
    bool check_stage2_stack = false;
    bool check_stage2_output = false;
    bool check_stage2_forward = false;
    bool check_stage2_no_cfg_sampler = false;
    bool check_stage2_nonlayered_cfg_sampler = false;
    bool check_stage2_layered_cfg_sampler = false;
    bool check_srp_input = false;
    bool check_srp_logits = false;
    bool check_vocos_head = false;
    bool check_vocos_waveform = false;
    bool synthesize_vocos_mel = false;
    bool synthesize_ref_mel = false;
    bool synthesize_ref_wav = false;
    bool check_ref_mel = false;
    bool no_ref_audio = false;
    bool decode_full_mel = false;
    bool progress_requested = false;
    bool progress_disabled = false;
    bool ref_auto_trim = false;
    bool profile_stage2_branches = false;
    bool stage2_batch_dit_forward = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model") model_path = require_value(argc, argv, i, arg);
            else if (arg == "--vocab") vocab_path = require_value(argc, argv, i, arg);
            else if (arg == "--pinyin-dict") pinyin_dict_path = require_value(argc, argv, i, arg);
            else if (arg == "--jieba-dict-dir") jieba_dict_path = require_value(argc, argv, i, arg);
            else if (arg == "--zh-override") zh_override_path = require_value(argc, argv, i, arg);
            else if (arg == "--text") text = require_value(argc, argv, i, arg);
            else if (arg == "--text-kind") text_kind = require_value(argc, argv, i, arg);
            else if (arg == "--language") language = require_value(argc, argv, i, arg);
            else if (arg == "--backend" || arg == "-b") backend = require_value(argc, argv, i, arg);
            else if (arg == "--threads" || arg == "-t") threads = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--device") device = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--seq-len") seq_len = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--block-index") block_index = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--stack-blocks") stack_blocks = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--step-count") step_count = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--noise-seed") noise_seed = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--threshold") threshold = std::stof(require_value(argc, argv, i, arg));
            else if (arg == "--speed-value") speed_value = std::stof(require_value(argc, argv, i, arg));
            else if (arg == "--target-rms") target_rms = std::stof(require_value(argc, argv, i, arg));
            else if (arg == "--sampler-mode") sampler_mode = require_value(argc, argv, i, arg);
            else if (arg == "--preset") preset = require_value(argc, argv, i, arg);
            else if (arg == "--auto-speed-min") auto_speed_min = std::stof(require_value(argc, argv, i, arg));
            else if (arg == "--auto-speed-max") auto_speed_max = std::stof(require_value(argc, argv, i, arg));
            else if (arg == "--ref-mel") ref_mel_path = require_value(argc, argv, i, arg);
            else if (arg == "--ref-wav") ref_wav_path = require_value(argc, argv, i, arg);
            else if (arg == "--ref-mel-ref") ref_mel_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--ref-mel-frames") ref_mel_frames = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--ref-max-frames") ref_max_frames = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--ref-start-frame") ref_start_frame = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--ref-auto-trim") ref_auto_trim = true;
            else if (arg == "--ref-auto-trim-threshold") ref_auto_trim_threshold_ratio = std::stof(require_value(argc, argv, i, arg));
            else if (arg == "--ref-auto-trim-margin-frames") ref_auto_trim_margin_frames = std::stoi(require_value(argc, argv, i, arg));
            else if (arg == "--stage2-time-hidden") stage2_time_hidden_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-time-language-ids") stage2_time_language_ids_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-time-ref") stage2_time_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-text-ids") stage2_text_ids_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-text-language-ids") stage2_text_language_ids_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-text-no-lang-mask") stage2_text_no_lang_mask_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-text-keep-mask") stage2_text_keep_mask_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-text-pos-embed") stage2_text_pos_embed_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-text-ref") stage2_text_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-x") stage2_x_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-cond") stage2_cond_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-input-text-embed") stage2_input_text_embed_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-input-ref") stage2_input_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-block-input") stage2_block_input_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-block-time-embed") stage2_block_time_embed_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-block-ref") stage2_block_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-stack-input") stage2_stack_input_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-stack-time-embed") stage2_stack_time_embed_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-stack-ref") stage2_stack_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-output-input") stage2_output_input_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-output-time-embed") stage2_output_time_embed_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-output-ref") stage2_output_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-forward-input") stage2_forward_input_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-forward-time-embed") stage2_forward_time_embed_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-forward-ref") stage2_forward_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-text-ids") stage2_sampler_text_ids_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-text-language-ids") stage2_sampler_text_language_ids_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-text-no-lang-mask") stage2_sampler_text_no_lang_mask_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-text-keep-mask") stage2_sampler_text_keep_mask_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-text-pos-embed") stage2_sampler_text_pos_embed_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-time-language-ids") stage2_sampler_time_language_ids_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-cond") stage2_sampler_cond_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-cond-mask") stage2_sampler_cond_mask_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-fixed-noise") stage2_sampler_fixed_noise_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-timesteps") stage2_sampler_timesteps_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-sampled-ref") stage2_sampler_sampled_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--stage2-sampler-out-ref") stage2_sampler_out_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--srp-mel") srp_mel_path = require_value(argc, argv, i, arg);
            else if (arg == "--srp-input-ref") srp_input_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--srp-logits-ref") srp_logits_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--vocos-mel") vocos_mel_path = require_value(argc, argv, i, arg);
            else if (arg == "--vocos-head-logits-ref") vocos_head_logits_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--vocos-real-ref") vocos_real_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--vocos-imag-ref") vocos_imag_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--vocos-waveform-ref") vocos_waveform_ref_path = require_value(argc, argv, i, arg);
            else if (arg == "--output-wav") output_wav_path = require_value(argc, argv, i, arg);
            else if (arg == "--metadata-json") metadata_json_path = require_value(argc, argv, i, arg);
            else if (arg == "--output-generated-mel") output_generated_mel_path = require_value(argc, argv, i, arg);
            else if (arg == "--no-ref-audio") no_ref_audio = true;
            else if (arg == "--decode-full-mel") decode_full_mel = true;
            else if (arg == "--progress") progress_requested = true;
            else if (arg == "--no-progress") progress_disabled = true;
            else if (arg == "--profile-stage2-branches") profile_stage2_branches = true;
            else if (arg == "--stage2-batch-dit-forward") stage2_batch_dit_forward = true;
            else if (arg == "--inspect") inspect = true;
            else if (arg == "--validate-tensors") validate_tensors = true;
            else if (arg == "--print-tokens") print_tokens = true;
            else if (arg == "--load-tensors") load_tensors = true;
            else if (arg == "--check-stage2-time") {
                check_stage2_time = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-text") {
                check_stage2_text = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-input") {
                check_stage2_input = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-block") {
                check_stage2_block = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-stack") {
                check_stage2_stack = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-output") {
                check_stage2_output = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-forward") {
                check_stage2_forward = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-no-cfg-sampler") {
                check_stage2_no_cfg_sampler = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-nonlayered-cfg-sampler") {
                check_stage2_nonlayered_cfg_sampler = true;
                load_tensors = true;
            }
            else if (arg == "--check-stage2-layered-cfg-sampler") {
                check_stage2_layered_cfg_sampler = true;
                load_tensors = true;
            }
            else if (arg == "--check-srp-input") {
                check_srp_input = true;
                load_tensors = true;
            } else if (arg == "--check-srp-logits") {
                check_srp_logits = true;
                load_tensors = true;
            }
            else if (arg == "--check-vocos-head") {
                check_vocos_head = true;
                load_tensors = true;
            }
            else if (arg == "--check-vocos-waveform") {
                check_vocos_waveform = true;
                load_tensors = true;
            }
            else if (arg == "--synthesize-vocos-mel") {
                synthesize_vocos_mel = true;
                load_tensors = true;
            }
            else if (arg == "--synthesize-ref-mel") {
                synthesize_ref_mel = true;
                load_tensors = true;
            }
            else if (arg == "--synthesize") {
                synthesize_ref_wav = true;
                load_tensors = true;
            }
            else if (arg == "--synthesize-ref-wav") {
                synthesize_ref_wav = true;
                load_tensors = true;
            }
            else if (arg == "--check-ref-mel") {
                check_ref_mel = true;
            }
            else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (!inspect && !validate_tensors && !print_tokens && !check_stage2_time && !check_stage2_text && !check_stage2_input && !check_stage2_block && !check_stage2_stack && !check_stage2_output && !check_stage2_forward && !check_stage2_no_cfg_sampler && !check_stage2_nonlayered_cfg_sampler && !check_stage2_layered_cfg_sampler && !check_srp_input && !check_srp_logits && !check_vocos_head && !check_vocos_waveform && !synthesize_vocos_mel && !synthesize_ref_mel && !synthesize_ref_wav && !check_ref_mel) inspect = true;

        if (!preset.empty()) {
            if (preset != "product") throw std::runtime_error("unsupported --preset: " + preset);
            if (sampler_mode == "default") sampler_mode = "cfg_nonlayered";
            if (!ref_auto_trim && ref_start_frame == 0 && ref_max_frames == 0) {
                ref_auto_trim = true;
                ref_max_frames = 384;
            }
            if (speed_value <= 0.0f && auto_speed_min <= 0.0f && auto_speed_max <= 0.0f) {
                auto_speed_min = 8.0f;
                auto_speed_max = 12.0f;
            }
        }
        if (auto_speed_min < 0.0f || auto_speed_max < 0.0f) {
            throw std::runtime_error("--auto-speed-min/max must be non-negative");
        }
        if (auto_speed_min > 0.0f && auto_speed_max > 0.0f && auto_speed_min > auto_speed_max) {
            throw std::runtime_error("--auto-speed-min must be <= --auto-speed-max");
        }

        const bool synthesis_mode = synthesize_vocos_mel || synthesize_ref_mel || synthesize_ref_wav;
        const bool show_progress = !progress_disabled && (progress_requested || (synthesis_mode && stderr_is_tty()));
        ProgressPrinter progress;
        ProgressRecorder progress_recorder;
        auto record_progress = [&progress, &progress_recorder, show_progress](const x_voice::ProgressEvent & event) {
            progress_recorder.update(event);
            if (show_progress) progress.update(event);
        };

        x_voice::RuntimeOptions options;
        options.vocab_path = vocab_path;
        options.pinyin_dict_path = pinyin_dict_path;
        options.jieba_dict_path = jieba_dict_path;
        options.zh_override_path = zh_override_path;
        options.load_tensor_data = load_tensors;
        options.backend = backend;
        options.threads = threads;
        options.device = device;
        options.profile_stage2_branches = profile_stage2_branches;
        options.stage2_batch_dit_forward = stage2_batch_dit_forward;
        options.auto_speed_min = auto_speed_min;
        options.auto_speed_max = auto_speed_max;
        if (synthesis_mode || show_progress) {
            options.progress_callback = record_progress;
        }
        x_voice::XVoiceRuntime runtime(model_path, options);

        if (inspect) {
            std::cout << runtime.inspect_report();
        }

        if (validate_tensors) {
            runtime.validate_tensor_contract();
            std::cout << "tensor_contract: ok\n";
        }

        if (print_tokens) {
            if (text.empty()) throw std::runtime_error("--print-tokens requires --text");
            const x_voice::TextKind kind = x_voice::parse_text_kind(text_kind);
            const x_voice::TextEncoding encoding = runtime.encode_text(text, kind, language);
            std::cout << "text_kind: " << x_voice::text_kind_name(kind) << "\n";
            std::cout << "language: " << encoding.language << "\n";
            std::cout << "language_id: " << encoding.language_id << "\n";
            std::cout << "unit_count: " << encoding.unit_count << "\n";
            std::cout << "tokens:";
            for (const auto & token : encoding.tokens) std::cout << " [" << token << "]";
            std::cout << "\n";
            std::cout << "token_ids:";
            for (int32_t id : encoding.token_ids) std::cout << " " << id;
            std::cout << "\n";
        }

        if (check_ref_mel) {
            if (ref_wav_path.empty()) throw std::runtime_error("--check-ref-mel requires --ref-wav");
            if (ref_mel_ref_path.empty()) throw std::runtime_error("--check-ref-mel requires --ref-mel-ref");
            const x_voice::ReferenceMelResult actual = runtime.reference_mel_from_wav(ref_wav_path, target_rms);
            const std::vector<float> expected = read_f32_file(ref_mel_ref_path);
            const Diff diff = compare_f32(actual.mel.data, expected);
            const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
            std::cout << "reference_mel: " << (passed ? "ok" : "fail")
                      << " ne=(" << actual.mel.ne[0] << ", " << actual.mel.ne[1] << ")"
                      << " source_sample_rate=" << actual.source_sample_rate
                      << " rms=" << actual.rms
                      << " max_abs=" << diff.max_abs
                      << " mean_abs=" << diff.mean_abs << "\n";
            if (!passed) return 1;
        }

        if (synthesize_ref_mel) {
            if (text.empty()) throw std::runtime_error("--synthesize-ref-mel requires --text");
            if (ref_mel_path.empty()) throw std::runtime_error("--synthesize-ref-mel requires --ref-mel");
            if (ref_mel_frames <= 0) throw std::runtime_error("--synthesize-ref-mel requires --ref-mel-frames > 0");
            if (output_wav_path.empty()) throw std::runtime_error("--synthesize-ref-mel requires --output-wav");
            const std::vector<float> ref_mel = read_f32_file(ref_mel_path);
            const x_voice::TextKind kind = x_voice::parse_text_kind(text_kind);
            const x_voice::XVoiceSynthesisResult result = runtime.synthesize_from_ref_mel(
                text,
                kind,
                language,
                ref_mel,
                ref_mel_frames,
                sampler_mode,
                step_count,
                speed_value,
                noise_seed,
                no_ref_audio,
                decode_full_mel);
            record_progress({"write wav", 0, 1, output_wav_path});
            write_wav_mono_i16(output_wav_path, result.vocos.waveform.data, result.vocos.sample_rate);
            record_progress({"write wav", 1, 1, output_wav_path});
            write_f32_file(output_generated_mel_path, result.generated_mel.data);
            write_synthesis_metadata_json(
                metadata_json_path,
                result,
                model_path,
                output_wav_path,
                "ref_mel",
                text_kind,
                ref_mel_path,
                0,
                ref_mel_frames,
                ref_mel_frames,
                0,
                0.0f,
                target_rms,
                progress_recorder.timings());
            std::cout << "synthesize_ref_mel: ok"
                      << " output_wav=" << output_wav_path
                      << (metadata_json_path.empty() ? "" : (" metadata_json=" + metadata_json_path))
                      << (output_generated_mel_path.empty() ? "" : (" output_generated_mel=" + output_generated_mel_path))
                      << " sampler_mode=" << result.sampler_mode
                      << " prompt_frames=" << result.prompt_frames
                      << " generated_frames=" << result.generated_frames
                      << " decode_frames=" << result.generated_mel.ne[1]
                      << " duration=" << result.duration
                      << " speed_class=" << result.speed_class
                      << " raw_speed_value=" << result.raw_speed_value
                      << " speed_value=" << result.speed_value
                      << " speed_policy=" << result.speed_policy
                      << " predicted_seconds=" << result.predicted_seconds
                      << " samples=" << result.vocos.waveform.data.size()
                      << " sample_rate=" << result.vocos.sample_rate << "\n";
            warn_if_slow_auto_speed(result, speed_value, text_kind, language);
        }

        if (synthesize_ref_wav) {
            if (text.empty()) throw std::runtime_error("--synthesize-ref-wav requires --text");
            if (ref_wav_path.empty()) throw std::runtime_error("--synthesize-ref-wav requires --ref-wav");
            if (output_wav_path.empty()) throw std::runtime_error("--synthesize-ref-wav requires --output-wav");
            const x_voice::TextKind kind = x_voice::parse_text_kind(text_kind);
            x_voice::XVoiceSynthesisResult result;
            int source_prompt_frames = 0;
            int source_sample_rate = 0;
            float ref_rms = 0.0f;
            int effective_ref_start_frame = ref_start_frame;
            int effective_ref_max_frames = ref_max_frames;
            if (ref_auto_trim && effective_ref_max_frames <= 0) effective_ref_max_frames = 384;
            if (effective_ref_max_frames > 0) {
                const x_voice::ReferenceMelResult reference = runtime.reference_mel_from_wav(ref_wav_path, target_rms);
                if (reference.mel.ne.size() < 2 || reference.mel.ne[0] <= 0 || reference.mel.ne[1] <= 0) {
                    throw std::runtime_error("reference WAV frontend returned an invalid mel tensor");
                }
                const int channels = static_cast<int>(reference.mel.ne[0]);
                source_prompt_frames = static_cast<int>(reference.mel.ne[1]);
                source_sample_rate = reference.source_sample_rate;
                ref_rms = reference.rms;
                const size_t source_values = static_cast<size_t>(source_prompt_frames) * static_cast<size_t>(channels);
                if (reference.mel.data.size() < source_values) {
                    throw std::runtime_error("reference WAV frontend returned fewer mel values than its GGML ne contract");
                }
                if (ref_auto_trim && ref_start_frame == 0) {
                    effective_ref_start_frame = detect_reference_start_frame_from_mel(
                        reference.mel.data,
                        source_prompt_frames,
                        channels,
                        ref_auto_trim_threshold_ratio,
                        ref_auto_trim_margin_frames);
                }
                if (effective_ref_start_frame < 0 || effective_ref_start_frame >= source_prompt_frames) {
                    throw std::runtime_error("--ref-start-frame must be in [0, source_prompt_frames)");
                }
                const int available_frames = source_prompt_frames - effective_ref_start_frame;
                const int used_frames = std::min(effective_ref_max_frames, available_frames);
                const size_t used_values = static_cast<size_t>(used_frames) * static_cast<size_t>(channels);
                const size_t start_values = static_cast<size_t>(effective_ref_start_frame) * static_cast<size_t>(channels);
                std::vector<float> trimmed_ref_mel(
                    reference.mel.data.begin() + static_cast<std::ptrdiff_t>(start_values),
                    reference.mel.data.begin() + static_cast<std::ptrdiff_t>(start_values + used_values));
                result = runtime.synthesize_from_ref_mel(
                    text,
                    kind,
                    language,
                    trimmed_ref_mel,
                    used_frames,
                    sampler_mode,
                    step_count,
                    speed_value,
                    noise_seed,
                    no_ref_audio,
                    decode_full_mel);
            } else {
                effective_ref_start_frame = 0;
                effective_ref_max_frames = 0;
                result = runtime.synthesize_from_wav(
                    text,
                    kind,
                    language,
                    ref_wav_path,
                    sampler_mode,
                    step_count,
                    speed_value,
                    noise_seed,
                    no_ref_audio,
                    decode_full_mel,
                    target_rms);
            }
            record_progress({"write wav", 0, 1, output_wav_path});
            write_wav_mono_i16(output_wav_path, result.vocos.waveform.data, result.vocos.sample_rate);
            record_progress({"write wav", 1, 1, output_wav_path});
            write_f32_file(output_generated_mel_path, result.generated_mel.data);
            write_synthesis_metadata_json(
                metadata_json_path,
                result,
                model_path,
                output_wav_path,
                "ref_wav",
                text_kind,
                ref_wav_path,
                effective_ref_start_frame,
                effective_ref_max_frames,
                source_prompt_frames > 0 ? source_prompt_frames : result.prompt_frames,
                source_sample_rate,
                ref_rms,
                target_rms,
                progress_recorder.timings());
            std::cout << "synthesize_ref_wav: ok"
                      << " output_wav=" << output_wav_path
                      << (metadata_json_path.empty() ? "" : (" metadata_json=" + metadata_json_path))
                      << (output_generated_mel_path.empty() ? "" : (" output_generated_mel=" + output_generated_mel_path))
                      << " sampler_mode=" << result.sampler_mode
                      << " prompt_frames=" << result.prompt_frames;
            if (source_prompt_frames > 0) {
                std::cout << " source_prompt_frames=" << source_prompt_frames
                          << " source_sample_rate=" << source_sample_rate
                          << " ref_rms=" << ref_rms
                          << " ref_start_frame=" << effective_ref_start_frame
                          << " ref_max_frames=" << effective_ref_max_frames;
            }
            std::cout << " generated_frames=" << result.generated_frames
                      << " decode_frames=" << result.generated_mel.ne[1]
                      << " duration=" << result.duration
                      << " speed_class=" << result.speed_class
                      << " raw_speed_value=" << result.raw_speed_value
                      << " speed_value=" << result.speed_value
                      << " speed_policy=" << result.speed_policy
                      << " predicted_seconds=" << result.predicted_seconds
                      << " samples=" << result.vocos.waveform.data.size()
                      << " sample_rate=" << result.vocos.sample_rate << "\n";
            warn_if_slow_auto_speed(result, speed_value, text_kind, language);
        }

        if (check_stage2_time) {
            if (stage2_time_hidden_path.empty()) throw std::runtime_error("--check-stage2-time requires --stage2-time-hidden");
            if (stage2_time_language_ids_path.empty()) throw std::runtime_error("--check-stage2-time requires --stage2-time-language-ids");
            if (stage2_time_ref_path.empty()) throw std::runtime_error("--check-stage2-time requires --stage2-time-ref");
            const std::vector<float> time_hidden = read_f32_file(stage2_time_hidden_path);
            const std::vector<int32_t> language_ids = read_i32_file(stage2_time_language_ids_path);
            const x_voice::FloatTensor actual = runtime.run_stage2_time_embedding(time_hidden, language_ids, static_cast<int>(language_ids.size()));
            const std::vector<float> expected = read_f32_file(stage2_time_ref_path);
            const Diff diff = compare_f32(actual.data, expected);
            const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
            std::cout << "stage2_time_embed: " << (passed ? "ok" : "fail")
                      << " ne=(" << actual.ne[0] << ", " << actual.ne[1] << ")"
                      << " max_abs=" << diff.max_abs
                      << " mean_abs=" << diff.mean_abs << "\n";
            if (!passed) return 1;
        }

        if (check_stage2_text) {
            if (stage2_text_ids_path.empty()) throw std::runtime_error("--check-stage2-text requires --stage2-text-ids");
            if (stage2_text_language_ids_path.empty()) throw std::runtime_error("--check-stage2-text requires --stage2-text-language-ids");
            if (stage2_text_no_lang_mask_path.empty()) throw std::runtime_error("--check-stage2-text requires --stage2-text-no-lang-mask");
            if (stage2_text_keep_mask_path.empty()) throw std::runtime_error("--check-stage2-text requires --stage2-text-keep-mask");
            if (stage2_text_pos_embed_path.empty()) throw std::runtime_error("--check-stage2-text requires --stage2-text-pos-embed");
            if (stage2_text_ref_path.empty()) throw std::runtime_error("--check-stage2-text requires --stage2-text-ref");
            const std::vector<int32_t> text_ids = read_i32_file(stage2_text_ids_path);
            const std::vector<int32_t> language_ids = read_i32_file(stage2_text_language_ids_path);
            const std::vector<float> no_lang_mask = read_f32_file(stage2_text_no_lang_mask_path);
            const std::vector<float> keep_mask = read_f32_file(stage2_text_keep_mask_path);
            const std::vector<float> pos_embed = read_f32_file(stage2_text_pos_embed_path);
            const x_voice::FloatTensor actual = runtime.run_stage2_text_embedding(
                text_ids,
                language_ids,
                no_lang_mask,
                keep_mask,
                pos_embed,
                seq_len);
            const std::vector<float> expected = read_f32_file(stage2_text_ref_path);
            const Diff diff = compare_f32(actual.data, expected);
            const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
            std::cout << "stage2_text_embed: " << (passed ? "ok" : "fail")
                      << " ne=(" << actual.ne[0] << ", " << actual.ne[1] << ")"
                      << " max_abs=" << diff.max_abs
                      << " mean_abs=" << diff.mean_abs << "\n";
            if (!passed) return 1;
        }

        if (check_stage2_input) {
            if (stage2_x_path.empty()) throw std::runtime_error("--check-stage2-input requires --stage2-x");
            if (stage2_cond_path.empty()) throw std::runtime_error("--check-stage2-input requires --stage2-cond");
            if (stage2_input_text_embed_path.empty()) throw std::runtime_error("--check-stage2-input requires --stage2-input-text-embed");
            if (stage2_input_ref_path.empty()) throw std::runtime_error("--check-stage2-input requires --stage2-input-ref");
            const std::vector<float> x = read_f32_file(stage2_x_path);
            const std::vector<float> cond = read_f32_file(stage2_cond_path);
            const std::vector<float> text_embed = read_f32_file(stage2_input_text_embed_path);
            const std::vector<float> expected = read_f32_file(stage2_input_ref_path);
            x_voice::FloatTensor actual;
            std::vector<float> actual_compare;
            if (stage2_batch_dit_forward) {
                const int branch_count = 2;
                actual = runtime.run_stage2_input_embedding_batch(
                    repeat_float_branch(x, branch_count),
                    repeat_float_branch(cond, branch_count),
                    repeat_float_branch(text_embed, branch_count),
                    seq_len,
                    branch_count);
                actual_compare.assign(actual.data.begin(), actual.data.begin() + static_cast<std::ptrdiff_t>(expected.size()));
            } else {
                actual = runtime.run_stage2_input_embedding(x, cond, text_embed, seq_len);
                actual_compare = actual.data;
            }
            const Diff diff = compare_f32(actual_compare, expected);
            const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
            std::cout << "stage2_input_embed: " << (passed ? "ok" : "fail")
                      << " ne=(" << actual.ne[0] << ", " << actual.ne[1];
            if (actual.ne.size() > 2 && actual.ne[2] > 1) std::cout << ", " << actual.ne[2];
            std::cout << ")"
                      << " max_abs=" << diff.max_abs
                      << " mean_abs=" << diff.mean_abs << "\n";
            if (!passed) return 1;
        }

        if (check_stage2_block) {
            if (stage2_block_input_path.empty()) throw std::runtime_error("--check-stage2-block requires --stage2-block-input");
            if (stage2_block_time_embed_path.empty()) throw std::runtime_error("--check-stage2-block requires --stage2-block-time-embed");
            if (stage2_block_ref_path.empty()) throw std::runtime_error("--check-stage2-block requires --stage2-block-ref");
            const std::vector<float> x = read_f32_file(stage2_block_input_path);
            const std::vector<float> time_embed = read_f32_file(stage2_block_time_embed_path);
            const x_voice::FloatTensor actual = runtime.run_stage2_dit_block(x, time_embed, seq_len, block_index);
            const std::vector<float> expected = read_f32_file(stage2_block_ref_path);
            const Diff diff = compare_f32(actual.data, expected);
            const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
            std::cout << "stage2_block" << block_index << ": " << (passed ? "ok" : "fail")
                      << " ne=(" << actual.ne[0] << ", " << actual.ne[1] << ")"
                      << " max_abs=" << diff.max_abs
                      << " mean_abs=" << diff.mean_abs << "\n";
            if (!passed) return 1;
        }

        if (check_stage2_stack) {
            if (stage2_stack_input_path.empty()) throw std::runtime_error("--check-stage2-stack requires --stage2-stack-input");
            if (stage2_stack_time_embed_path.empty()) throw std::runtime_error("--check-stage2-stack requires --stage2-stack-time-embed");
            if (stage2_stack_ref_path.empty()) throw std::runtime_error("--check-stage2-stack requires --stage2-stack-ref");
            const std::vector<float> x = read_f32_file(stage2_stack_input_path);
            const std::vector<float> time_embed = read_f32_file(stage2_stack_time_embed_path);
            const x_voice::FloatTensor actual = runtime.run_stage2_dit_stack(x, time_embed, seq_len, stack_blocks);
            const std::vector<float> expected = read_f32_file(stage2_stack_ref_path);
            const Diff diff = compare_f32(actual.data, expected);
            const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
            std::cout << "stage2_stack" << stack_blocks << ": " << (passed ? "ok" : "fail")
                      << " ne=(" << actual.ne[0] << ", " << actual.ne[1] << ")"
                      << " max_abs=" << diff.max_abs
                      << " mean_abs=" << diff.mean_abs << "\n";
            if (!passed) return 1;
        }

        if (check_stage2_output) {
            if (stage2_output_input_path.empty()) throw std::runtime_error("--check-stage2-output requires --stage2-output-input");
            if (stage2_output_time_embed_path.empty()) throw std::runtime_error("--check-stage2-output requires --stage2-output-time-embed");
            if (stage2_output_ref_path.empty()) throw std::runtime_error("--check-stage2-output requires --stage2-output-ref");
            const std::vector<float> x = read_f32_file(stage2_output_input_path);
            const std::vector<float> time_embed = read_f32_file(stage2_output_time_embed_path);
            const x_voice::FloatTensor actual = runtime.run_stage2_output(x, time_embed, seq_len);
            const std::vector<float> expected = read_f32_file(stage2_output_ref_path);
            const Diff diff = compare_f32(actual.data, expected);
            const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
            std::cout << "stage2_output: " << (passed ? "ok" : "fail")
                      << " ne=(" << actual.ne[0] << ", " << actual.ne[1] << ")"
                      << " max_abs=" << diff.max_abs
                      << " mean_abs=" << diff.mean_abs << "\n";
            if (!passed) return 1;
        }

        if (check_stage2_forward) {
            if (stage2_forward_input_path.empty()) throw std::runtime_error("--check-stage2-forward requires --stage2-forward-input");
            if (stage2_forward_time_embed_path.empty()) throw std::runtime_error("--check-stage2-forward requires --stage2-forward-time-embed");
            if (stage2_forward_ref_path.empty()) throw std::runtime_error("--check-stage2-forward requires --stage2-forward-ref");
            const std::vector<float> x = read_f32_file(stage2_forward_input_path);
            const std::vector<float> time_embed = read_f32_file(stage2_forward_time_embed_path);
            const x_voice::FloatTensor actual = runtime.run_stage2_dit_forward(x, time_embed, seq_len, stack_blocks);
            const std::vector<float> expected = read_f32_file(stage2_forward_ref_path);
            const Diff diff = compare_f32(actual.data, expected);
            const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
            std::cout << "stage2_forward" << stack_blocks << ": " << (passed ? "ok" : "fail")
                      << " ne=(" << actual.ne[0] << ", " << actual.ne[1] << ")"
                      << " max_abs=" << diff.max_abs
                      << " mean_abs=" << diff.mean_abs << "\n";
            if (!passed) return 1;
        }

        if (check_stage2_no_cfg_sampler) {
            if (stage2_sampler_text_ids_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-text-ids");
            if (stage2_sampler_text_language_ids_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-text-language-ids");
            if (stage2_sampler_text_no_lang_mask_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-text-no-lang-mask");
            if (stage2_sampler_text_keep_mask_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-text-keep-mask");
            if (stage2_sampler_text_pos_embed_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-text-pos-embed");
            if (stage2_sampler_time_language_ids_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-time-language-ids");
            if (stage2_sampler_cond_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-cond");
            if (stage2_sampler_cond_mask_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-cond-mask");
            if (stage2_sampler_fixed_noise_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-fixed-noise");
            if (stage2_sampler_timesteps_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-timesteps");
            if (stage2_sampler_sampled_ref_path.empty()) throw std::runtime_error("--check-stage2-no-cfg-sampler requires --stage2-sampler-sampled-ref");
            const std::vector<int32_t> text_ids = read_i32_file(stage2_sampler_text_ids_path);
            const std::vector<int32_t> text_language_ids = read_i32_file(stage2_sampler_text_language_ids_path);
            const std::vector<float> no_lang_mask = read_f32_file(stage2_sampler_text_no_lang_mask_path);
            const std::vector<float> keep_mask = read_f32_file(stage2_sampler_text_keep_mask_path);
            const std::vector<float> pos_embed = read_f32_file(stage2_sampler_text_pos_embed_path);
            const std::vector<int32_t> time_language_ids = read_i32_file(stage2_sampler_time_language_ids_path);
            const std::vector<float> cond = read_f32_file(stage2_sampler_cond_path);
            const std::vector<float> cond_mask = read_f32_file(stage2_sampler_cond_mask_path);
            const std::vector<float> fixed_noise = read_f32_file(stage2_sampler_fixed_noise_path);
            const std::vector<float> timesteps = read_f32_file(stage2_sampler_timesteps_path);
            const x_voice::Stage2NoCfgSamplerResult actual = runtime.run_stage2_no_cfg_sampler(
                text_ids,
                text_language_ids,
                no_lang_mask,
                keep_mask,
                pos_embed,
                time_language_ids,
                cond,
                cond_mask,
                fixed_noise,
                timesteps,
                seq_len,
                step_count);
            const std::vector<float> sampled_expected = read_f32_file(stage2_sampler_sampled_ref_path);
            const Diff sampled_diff = compare_f32(actual.sampled.data, sampled_expected);
            const bool sampled_passed = threshold < 0.0f || sampled_diff.max_abs <= threshold;
            std::cout << "stage2_no_cfg_sampler_y" << step_count << ": " << (sampled_passed ? "ok" : "fail")
                      << " ne=(" << actual.sampled.ne[0] << ", " << actual.sampled.ne[1] << ")"
                      << " max_abs=" << sampled_diff.max_abs
                      << " mean_abs=" << sampled_diff.mean_abs << "\n";
            if (!sampled_passed) return 1;
            if (!stage2_sampler_out_ref_path.empty()) {
                const std::vector<float> out_expected = read_f32_file(stage2_sampler_out_ref_path);
                const Diff out_diff = compare_f32(actual.mel.data, out_expected);
                const bool out_passed = threshold < 0.0f || out_diff.max_abs <= threshold;
                std::cout << "stage2_no_cfg_sampler_out: " << (out_passed ? "ok" : "fail")
                          << " ne=(" << actual.mel.ne[0] << ", " << actual.mel.ne[1] << ")"
                          << " max_abs=" << out_diff.max_abs
                          << " mean_abs=" << out_diff.mean_abs << "\n";
                if (!out_passed) return 1;
            }
        }

        if (check_stage2_nonlayered_cfg_sampler) {
            if (stage2_sampler_text_ids_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-text-ids");
            if (stage2_sampler_text_language_ids_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-text-language-ids");
            if (stage2_sampler_text_no_lang_mask_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-text-no-lang-mask");
            if (stage2_sampler_text_keep_mask_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-text-keep-mask");
            if (stage2_sampler_text_pos_embed_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-text-pos-embed");
            if (stage2_sampler_time_language_ids_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-time-language-ids");
            if (stage2_sampler_cond_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-cond");
            if (stage2_sampler_cond_mask_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-cond-mask");
            if (stage2_sampler_fixed_noise_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-fixed-noise");
            if (stage2_sampler_timesteps_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-timesteps");
            if (stage2_sampler_sampled_ref_path.empty()) throw std::runtime_error("--check-stage2-nonlayered-cfg-sampler requires --stage2-sampler-sampled-ref");
            const std::vector<int32_t> text_ids = read_i32_file(stage2_sampler_text_ids_path);
            const std::vector<int32_t> text_language_ids = read_i32_file(stage2_sampler_text_language_ids_path);
            const std::vector<float> no_lang_mask = read_f32_file(stage2_sampler_text_no_lang_mask_path);
            const std::vector<float> keep_mask = read_f32_file(stage2_sampler_text_keep_mask_path);
            const std::vector<float> pos_embed = read_f32_file(stage2_sampler_text_pos_embed_path);
            const std::vector<int32_t> time_language_ids = read_i32_file(stage2_sampler_time_language_ids_path);
            const std::vector<float> cond = read_f32_file(stage2_sampler_cond_path);
            const std::vector<float> cond_mask = read_f32_file(stage2_sampler_cond_mask_path);
            const std::vector<float> fixed_noise = read_f32_file(stage2_sampler_fixed_noise_path);
            const std::vector<float> timesteps = read_f32_file(stage2_sampler_timesteps_path);
            const x_voice::Stage2NonLayeredCfgSamplerResult actual = runtime.run_stage2_nonlayered_cfg_sampler(
                text_ids,
                text_language_ids,
                no_lang_mask,
                keep_mask,
                pos_embed,
                time_language_ids,
                cond,
                cond_mask,
                fixed_noise,
                timesteps,
                seq_len,
                step_count);
            const std::vector<float> sampled_expected = read_f32_file(stage2_sampler_sampled_ref_path);
            const Diff sampled_diff = compare_f32(actual.sampled.data, sampled_expected);
            const bool sampled_passed = threshold < 0.0f || sampled_diff.max_abs <= threshold;
            std::cout << "stage2_nonlayered_cfg_sampler_y" << step_count << ": " << (sampled_passed ? "ok" : "fail")
                      << " ne=(" << actual.sampled.ne[0] << ", " << actual.sampled.ne[1] << ")"
                      << " max_abs=" << sampled_diff.max_abs
                      << " mean_abs=" << sampled_diff.mean_abs << "\n";
            if (!sampled_passed) return 1;
            if (!stage2_sampler_out_ref_path.empty()) {
                const std::vector<float> out_expected = read_f32_file(stage2_sampler_out_ref_path);
                const Diff out_diff = compare_f32(actual.mel.data, out_expected);
                const bool out_passed = threshold < 0.0f || out_diff.max_abs <= threshold;
                std::cout << "stage2_nonlayered_cfg_sampler_out: " << (out_passed ? "ok" : "fail")
                          << " ne=(" << actual.mel.ne[0] << ", " << actual.mel.ne[1] << ")"
                          << " max_abs=" << out_diff.max_abs
                          << " mean_abs=" << out_diff.mean_abs << "\n";
                if (!out_passed) return 1;
            }
        }

        if (check_stage2_layered_cfg_sampler) {
            if (stage2_sampler_text_ids_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-text-ids");
            if (stage2_sampler_text_language_ids_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-text-language-ids");
            if (stage2_sampler_text_no_lang_mask_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-text-no-lang-mask");
            if (stage2_sampler_text_keep_mask_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-text-keep-mask");
            if (stage2_sampler_text_pos_embed_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-text-pos-embed");
            if (stage2_sampler_time_language_ids_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-time-language-ids");
            if (stage2_sampler_cond_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-cond");
            if (stage2_sampler_cond_mask_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-cond-mask");
            if (stage2_sampler_fixed_noise_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-fixed-noise");
            if (stage2_sampler_timesteps_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-timesteps");
            if (stage2_sampler_sampled_ref_path.empty()) throw std::runtime_error("--check-stage2-layered-cfg-sampler requires --stage2-sampler-sampled-ref");
            const std::vector<int32_t> text_ids = read_i32_file(stage2_sampler_text_ids_path);
            const std::vector<int32_t> text_language_ids = read_i32_file(stage2_sampler_text_language_ids_path);
            const std::vector<float> no_lang_mask = read_f32_file(stage2_sampler_text_no_lang_mask_path);
            const std::vector<float> keep_mask = read_f32_file(stage2_sampler_text_keep_mask_path);
            const std::vector<float> pos_embed = read_f32_file(stage2_sampler_text_pos_embed_path);
            const std::vector<int32_t> time_language_ids = read_i32_file(stage2_sampler_time_language_ids_path);
            const std::vector<float> cond = read_f32_file(stage2_sampler_cond_path);
            const std::vector<float> cond_mask = read_f32_file(stage2_sampler_cond_mask_path);
            const std::vector<float> fixed_noise = read_f32_file(stage2_sampler_fixed_noise_path);
            const std::vector<float> timesteps = read_f32_file(stage2_sampler_timesteps_path);
            const x_voice::Stage2LayeredCfgSamplerResult actual = runtime.run_stage2_layered_cfg_sampler(
                text_ids,
                text_language_ids,
                no_lang_mask,
                keep_mask,
                pos_embed,
                time_language_ids,
                cond,
                cond_mask,
                fixed_noise,
                timesteps,
                seq_len,
                step_count);
            const std::vector<float> sampled_expected = read_f32_file(stage2_sampler_sampled_ref_path);
            const Diff sampled_diff = compare_f32(actual.sampled.data, sampled_expected);
            const bool sampled_passed = threshold < 0.0f || sampled_diff.max_abs <= threshold;
            std::cout << "stage2_layered_cfg_sampler_y" << step_count << ": " << (sampled_passed ? "ok" : "fail")
                      << " ne=(" << actual.sampled.ne[0] << ", " << actual.sampled.ne[1] << ")"
                      << " max_abs=" << sampled_diff.max_abs
                      << " mean_abs=" << sampled_diff.mean_abs << "\n";
            if (!sampled_passed) return 1;
            if (!stage2_sampler_out_ref_path.empty()) {
                const std::vector<float> out_expected = read_f32_file(stage2_sampler_out_ref_path);
                const Diff out_diff = compare_f32(actual.mel.data, out_expected);
                const bool out_passed = threshold < 0.0f || out_diff.max_abs <= threshold;
                std::cout << "stage2_layered_cfg_sampler_out: " << (out_passed ? "ok" : "fail")
                          << " ne=(" << actual.mel.ne[0] << ", " << actual.mel.ne[1] << ")"
                          << " max_abs=" << out_diff.max_abs
                          << " mean_abs=" << out_diff.mean_abs << "\n";
                if (!out_passed) return 1;
            }
        }

        if (check_srp_input || check_srp_logits) {
            if (srp_mel_path.empty()) throw std::runtime_error("--check-srp-* requires --srp-mel");
            const std::vector<float> mel = read_f32_file(srp_mel_path);
            if (check_srp_input) {
                if (srp_input_ref_path.empty()) throw std::runtime_error("--check-srp-input requires --srp-input-ref");
                const x_voice::FloatTensor actual = runtime.run_srp_input_embedding(mel, seq_len);
                const std::vector<float> expected = read_f32_file(srp_input_ref_path);
                const Diff diff = compare_f32(actual.data, expected);
                const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
                std::cout << "srp_input_embed: " << (passed ? "ok" : "fail")
                          << " ne=(" << actual.ne[0] << ", " << actual.ne[1] << ")"
                          << " max_abs=" << diff.max_abs
                          << " mean_abs=" << diff.mean_abs << "\n";
                if (!passed) return 1;
            }
            if (check_srp_logits) {
                if (srp_logits_ref_path.empty()) throw std::runtime_error("--check-srp-logits requires --srp-logits-ref");
                const x_voice::FloatTensor actual = runtime.run_srp_logits(mel, seq_len);
                const std::vector<float> expected = read_f32_file(srp_logits_ref_path);
                const Diff diff = compare_f32(actual.data, expected);
                const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
                std::cout << "srp_logits: " << (passed ? "ok" : "fail")
                          << " ne=(" << actual.ne[0] << ", " << actual.ne[1] << ")"
                          << " max_abs=" << diff.max_abs
                          << " mean_abs=" << diff.mean_abs
                          << " actual_class=" << argmax(actual.data)
                          << " ref_class=" << argmax(expected)
                          << " actual_speed=" << xvoice_srp_speed_value(argmax(actual.data))
                          << " ref_speed=" << xvoice_srp_speed_value(argmax(expected)) << "\n";
                if (!passed) return 1;
                if (argmax(actual.data) != argmax(expected)) return 1;
            }
        }

        if (check_vocos_head) {
            if (vocos_mel_path.empty()) throw std::runtime_error("--check-vocos-head requires --vocos-mel");
            if (vocos_head_logits_ref_path.empty()) throw std::runtime_error("--check-vocos-head requires --vocos-head-logits-ref");
            if (vocos_real_ref_path.empty()) throw std::runtime_error("--check-vocos-head requires --vocos-real-ref");
            if (vocos_imag_ref_path.empty()) throw std::runtime_error("--check-vocos-head requires --vocos-imag-ref");
            const std::vector<float> mel = read_f32_file(vocos_mel_path);
            const x_voice::VocosBackboneHeadResult actual = runtime.run_vocos_backbone_head(mel, seq_len);
            const std::vector<float> expected_logits = read_f32_file(vocos_head_logits_ref_path);
            const std::vector<float> expected_real = read_f32_file(vocos_real_ref_path);
            const std::vector<float> expected_imag = read_f32_file(vocos_imag_ref_path);
            const Diff logits_diff = compare_f32(actual.head_logits.data, expected_logits);
            const Diff real_diff = compare_f32(actual.real.data, expected_real);
            const Diff imag_diff = compare_f32(actual.imag.data, expected_imag);
            const bool logits_passed = threshold < 0.0f || logits_diff.max_abs <= threshold;
            const bool real_passed = threshold < 0.0f || real_diff.max_abs <= threshold;
            const bool imag_passed = threshold < 0.0f || imag_diff.max_abs <= threshold;
            std::cout << "vocos_head_logits: " << (logits_passed ? "ok" : "fail")
                      << " ne=(" << actual.head_logits.ne[0] << ", " << actual.head_logits.ne[1] << ")"
                      << " max_abs=" << logits_diff.max_abs
                      << " mean_abs=" << logits_diff.mean_abs << "\n";
            std::cout << "vocos_head_real: " << (real_passed ? "ok" : "fail")
                      << " ne=(" << actual.real.ne[0] << ", " << actual.real.ne[1] << ")"
                      << " max_abs=" << real_diff.max_abs
                      << " mean_abs=" << real_diff.mean_abs << "\n";
            std::cout << "vocos_head_imag: " << (imag_passed ? "ok" : "fail")
                      << " ne=(" << actual.imag.ne[0] << ", " << actual.imag.ne[1] << ")"
                      << " max_abs=" << imag_diff.max_abs
                      << " mean_abs=" << imag_diff.mean_abs << "\n";
            if (!logits_passed || !real_passed || !imag_passed) return 1;
        }

        if (check_vocos_waveform || synthesize_vocos_mel) {
            if (vocos_mel_path.empty()) throw std::runtime_error("--check-vocos-waveform/--synthesize-vocos-mel requires --vocos-mel");
            const std::vector<float> mel = read_f32_file(vocos_mel_path);
            const x_voice::VocosDecodeResult actual = runtime.run_vocos_decode(mel, seq_len);
            if (check_vocos_waveform) {
                if (vocos_waveform_ref_path.empty()) throw std::runtime_error("--check-vocos-waveform requires --vocos-waveform-ref");
                const std::vector<float> expected_waveform = read_f32_file(vocos_waveform_ref_path);
                const Diff diff = compare_f32(actual.waveform.data, expected_waveform);
                const bool passed = threshold < 0.0f || diff.max_abs <= threshold;
                std::cout << "vocos_waveform: " << (passed ? "ok" : "fail")
                          << " ne=(" << actual.waveform.ne[0] << ")"
                          << " max_abs=" << diff.max_abs
                          << " mean_abs=" << diff.mean_abs
                          << " sample_rate=" << actual.sample_rate
                          << " decoded_frames=" << actual.decoded_frames << "\n";
                if (!passed) return 1;
            }
            if (synthesize_vocos_mel) {
                if (output_wav_path.empty()) throw std::runtime_error("--synthesize-vocos-mel requires --output-wav");
                if (show_progress) progress.update({"write wav", 0, 1, output_wav_path});
                write_wav_mono_i16(output_wav_path, actual.waveform.data, actual.sample_rate);
                if (show_progress) progress.update({"write wav", 1, 1, output_wav_path});
                std::cout << "output_wav: " << output_wav_path
                          << " samples=" << actual.waveform.data.size()
                          << " sample_rate=" << actual.sample_rate << "\n";
            }
        }
    } catch (const std::exception & exc) {
        std::cerr << "error: " << exc.what() << "\n";
        usage(argv[0]);
        return 1;
    }

    return 0;
}
