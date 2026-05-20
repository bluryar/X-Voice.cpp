#include "x_voice_internal.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace x_voice {
namespace {

constexpr float kPi = 3.14159265358979323846f;

uint16_t read_u16_le(std::istream & in) {
    unsigned char bytes[2]{};
    in.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!in) throw std::runtime_error("unexpected EOF while reading WAV u16");
    return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t read_u32_le(std::istream & in) {
    unsigned char bytes[4]{};
    in.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!in) throw std::runtime_error("unexpected EOF while reading WAV u32");
    return static_cast<uint32_t>(bytes[0] |
                                 (bytes[1] << 8) |
                                 (bytes[2] << 16) |
                                 (bytes[3] << 24));
}

std::string read_fourcc(std::istream & in) {
    char bytes[4]{};
    in.read(bytes, sizeof(bytes));
    if (!in) throw std::runtime_error("unexpected EOF while reading WAV fourcc");
    return std::string(bytes, bytes + 4);
}

struct WavData {
    std::vector<float> mono;
    int sample_rate = 0;
};

WavData load_wav_mono(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open WAV file: " + path);
    if (read_fourcc(in) != "RIFF") throw std::runtime_error("WAV is missing RIFF header: " + path);
    (void) read_u32_le(in);
    if (read_fourcc(in) != "WAVE") throw std::runtime_error("WAV is missing WAVE header: " + path);

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<unsigned char> pcm;

    while (in && (!audio_format || pcm.empty())) {
        std::string id;
        try {
            id = read_fourcc(in);
        } catch (...) {
            break;
        }
        const uint32_t size = read_u32_le(in);
        const std::streampos next = in.tellg() + static_cast<std::streamoff>(size + (size & 1u));
        if (id == "fmt ") {
            audio_format = read_u16_le(in);
            channels = read_u16_le(in);
            sample_rate = read_u32_le(in);
            (void) read_u32_le(in);
            (void) read_u16_le(in);
            bits_per_sample = read_u16_le(in);
        } else if (id == "data") {
            pcm.resize(size);
            in.read(reinterpret_cast<char *>(pcm.data()), static_cast<std::streamsize>(pcm.size()));
            if (!in) throw std::runtime_error("failed to read WAV data chunk: " + path);
        }
        in.seekg(next);
    }

    if (audio_format != 1) throw std::runtime_error("only PCM WAV is supported for reference audio");
    if (channels == 0) throw std::runtime_error("WAV channel count must be positive");
    if (sample_rate == 0) throw std::runtime_error("WAV sample rate must be positive");
    const int bytes_per_sample = static_cast<int>(bits_per_sample / 8);
    if (!(bits_per_sample == 8 || bits_per_sample == 16 || bits_per_sample == 24 || bits_per_sample == 32)) {
        throw std::runtime_error("unsupported PCM WAV bit depth: " + std::to_string(bits_per_sample));
    }
    const size_t frame_bytes = static_cast<size_t>(channels) * static_cast<size_t>(bytes_per_sample);
    if (pcm.empty() || pcm.size() % frame_bytes != 0) throw std::runtime_error("WAV data size does not match format");
    const size_t frames = pcm.size() / frame_bytes;
    std::vector<float> mono(frames, 0.0f);
    for (size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (uint16_t channel = 0; channel < channels; ++channel) {
            const unsigned char * ptr = pcm.data() + frame * frame_bytes + static_cast<size_t>(channel) * bytes_per_sample;
            int32_t value = 0;
            if (bits_per_sample == 8) {
                value = static_cast<int32_t>(ptr[0]) - 128;
                sum += static_cast<double>(value) / 128.0;
            } else if (bits_per_sample == 16) {
                value = static_cast<int16_t>(ptr[0] | (ptr[1] << 8));
                sum += static_cast<double>(value) / 32768.0;
            } else if (bits_per_sample == 24) {
                value = static_cast<int32_t>(ptr[0] | (ptr[1] << 8) | (ptr[2] << 16));
                if (value & 0x800000) value -= 0x1000000;
                sum += static_cast<double>(value) / 8388608.0;
            } else {
                value = static_cast<int32_t>(ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24));
                sum += static_cast<double>(value) / 2147483648.0;
            }
        }
        mono[frame] = static_cast<float>(sum / static_cast<double>(channels));
    }
    return {mono, static_cast<int>(sample_rate)};
}

std::vector<float> resample_linear(const std::vector<float> & waveform, int source_sample_rate, int target_sample_rate) {
    if (source_sample_rate <= 0 || target_sample_rate <= 0) throw std::runtime_error("sample rates must be positive");
    if (waveform.empty()) throw std::runtime_error("waveform must not be empty");
    if (source_sample_rate == target_sample_rate) return waveform;
    const size_t target_size = std::max<size_t>(
        static_cast<size_t>(std::llround(static_cast<double>(waveform.size()) *
                                         static_cast<double>(target_sample_rate) /
                                         static_cast<double>(source_sample_rate))),
        1);
    std::vector<float> out(target_size);
    const double scale = waveform.size() == 1 ? 0.0 : static_cast<double>(waveform.size() - 1) /
        static_cast<double>(std::max<size_t>(target_size - 1, 1));
    for (size_t i = 0; i < target_size; ++i) {
        const double x = static_cast<double>(i) * scale;
        const size_t left = static_cast<size_t>(std::floor(x));
        const size_t right = std::min(left + 1, waveform.size() - 1);
        const double frac = x - static_cast<double>(left);
        out[i] = static_cast<float>((1.0 - frac) * waveform[left] + frac * waveform[right]);
    }
    return out;
}

float rms_normalize_if_low(std::vector<float> & waveform, float target_rms) {
    if (waveform.empty()) throw std::runtime_error("waveform must not be empty");
    double sum = 0.0;
    for (float value : waveform) sum += static_cast<double>(value) * static_cast<double>(value);
    const float rms = static_cast<float>(std::sqrt(sum / static_cast<double>(waveform.size())));
    if (rms > 0.0f && rms < target_rms) {
        const float gain = target_rms / rms;
        for (float & value : waveform) value *= gain;
    }
    return rms;
}

int reflect_index(int index, int size) {
    if (size <= 1) return 0;
    while (index < 0 || index >= size) {
        if (index < 0) index = -index;
        if (index >= size) index = 2 * size - 2 - index;
    }
    return index;
}

std::vector<float> pad_center_reflect(const std::vector<float> & waveform, int n_fft) {
    const int pad = n_fft / 2;
    std::vector<float> out(waveform.size() + static_cast<size_t>(2 * pad));
    const int size = static_cast<int>(waveform.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = waveform[static_cast<size_t>(reflect_index(static_cast<int>(i) - pad, size))];
    }
    return out;
}

std::vector<float> periodic_hann(int length) {
    std::vector<float> out(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i) {
        out[static_cast<size_t>(i)] = 0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(length));
    }
    return out;
}

float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> mel_filterbank_htk(int sample_rate, int n_fft, int n_mels) {
    const int n_freqs = n_fft / 2 + 1;
    std::vector<float> out(static_cast<size_t>(n_mels) * static_cast<size_t>(n_freqs), 0.0f);
    const float m_min = hz_to_mel(0.0f);
    const float m_max = hz_to_mel(static_cast<float>(sample_rate) / 2.0f);
    std::vector<float> f_pts(static_cast<size_t>(n_mels + 2));
    for (int i = 0; i < n_mels + 2; ++i) {
        const float mel = m_min + (m_max - m_min) * static_cast<float>(i) / static_cast<float>(n_mels + 1);
        f_pts[static_cast<size_t>(i)] = mel_to_hz(mel);
    }
    for (int mel = 0; mel < n_mels; ++mel) {
        const float lower = f_pts[static_cast<size_t>(mel)];
        const float center = f_pts[static_cast<size_t>(mel + 1)];
        const float upper = f_pts[static_cast<size_t>(mel + 2)];
        for (int bin = 0; bin < n_freqs; ++bin) {
            const float freq = (static_cast<float>(sample_rate) / 2.0f) *
                static_cast<float>(bin) / static_cast<float>(n_freqs - 1);
            const float left = (freq - lower) / std::max(center - lower, 1e-12f);
            const float right = (upper - freq) / std::max(upper - center, 1e-12f);
            out[static_cast<size_t>(mel) * static_cast<size_t>(n_freqs) + static_cast<size_t>(bin)] =
                std::max(0.0f, std::min(left, right));
        }
    }
    return out;
}

void fft(std::vector<std::complex<double>> & values) {
    const size_t n = values.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = -2.0 * 3.14159265358979323846 / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                const std::complex<double> u = values[i + j];
                const std::complex<double> v = values[i + j + len / 2] * w;
                values[i + j] = u + v;
                values[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

} // namespace

ReferenceMelResult reference_mel_from_wav_impl(
        const XVoiceSpec & spec,
        const std::string & wav_path,
        float target_rms) {
    if (spec.audio.mel_spec_type != "vocos") {
        throw std::runtime_error("only vocos reference mel frontend is supported");
    }
    WavData wav = load_wav_mono(wav_path);
    std::vector<float> waveform = resample_linear(wav.mono, wav.sample_rate, spec.audio.sample_rate);
    const float rms = rms_normalize_if_low(waveform, target_rms);
    waveform = pad_center_reflect(waveform, spec.audio.n_fft);
    if (static_cast<int>(waveform.size()) < spec.audio.n_fft) {
        waveform.resize(static_cast<size_t>(spec.audio.n_fft), waveform.empty() ? 0.0f : waveform.back());
    }
    const int frames = 1 + std::max(
        (static_cast<int>(waveform.size()) - spec.audio.n_fft) / spec.audio.hop_length,
        0);
    std::vector<float> window = periodic_hann(spec.audio.win_length);
    if (spec.audio.win_length < spec.audio.n_fft) {
        const int left = (spec.audio.n_fft - spec.audio.win_length) / 2;
        std::vector<float> padded(static_cast<size_t>(spec.audio.n_fft), 0.0f);
        std::copy(window.begin(), window.end(), padded.begin() + left);
        window = std::move(padded);
    }
    const int bins = spec.audio.n_fft / 2 + 1;
    const std::vector<float> filters = mel_filterbank_htk(
        spec.audio.sample_rate,
        spec.audio.n_fft,
        spec.audio.mel_channel_count);
    std::vector<float> mel(static_cast<size_t>(frames) * static_cast<size_t>(spec.audio.mel_channel_count), 0.0f);
    std::vector<std::complex<double>> spectrum(static_cast<size_t>(spec.audio.n_fft));
    std::vector<float> mags(static_cast<size_t>(bins));
    for (int frame = 0; frame < frames; ++frame) {
        const int start = frame * spec.audio.hop_length;
        std::fill(spectrum.begin(), spectrum.end(), std::complex<double>(0.0, 0.0));
        for (int i = 0; i < spec.audio.n_fft; ++i) {
            spectrum[static_cast<size_t>(i)] = std::complex<double>(
                static_cast<double>(waveform[static_cast<size_t>(start + i)] * window[static_cast<size_t>(i)]),
                0.0);
        }
        fft(spectrum);
        for (int bin = 0; bin < bins; ++bin) {
            mags[static_cast<size_t>(bin)] = static_cast<float>(std::abs(spectrum[static_cast<size_t>(bin)]));
        }
        for (int m = 0; m < spec.audio.mel_channel_count; ++m) {
            double sum = 0.0;
            for (int bin = 0; bin < bins; ++bin) {
                sum += static_cast<double>(filters[static_cast<size_t>(m) * static_cast<size_t>(bins) + static_cast<size_t>(bin)]) *
                       static_cast<double>(mags[static_cast<size_t>(bin)]);
            }
            mel[static_cast<size_t>(frame) * static_cast<size_t>(spec.audio.mel_channel_count) + static_cast<size_t>(m)] =
                std::log(std::max(static_cast<float>(sum), 1e-5f));
        }
    }

    ReferenceMelResult result;
    result.mel.ne = {spec.audio.mel_channel_count, frames, 1, 1};
    result.mel.data = std::move(mel);
    result.source_sample_rate = wav.sample_rate;
    result.rms = rms;
    return result;
}

} // namespace x_voice
