#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string input;
    std::string output;
    std::string type = "q8_0";
    std::string policy = "linear";
    bool dry_run = false;
};

struct Counts {
    int64_t copied = 0;
    int64_t converted = 0;
    int64_t quantized = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
};

using GgufPtr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using GgmlPtr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

bool has_suffix(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains(const std::string & value, const std::string & needle) {
    return value.find(needle) != std::string::npos;
}

enum ggml_type parse_ggml_type(const std::string & type) {
    if (type == "f16" || type == "F16") return GGML_TYPE_F16;
    if (type == "q8_0" || type == "Q8_0") return GGML_TYPE_Q8_0;
    if (type == "q6_k" || type == "Q6_K") return GGML_TYPE_Q6_K;
    if (type == "q4_k" || type == "Q4_K") return GGML_TYPE_Q4_K;
    throw std::runtime_error("unsupported quantization type: " + type + " (expected f16, q8_0, q6_k, or q4_k)");
}

std::string usage(const char * argv0) {
    return std::string("usage: ") + argv0 +
           " --input x-voice-f32.gguf --output x-voice-q8_0.gguf --type f16|q8_0|q6_k|q4_k "
           "[--policy linear|all-weight] [--dry-run]\n";
}

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (key == "--input" || key == "-i") {
            args.input = need_value(key.c_str());
        } else if (key == "--output" || key == "-o") {
            args.output = need_value(key.c_str());
        } else if (key == "--type" || key == "-t") {
            args.type = need_value(key.c_str());
        } else if (key == "--policy") {
            args.policy = need_value(key.c_str());
        } else if (key == "--dry-run") {
            args.dry_run = true;
        } else if (key == "--help" || key == "-h") {
            std::cout << usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.input.empty()) throw std::runtime_error("missing --input");
    if (args.output.empty() && !args.dry_run) throw std::runtime_error("missing --output");
    if (args.policy != "linear" && args.policy != "all-weight") {
        throw std::runtime_error("unsupported --policy: " + args.policy + " (expected linear or all-weight)");
    }
    return args;
}

int64_t nelements(const ggml_tensor * tensor) {
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

int64_t nrows(const ggml_tensor * tensor) {
    return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

bool is_large_weight(const ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_F32 &&
           tensor->ne[0] >= 32 &&
           nrows(tensor) >= 2 &&
           nelements(tensor) >= 4096;
}

bool is_linear_weight_name(const std::string & name) {
    if (!has_suffix(name, ".weight")) return false;
    if (contains(name, ".norm") || contains(name, "norm.")) return false;
    if (contains(name, ".bias")) return false;
    if (contains(name, ".emb.") || contains(name, "embed.weight")) return false;
    if (contains(name, ".pos.") || contains(name, ".conv.") || contains(name, ".dwconv.")) return false;
    if (contains(name, "backbone.embed.weight")) return false;

    return contains(name, ".attn.") ||
           contains(name, ".ff.") ||
           contains(name, ".pwconv") ||
           contains(name, ".proj.weight") ||
           contains(name, ".lang_ada.weight") ||
           contains(name, ".cls.") ||
           name == "stage2.out.weight" ||
           name == "vocos.head.out.weight";
}

bool should_convert(const std::string & name, const ggml_tensor * tensor, enum ggml_type target_type, const Args & args) {
    if (!is_large_weight(tensor)) return false;
    if (tensor->ne[0] % ggml_blck_size(target_type) != 0) return false;
    if (args.policy == "all-weight") {
        if (!has_suffix(name, ".weight")) return false;
        if (contains(name, ".norm") || contains(name, "norm.") || contains(name, ".bias")) return false;
        return true;
    }
    return is_linear_weight_name(name);
}

std::vector<uint8_t> convert_tensor(const ggml_tensor * src, enum ggml_type target_type) {
    if (src->type != GGML_TYPE_F32) throw std::runtime_error("only F32 source tensors can be converted");
    if (src->ne[0] % ggml_blck_size(target_type) != 0) throw std::runtime_error("ne0 is not divisible by target block size");

    const int64_t rows = nrows(src);
    const int64_t per_row = src->ne[0];
    const size_t out_size = ggml_row_size(target_type, per_row) * static_cast<size_t>(rows);
    std::vector<uint8_t> out(out_size);
    const size_t written = ggml_quantize_chunk(target_type, static_cast<const float *>(src->data), out.data(), 0, rows, per_row, nullptr);
    if (written != out_size) {
        throw std::runtime_error("ggml_quantize_chunk wrote unexpected byte count");
    }
    return out;
}

std::vector<uint8_t> copy_tensor_bytes(const ggml_tensor * src) {
    const size_t size = ggml_nbytes(src);
    std::vector<uint8_t> out(size);
    std::memcpy(out.data(), src->data, size);
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const enum ggml_type target_type = parse_ggml_type(args.type);

        ggml_context * in_ggml_raw = nullptr;
        gguf_init_params in_params = {
            /*.no_alloc =*/ false,
            /*.ctx      =*/ &in_ggml_raw,
        };
        GgufPtr in_gguf(gguf_init_from_file(args.input.c_str(), in_params), gguf_free);
        GgmlPtr in_ggml(in_ggml_raw, ggml_free);
        if (!in_gguf || !in_ggml) throw std::runtime_error("failed to read input GGUF: " + args.input);

        ggml_init_params out_ggml_params = {
            /*.mem_size   =*/ static_cast<size_t>(ggml_tensor_overhead() * (gguf_get_n_tensors(in_gguf.get()) + 1)),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        GgmlPtr out_ggml(ggml_init(out_ggml_params), ggml_free);
        if (!out_ggml) throw std::runtime_error("failed to create output ggml context");

        GgufPtr out_gguf(gguf_init_empty(), gguf_free);
        if (!out_gguf) throw std::runtime_error("failed to create output GGUF context");
        gguf_set_kv(out_gguf.get(), in_gguf.get());
        gguf_set_val_str(out_gguf.get(), "xvoice.quantization.type", ggml_type_name(target_type));
        gguf_set_val_str(out_gguf.get(), "xvoice.quantization.policy", args.policy.c_str());
        gguf_set_val_str(out_gguf.get(), "xvoice.quantization.tool", "xvoice-quantize-gguf");

        const int64_t n_tensors_total = gguf_get_n_tensors(in_gguf.get());
        std::vector<std::vector<uint8_t>> buffers;
        buffers.reserve(static_cast<size_t>(n_tensors_total));

        Counts counts;
        for (int64_t i = 0; i < n_tensors_total; ++i) {
            const std::string name = gguf_get_tensor_name(in_gguf.get(), i);
            const ggml_tensor * src = ggml_get_tensor(in_ggml.get(), name.c_str());
            if (!src || !src->data) throw std::runtime_error("missing input tensor payload: " + name);

            const bool convert = should_convert(name, src, target_type, args);
            const enum ggml_type out_type = convert ? target_type : src->type;
            std::vector<uint8_t> payload = convert ? convert_tensor(src, target_type) : copy_tensor_bytes(src);

            ggml_tensor * dst = ggml_new_tensor(out_ggml.get(), out_type, GGML_MAX_DIMS, src->ne);
            if (!dst) throw std::runtime_error("failed to create output tensor: " + name);
            ggml_set_name(dst, name.c_str());
            gguf_add_tensor(out_gguf.get(), dst);

            buffers.push_back(std::move(payload));
            gguf_set_tensor_data(out_gguf.get(), name.c_str(), buffers.back().data());

            counts.input_bytes += ggml_nbytes(src);
            counts.output_bytes += gguf_get_tensor_size(out_gguf.get(), i);
            if (convert) {
                if (ggml_is_quantized(target_type)) ++counts.quantized;
                else ++counts.converted;
            } else {
                ++counts.copied;
            }

            if ((i + 1) % 64 == 0 || i + 1 == n_tensors_total) {
                std::cerr << "\rquantize: " << (i + 1) << "/" << n_tensors_total
                          << " quantized=" << counts.quantized
                          << " converted=" << counts.converted
                          << " copied=" << counts.copied << std::flush;
            }
        }
        std::cerr << "\n";

        gguf_set_val_i64(out_gguf.get(), "xvoice.quantization.quantized_tensor_count", counts.quantized);
        gguf_set_val_i64(out_gguf.get(), "xvoice.quantization.converted_tensor_count", counts.converted);
        gguf_set_val_i64(out_gguf.get(), "xvoice.quantization.copied_tensor_count", counts.copied);

        std::cout << "input_bytes=" << counts.input_bytes
                  << " output_bytes=" << counts.output_bytes
                  << " target=" << ggml_type_name(target_type)
                  << " policy=" << args.policy
                  << " quantized=" << counts.quantized
                  << " converted=" << counts.converted
                  << " copied=" << counts.copied << "\n";

        if (!args.dry_run) {
            gguf_write_to_file(out_gguf.get(), args.output.c_str(), false);
            std::cout << "wrote " << args.output << "\n";
        }

        ggml_quantize_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n" << usage(argv[0]);
        return 1;
    }
}
