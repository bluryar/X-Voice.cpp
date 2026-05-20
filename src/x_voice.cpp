#include "x_voice_internal.h"

namespace x_voice {

void validate_stage2_graph_contract() {
    throw std::runtime_error("Stage-2 GGML graph execution is not implemented in C++ v0");
}

void validate_srp_graph_contract() {
    throw std::runtime_error("SRP GGML graph execution is not implemented in C++ v0");
}

void validate_vocos_graph_contract() {
    throw std::runtime_error("Vocos GGML graph execution is not implemented in C++ v0");
}

} // namespace x_voice
