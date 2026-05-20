#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Stats {
    int64_t python_phrase_entries = 0;
    int64_t python_word_entries = 0;
    int64_t generated_phrase_entries = 0;
    int64_t generated_word_entries = 0;
    int64_t generated_phrase_map_entries = 0;
    int64_t skipped_phrase_entries = 0;
    int64_t skipped_word_entries = 0;
    int64_t phrase_slots_with_multiple_candidates = 0;
    int64_t word_entries_with_multiple_candidates = 0;
};

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

std::vector<std::string> split(const std::string & value, char sep) {
    std::vector<std::string> out;
    std::string current;
    std::istringstream in(value);
    while (std::getline(in, current, sep)) {
        current = trim(current);
        if (!current.empty()) out.push_back(current);
    }
    return out;
}

size_t utf8_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

uint32_t decode_utf8_at(const std::string & value, size_t pos, size_t & n) {
    n = utf8_len(static_cast<unsigned char>(value[pos]));
    if (pos + n > value.size()) n = 1;
    const unsigned char c0 = static_cast<unsigned char>(value[pos]);
    if (n == 1) return c0;
    if (n == 2) {
        return ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(value[pos + 1]) & 0x3F);
    }
    if (n == 3) {
        return ((c0 & 0x0F) << 12) |
               ((static_cast<unsigned char>(value[pos + 1]) & 0x3F) << 6) |
               (static_cast<unsigned char>(value[pos + 2]) & 0x3F);
    }
    return ((c0 & 0x07) << 18) |
           ((static_cast<unsigned char>(value[pos + 1]) & 0x3F) << 12) |
           ((static_cast<unsigned char>(value[pos + 2]) & 0x3F) << 6) |
           (static_cast<unsigned char>(value[pos + 3]) & 0x3F);
}

std::string codepoint_to_utf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

std::vector<std::string> utf8_bmp_chars(const std::string & value, bool & all_bmp) {
    std::vector<std::string> out;
    all_bmp = true;
    for (size_t i = 0; i < value.size();) {
        size_t n = 1;
        const uint32_t cp = decode_utf8_at(value, i, n);
        if (cp > 0xFFFF) all_bmp = false;
        out.push_back(value.substr(i, n));
        i += n;
    }
    return out;
}

json read_json(const fs::path & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open JSON: " + path.string());
    json value;
    in >> value;
    return value;
}

std::string join(const std::vector<std::string> & values, const std::string & sep) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << sep;
        out << values[i];
    }
    return out.str();
}

void ensure_parent(const fs::path & path) {
    fs::create_directories(path.parent_path());
}

void write_text_file(const fs::path & path, const std::string & text) {
    ensure_parent(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to write: " + path.string());
    out << text;
}

void copy_if_exists(const fs::path & src, const fs::path & dst) {
    if (!fs::exists(src)) return;
    ensure_parent(dst);
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
}

struct GeneratedDict {
    std::map<std::string, std::vector<std::string>> phrases;
    std::map<std::string, std::string> words;
    std::set<std::string> phrase_map_chars;
    Stats stats;
};

GeneratedDict load_python_pinyin_dict(const fs::path & python_pinyin_root) {
    const fs::path pypinyin_dir = python_pinyin_root / "pypinyin";
    const json phrases_json = read_json(pypinyin_dir / "phrases_dict.json");
    const json words_json = read_json(pypinyin_dir / "pinyin_dict.json");

    GeneratedDict out;

    for (const auto & item : phrases_json.items()) {
        ++out.stats.python_phrase_entries;
        const std::string word = item.key();
        bool all_bmp = true;
        const std::vector<std::string> chars = utf8_bmp_chars(word, all_bmp);
        if (!all_bmp || !item.value().is_array()) {
            ++out.stats.skipped_phrase_entries;
            continue;
        }

        std::vector<std::string> reading;
        for (const json & slot : item.value()) {
            if (!slot.is_array() || slot.empty() || !slot[0].is_string()) continue;
            if (slot.size() > 1) ++out.stats.phrase_slots_with_multiple_candidates;
            reading.push_back(slot[0].get<std::string>());
        }
        if (reading.empty() || reading.size() != chars.size()) {
            ++out.stats.skipped_phrase_entries;
            continue;
        }
        out.phrases[word] = std::move(reading);
        for (const std::string & ch : chars) out.phrase_map_chars.insert(ch);
        ++out.stats.generated_phrase_entries;
    }

    for (const auto & item : words_json.items()) {
        ++out.stats.python_word_entries;
        uint32_t cp = 0;
        try {
            cp = static_cast<uint32_t>(std::stoul(item.key()));
        } catch (const std::exception &) {
            ++out.stats.skipped_word_entries;
            continue;
        }
        if (cp > 0xFFFF || !item.value().is_string()) {
            ++out.stats.skipped_word_entries;
            continue;
        }
        const std::string reading = trim(item.value().get<std::string>());
        if (reading.empty()) {
            ++out.stats.skipped_word_entries;
            continue;
        }
        const std::string word = codepoint_to_utf8(cp);
        out.words[word] = reading;
        if (split(reading, ',').size() > 1) {
            out.phrase_map_chars.insert(word);
            ++out.stats.word_entries_with_multiple_candidates;
        }
        ++out.stats.generated_word_entries;
    }

    out.stats.generated_phrase_map_entries = static_cast<int64_t>(out.phrase_map_chars.size());
    return out;
}

void write_cpp_pinyin_dict(
        const fs::path & out_root,
        const fs::path & cpp_pinyin_dict_root,
        const fs::path & python_pinyin_root,
        const GeneratedDict & dict) {
    const fs::path mandarin = out_root / "mandarin";
    fs::remove_all(out_root);
    fs::create_directories(mandarin);

    {
        std::ofstream out(mandarin / "phrases_dict.txt");
        if (!out) throw std::runtime_error("failed to write phrases_dict.txt");
        for (const auto & item : dict.phrases) out << item.first << ":" << join(item.second, ",") << "\n";
    }
    {
        std::ofstream out(mandarin / "word.txt");
        if (!out) throw std::runtime_error("failed to write word.txt");
        for (const auto & item : dict.words) out << item.first << ":" << item.second << "\n";
    }
    {
        std::ofstream out(mandarin / "phrases_map.txt");
        if (!out) throw std::runtime_error("failed to write phrases_map.txt");
        for (const std::string & ch : dict.phrase_map_chars) out << ch << ":234\n";
    }

    write_text_file(
        mandarin / "user_dict.txt",
        "# Runtime dictionary is generated from python-pinyin. Project-specific\n"
        "# narration overrides live in resources/zh_frontend_overrides.tsv.\n");

    copy_if_exists(cpp_pinyin_dict_root / "mandarin" / "trans_word.txt", mandarin / "trans_word.txt");
    copy_if_exists(cpp_pinyin_dict_root / "mandarin" / "License.txt", mandarin / "License.txt");

    const Stats & stats = dict.stats;
    json manifest = {
        {"schema_version", 2},
        {"output", out_root.string()},
        {"python_pinyin", python_pinyin_root.string()},
        {"python_pinyin_phrases_json", (python_pinyin_root / "pypinyin" / "phrases_dict.json").string()},
        {"python_pinyin_words_json", (python_pinyin_root / "pypinyin" / "pinyin_dict.json").string()},
        {"cpp_pinyin_support_dict", cpp_pinyin_dict_root.string()},
        {"policy", "Runtime pinyin readings are generated from python-pinyin dictionaries; cpp-pinyin source files only provide trans_word.txt and License.txt."},
        {"generated", {
            {"phrase_entries", stats.generated_phrase_entries},
            {"word_entries", stats.generated_word_entries},
            {"phrase_map_entries", stats.generated_phrase_map_entries},
        }},
        {"stats", {
            {"python_phrase_entries", stats.python_phrase_entries},
            {"python_word_entries", stats.python_word_entries},
            {"skipped_phrase_entries", stats.skipped_phrase_entries},
            {"skipped_word_entries", stats.skipped_word_entries},
            {"phrase_slots_with_multiple_candidates", stats.phrase_slots_with_multiple_candidates},
            {"word_entries_with_multiple_candidates", stats.word_entries_with_multiple_candidates},
        }},
    };
    write_text_file(out_root / "manifest.json", manifest.dump(2) + "\n");
}

void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0
        << " --python-pinyin DIR --cpp-pinyin-dict DIR --out DIR\n";
}

} // namespace

int main(int argc, char ** argv) {
    fs::path python_pinyin_root;
    fs::path cpp_pinyin_dict_root;
    fs::path out_root;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const std::string & flag) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
                return argv[++i];
            };
            if (arg == "--python-pinyin") python_pinyin_root = require_value(arg);
            else if (arg == "--cpp-pinyin-dict") cpp_pinyin_dict_root = require_value(arg);
            else if (arg == "--out") out_root = require_value(arg);
            else if (arg == "--pypinyin-dict") {
                (void) require_value(arg);
                throw std::runtime_error("--pypinyin-dict is obsolete; use --python-pinyin");
            } else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        if (python_pinyin_root.empty() || cpp_pinyin_dict_root.empty() || out_root.empty()) {
            usage(argv[0]);
            return 2;
        }

        python_pinyin_root = fs::absolute(python_pinyin_root);
        cpp_pinyin_dict_root = fs::absolute(cpp_pinyin_dict_root);
        out_root = fs::absolute(out_root);

        GeneratedDict dict = load_python_pinyin_dict(python_pinyin_root);
        write_cpp_pinyin_dict(out_root, cpp_pinyin_dict_root, python_pinyin_root, dict);

        std::cout << "merged_pinyin_dict: ok"
                  << " policy=python-pinyin"
                  << " phrases=" << dict.stats.generated_phrase_entries
                  << " words=" << dict.stats.generated_word_entries
                  << " phrase_map=" << dict.stats.generated_phrase_map_entries
                  << " out=" << out_root << "\n";
    } catch (const std::exception & exc) {
        std::cerr << "merge_pinyin_dicts: error: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
