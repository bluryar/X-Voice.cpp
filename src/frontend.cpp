#include "x_voice_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

#if defined(XVOICE_ENABLE_ZH_FRONTEND)
#include <cpp-pinyin/G2pglobal.h>
#include <cpp-pinyin/Pinyin.h>
#include <cppjieba/Jieba.hpp>
#endif

namespace x_voice {
namespace {

bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::vector<std::string> split_spaces(const std::string & value) {
    std::vector<std::string> out;
    std::string current;
    for (char c : value) {
        if (is_ascii_space(c)) {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) out.push_back(current);
    return out;
}

std::vector<std::string> split_char(const std::string & value, char sep) {
    std::vector<std::string> out;
    std::string current;
    for (char c : value) {
        if (c == sep) {
            out.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    out.push_back(current);
    return out;
}

size_t utf8_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

uint32_t utf8_codepoint(const std::string & s) {
    if (s.empty()) return 0;
    const unsigned char c0 = static_cast<unsigned char>(s[0]);
    if ((c0 & 0x80) == 0 || s.size() == 1) return c0;
    if ((c0 & 0xE0) == 0xC0 && s.size() >= 2) {
        return ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(s[1]) & 0x3F);
    }
    if ((c0 & 0xF0) == 0xE0 && s.size() >= 3) {
        return ((c0 & 0x0F) << 12) |
               ((static_cast<unsigned char>(s[1]) & 0x3F) << 6) |
               (static_cast<unsigned char>(s[2]) & 0x3F);
    }
    if ((c0 & 0xF8) == 0xF0 && s.size() >= 4) {
        return ((c0 & 0x07) << 18) |
               ((static_cast<unsigned char>(s[1]) & 0x3F) << 12) |
               ((static_cast<unsigned char>(s[2]) & 0x3F) << 6) |
               (static_cast<unsigned char>(s[3]) & 0x3F);
    }
    return c0;
}

std::vector<std::string> utf8_chars(const std::string & value) {
    std::vector<std::string> out;
    for (size_t i = 0; i < value.size();) {
        size_t n = utf8_len(static_cast<unsigned char>(value[i]));
        if (i + n > value.size()) n = 1;
        out.push_back(value.substr(i, n));
        i += n;
    }
    return out;
}

bool is_ipa_run_char(const std::string & ch) {
    if (ch.size() == 1) {
        const unsigned char c = static_cast<unsigned char>(ch[0]);
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }
    const uint32_t cp = utf8_codepoint(ch);
    return cp >= 0x00C0 && cp <= 0x02AF;
}

void replace_all(std::string & value, const std::string & from, const std::string & to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string normalize_ipa_text(std::string value) {
    static const std::vector<std::pair<std::string, std::string>> replacements = {
        {"eɪɛ", "eɪ|ɛ"},
        {"zeɪɛ", "z|eɪ|ɛ"},
        {"teɪɛ", "t|eɪ|ɛ"},
        {"əeɪ", "ə|eɪ"},
        {"aɪɛ", "aɪ|ɛ"},
        {"taɪ", "t|aɪ"},
        {"jap", "ja|p"},
        {"jud", "ju|d"},
        {"jaɛ", "ja|ɛ"},
        {"ʃja", "ʃ|ja"},
        {"jat", "ja|t"},
        {"ɑja", "ɑ|ja"},
        {"əlɹ", "əl|ɹ"},
        {"əlf", "əl|f"},
        {"oʊw", "oʊ|w"},
        {"daʊ", "d|aʊ"},
        {"meɪ", "m|eɪ"},
        {"taʊ", "t|aʊ"},
        {"daɪ", "d|aɪ"},
    };
    static const std::vector<std::string> garbage = {"&", "ㄜ", "で", "か", "す", "π", "%", "@"};
    for (const auto & g : garbage) replace_all(value, g, "");
    for (const auto & item : replacements) replace_all(value, item.first, item.second);
    return value;
}

#if defined(XVOICE_ENABLE_ZH_FRONTEND)
std::mutex & cpp_pinyin_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unique_ptr<Pinyin::Pinyin> & cpp_pinyin_engine() {
    static std::unique_ptr<Pinyin::Pinyin> engine;
    return engine;
}

std::mutex & cppjieba_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unique_ptr<cppjieba::Jieba> & cppjieba_engine() {
    static std::unique_ptr<cppjieba::Jieba> engine;
    return engine;
}

std::string default_cpp_pinyin_dict() {
#if defined(XVOICE_DEFAULT_CPP_PINYIN_DICT)
    return XVOICE_DEFAULT_CPP_PINYIN_DICT;
#else
    return {};
#endif
}

std::string default_cppjieba_dict() {
#if defined(XVOICE_DEFAULT_CPPJIEBA_DICT)
    return XVOICE_DEFAULT_CPPJIEBA_DICT;
#else
    return {};
#endif
}

std::string default_zh_override_path() {
#if defined(XVOICE_DEFAULT_ZH_OVERRIDE_PATH)
    return XVOICE_DEFAULT_ZH_OVERRIDE_PATH;
#else
    return {};
#endif
}

std::string resolve_path(const std::string & value, const std::string & fallback) {
    return value.empty() ? fallback : value;
}

void require_cpp_pinyin_dict(const std::filesystem::path & dir) {
    const std::filesystem::path mandarin = dir / "mandarin";
    static const char * const required[] = {
        "phrases_map.txt",
        "phrases_dict.txt",
        "user_dict.txt",
        "word.txt",
        "trans_word.txt",
    };
    for (const char * name : required) {
        if (!std::filesystem::exists(mandarin / name)) {
            throw std::runtime_error("X-Voice zh frontend cpp-pinyin dictionary is incomplete: " + dir.string());
        }
    }
}

void ensure_zh_frontend_configured(const std::string & pinyin_dict_path, const std::string & jieba_dict_path) {
    static std::string configured_pinyin_dir;
    static std::string configured_jieba_dir;
    const std::string pinyin_dir = resolve_path(pinyin_dict_path, default_cpp_pinyin_dict());
    const std::string jieba_dir = resolve_path(jieba_dict_path, default_cppjieba_dict());
    if (pinyin_dir.empty() || jieba_dir.empty()) {
        throw std::runtime_error("X-Voice zh frontend dictionary defaults are not configured in this build");
    }
    if (configured_pinyin_dir == pinyin_dir && configured_jieba_dir == jieba_dir) return;
    {
        std::lock_guard<std::mutex> lock(cpp_pinyin_mutex());
        if (configured_pinyin_dir != pinyin_dir) {
            require_cpp_pinyin_dict(std::filesystem::path(pinyin_dir));
            Pinyin::setDictionaryPath(std::filesystem::path(pinyin_dir));
            cpp_pinyin_engine() = std::make_unique<Pinyin::Pinyin>();
            configured_pinyin_dir = pinyin_dir;
        }
    }
    {
        const std::filesystem::path dir(jieba_dir);
        const std::filesystem::path dict = dir / "jieba.dict.utf8";
        const std::filesystem::path hmm = dir / "hmm_model.utf8";
        const std::filesystem::path user = dir / "user.dict.utf8";
        const std::filesystem::path idf = dir / "idf.utf8";
        const std::filesystem::path stop = dir / "stop_words.utf8";
        if (!std::filesystem::exists(dict) || !std::filesystem::exists(hmm) ||
            !std::filesystem::exists(user) || !std::filesystem::exists(idf) ||
            !std::filesystem::exists(stop)) {
            throw std::runtime_error("X-Voice zh frontend cppjieba dictionary is incomplete: " + dir.string());
        }
        std::lock_guard<std::mutex> lock(cppjieba_mutex());
        if (configured_jieba_dir != jieba_dir) {
            cppjieba_engine() = std::make_unique<cppjieba::Jieba>(
                dict.string(),
                hmm.string(),
                user.string(),
                idf.string(),
                stop.string());
            configured_jieba_dir = jieba_dir;
        }
    }
}

bool is_ascii_token(const std::string & ch) {
    return ch.size() == 1 && (static_cast<unsigned char>(ch[0]) & 0x80) == 0;
}

bool is_zh_punctuation(const std::string & ch) {
    static const std::vector<std::string> punct = {
        "。", "，", "、", "；", "：", "？", "！", "（", "）", "《", "》", "“", "”", "‘", "’", "—", "…"
    };
    return std::find(punct.begin(), punct.end(), ch) != punct.end();
}

std::string normalize_plain_zh_punctuation(const std::string & ch) {
    if (ch == "。") return ".";
    if (ch == "，" || ch == "、" || ch == "；") return ",";
    if (ch == "：") return ":";
    if (ch == "？") return "?";
    if (ch == "！") return "!";
    if (ch == "（" || ch == "《" || ch == "“" || ch == "‘") return "(";
    if (ch == "）" || ch == "》" || ch == "”" || ch == "’") return ")";
    if (ch == "—") return "-";
    if (ch == "…") return ".";
    return ch;
}

struct PinyinItem {
    std::string hanzi;
    std::string pinyin;
};

int tone_number(const std::string & pinyin) {
    for (char c : pinyin) {
        if (c >= '1' && c <= '5') return c - '0';
    }
    return 0;
}

void set_tone(std::string & pinyin, char tone) {
    for (char & c : pinyin) {
        if (c >= '1' && c <= '5') {
            c = tone;
            return;
        }
    }
    pinyin.push_back(tone);
}

void apply_basic_tone_sandhi(std::vector<PinyinItem> & items) {
    for (size_t i = 0; i < items.size();) {
        if (tone_number(items[i].pinyin) != 3) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < items.size() && tone_number(items[j].pinyin) == 3) ++j;
        if (j - i >= 2) {
            for (size_t k = i; k + 1 < j; ++k) set_tone(items[k].pinyin, '2');
        }
        i = j;
    }
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].hanzi == "不") {
            if (i + 1 < items.size() && tone_number(items[i + 1].pinyin) == 4) set_tone(items[i].pinyin, '2');
            else set_tone(items[i].pinyin, '4');
        }
        if (items[i].hanzi == "一") {
            if (i + 1 < items.size() && tone_number(items[i + 1].pinyin) == 4) set_tone(items[i].pinyin, '2');
            else if (i + 1 < items.size()) set_tone(items[i].pinyin, '4');
            else set_tone(items[i].pinyin, '1');
        }
    }
}

std::unordered_map<std::string, std::string> built_in_zh_overrides() {
    return {
        {"重庆", "chong2|qing4"},
        {"重庆银行", "chong2|qing4|yin2|hang2"},
        {"银行", "yin2|hang2"},
        {"银行行长", "yin2|hang2|hang2|zhang3"},
        {"行长", "hang2|zhang3"},
        {"重新", "chong2|xin1"},
        {"重复", "chong2|fu4"},
        {"重要", "zhong4|yao4"},
        {"重量", "zhong4|liang4"},
        {"音乐", "yin1|yue4"},
        {"乐队", "yue4|dui4"},
        {"快乐", "kuai4|le4"},
        {"为了", "wei4|le"},
        {"作为", "zuo4|wei2"},
        {"说书人", "shuo1|shu1|ren2"},
        {"发帖人", "fa1|tie1|ren2"},
        {"弟弟", "di4|di"},
        {"五口之家", "wu3|kou3|zhi1|jia1"},
        {"详细", "xiang2|xi4"},
        {"描述", "miao2|shu4"},
        {"现状", "xian4|zhuang4"},
        {"寻求", "xun2|qiu2"},
        {"处理", "chu3|li3"},
        {"家庭", "jia1|ting2"},
        {"问题", "wen4|ti2"},
        {"建议", "jian4|yi4"},
        {"提出", "ti2|chu1"},
        {"初步", "chu1|bu4"},
        {"计划", "ji4|hua4"},
    };
}

std::string trim_ascii(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

void merge_zh_override_file(std::unordered_map<std::string, std::string> & overrides, const std::string & path) {
    if (path.empty()) return;
    if (!std::filesystem::exists(path)) return;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open X-Voice zh override file: " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        line = trim_ascii(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string word = trim_ascii(line.substr(0, tab));
        std::string ipa = trim_ascii(line.substr(tab + 1));
        if (!word.empty() && !ipa.empty()) overrides[word] = ipa;
    }
}

std::unordered_map<std::string, std::string> load_zh_overrides(const std::string & override_path) {
    std::unordered_map<std::string, std::string> overrides = built_in_zh_overrides();
    merge_zh_override_file(overrides, default_zh_override_path());
    merge_zh_override_file(overrides, override_path);
    return overrides;
}

std::string override_word_ipa(const std::string & word, const std::unordered_map<std::string, std::string> & overrides) {
    auto it = overrides.find(word);
    return it == overrides.end() ? std::string() : it->second;
}

std::string word_to_pinyin_ipa(const std::string & word, const std::unordered_map<std::string, std::string> & overrides) {
    if (word.empty()) return {};
    const std::string override = override_word_ipa(word, overrides);
    if (!override.empty()) return override;

    std::vector<PinyinItem> items;
    {
        std::lock_guard<std::mutex> lock(cpp_pinyin_mutex());
        if (!cpp_pinyin_engine()) throw std::runtime_error("X-Voice zh frontend cpp-pinyin engine is not configured");
        Pinyin::PinyinResVector res = cpp_pinyin_engine()
            ->hanziToPinyin(word, Pinyin::ManTone::Style::TONE3, Pinyin::Error::Default, true, false, false);
        items.reserve(res.size());
        for (const Pinyin::PinyinRes & item : res) {
            if (item.error) {
                items.push_back({item.hanzi, item.hanzi});
            } else {
                std::string pinyin = item.pinyin;
                if (pinyin == "shi") pinyin = "shi4";
                items.push_back({item.hanzi, pinyin});
            }
        }
    }
    apply_basic_tone_sandhi(items);

    std::string out;
    for (const PinyinItem & item : items) {
        std::string token = normalize_plain_zh_punctuation(item.pinyin);
        if (token.empty()) continue;
        if (is_zh_punctuation(item.hanzi) || is_ascii_token(token)) {
            if (!out.empty()) out += "|";
            out += token;
        } else {
            if (!out.empty()) out += "|";
            out += token;
        }
    }
    return out;
}

std::vector<std::string> sorted_override_words(const std::unordered_map<std::string, std::string> & overrides) {
    std::vector<std::string> words;
    words.reserve(overrides.size());
    for (const auto & item : overrides) words.push_back(item.first);
    std::sort(words.begin(), words.end(), [](const std::string & a, const std::string & b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return a < b;
    });
    return words;
}

bool match_override_word(
        const std::string & text,
        size_t pos,
        const std::vector<std::string> & override_words,
        std::string & matched) {
    for (const std::string & word : override_words) {
        if (!word.empty() && pos + word.size() <= text.size() &&
            text.compare(pos, word.size(), word) == 0) {
            matched = word;
            return true;
        }
    }
    return false;
}

void cut_with_jieba(const std::string & text, std::vector<std::string> & words) {
    if (text.empty()) return;
    std::lock_guard<std::mutex> lock(cppjieba_mutex());
    if (!cppjieba_engine()) throw std::runtime_error("X-Voice zh frontend cppjieba engine is not configured");
    std::vector<std::string> chunk;
    cppjieba_engine()->Cut(text, chunk, true);
    words.insert(words.end(), chunk.begin(), chunk.end());
}

std::vector<std::string> cut_zh_text_with_overrides(
        const std::string & text,
        const std::unordered_map<std::string, std::string> & overrides) {
    const std::vector<std::string> override_words = sorted_override_words(overrides);
    std::vector<std::string> words;
    std::string pending;
    for (size_t i = 0; i < text.size();) {
        std::string matched;
        if (match_override_word(text, i, override_words, matched)) {
            cut_with_jieba(pending, words);
            pending.clear();
            words.push_back(matched);
            i += matched.size();
            continue;
        }
        size_t n = utf8_len(static_cast<unsigned char>(text[i]));
        if (i + n > text.size()) n = 1;
        pending.append(text, i, n);
        i += n;
    }
    cut_with_jieba(pending, words);
    return words;
}

std::string plain_zh_to_ipa_v6_impl(
        const std::string & text,
        const std::string & pinyin_dict_path,
        const std::string & jieba_dict_path,
        const std::string & override_path) {
    ensure_zh_frontend_configured(pinyin_dict_path, jieba_dict_path);
    const std::unordered_map<std::string, std::string> overrides = load_zh_overrides(override_path);
    const std::vector<std::string> words = cut_zh_text_with_overrides(text, overrides);
    std::vector<std::string> pieces;
    for (const std::string & word : words) {
        const std::string converted = word_to_pinyin_ipa(word, overrides);
        if (!converted.empty()) pieces.push_back(converted);
    }
    std::ostringstream out;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i) out << ' ';
        out << pieces[i];
    }
    return out.str();
}
#endif

} // namespace

std::string normalize_language_code(const std::string & language) {
    std::string value = language;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "eng" || value == "en-us" || value == "en_us") return "en";
    if (value == "cmn" || value == "zh-cn" || value == "zh_cn" || value == "zho") return "zh";
    if (value == "pt-br" || value == "pt_br") return "pt";
    return value;
}

int language_id_for_code(const XVoiceSpec & spec, const std::string & language) {
    const std::string normalized = normalize_language_code(language);
    for (size_t i = 0; i < spec.languages.size(); ++i) {
        if (spec.languages[i] == normalized) return static_cast<int>(i);
    }
    throw std::runtime_error("language is not in xvoice.languages metadata: " + normalized);
}

XVoiceTokenizer::XVoiceTokenizer(
        std::vector<std::string> tokens,
        std::string pinyin_dict_path,
        std::string jieba_dict_path,
        std::string zh_override_path)
    : tokens_(std::move(tokens)),
      pinyin_dict_path_(std::move(pinyin_dict_path)),
      jieba_dict_path_(std::move(jieba_dict_path)),
      zh_override_path_(std::move(zh_override_path)) {
    for (size_t i = 0; i < tokens_.size(); ++i) token_to_id_[tokens_[i]] = static_cast<int32_t>(i);
    auto it = token_to_id_.find("<pad>");
    pad_id_ = it == token_to_id_.end() ? 0 : it->second;
}

std::vector<std::string> XVoiceTokenizer::split_ipa_v6(const std::string & ipa_text) const {
    const std::string normalized = normalize_ipa_text(ipa_text);
    const std::vector<std::string> words = split_spaces(normalized);
    std::vector<std::string> fields;
    for (size_t word_index = 0; word_index < words.size(); ++word_index) {
        const std::vector<std::string> pieces = split_char(words[word_index], '|');
        for (const std::string & piece : pieces) {
            if (piece.empty()) continue;
            std::string run;
            for (const std::string & ch : utf8_chars(piece)) {
                if (is_ipa_run_char(ch)) {
                    run += ch;
                } else {
                    if (!run.empty()) {
                        fields.push_back(run);
                        run.clear();
                    }
                    fields.push_back(ch);
                }
            }
            if (!run.empty()) fields.push_back(run);
        }
        if (word_index + 1 < words.size()) fields.emplace_back(" ");
    }
    return fields;
}

std::vector<std::string> XVoiceTokenizer::split_token_text(const std::string & token_text) const {
    return split_spaces(token_text);
}

std::string XVoiceTokenizer::plain_text_to_ipa_v6(const std::string & text, const std::string & language) const {
    if (normalize_language_code(language) != "zh") {
        throw std::runtime_error("text-kind plain is currently only supported for language=zh in x-voice-ggml-cpp");
    }
#if defined(XVOICE_ENABLE_ZH_FRONTEND)
    return plain_zh_to_ipa_v6_impl(text, pinyin_dict_path_, jieba_dict_path_, zh_override_path_);
#else
    (void) text;
    throw std::runtime_error(
        "text-kind plain requires rebuilding x-voice-ggml-cpp with -DXVOICE_ENABLE_ZH_FRONTEND=AUTO or ON; "
        "C++ v0 default input remains ipa or tokens");
#endif
}

std::vector<int32_t> XVoiceTokenizer::tokens_to_ids(const std::vector<std::string> & tokens) const {
    std::vector<int32_t> ids;
    for (const std::string & token : tokens) {
        auto it = token_to_id_.find(token);
        const int32_t idx = it == token_to_id_.end() ? pad_id_ : it->second;
        if (idx == pad_id_ && token != "…" && token != "<pad>") {
            for (const std::string & ch : utf8_chars(token)) {
                auto cit = token_to_id_.find(ch);
                ids.push_back(cit == token_to_id_.end() ? pad_id_ : cit->second);
            }
        } else {
            ids.push_back(idx);
        }
    }
    if (ids.empty()) throw std::runtime_error("X-Voice text produced no tokens");
    return ids;
}

const char * text_kind_name(TextKind kind) {
    switch (kind) {
        case TextKind::Ipa: return "ipa";
        case TextKind::Tokens: return "tokens";
        case TextKind::Plain: return "plain";
    }
    return "unknown";
}

TextKind parse_text_kind(const std::string & value) {
    if (value == "ipa") return TextKind::Ipa;
    if (value == "tokens") return TextKind::Tokens;
    if (value == "plain") return TextKind::Plain;
    throw std::runtime_error("unsupported text-kind: " + value);
}

} // namespace x_voice
