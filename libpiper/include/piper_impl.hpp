#ifndef PIPER_IMPL_H_
#define PIPER_IMPL_H_

#include "json.hpp"
#include "uni_algo.h"

#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <stdint.h>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

typedef char32_t Phoneme;
typedef int64_t PhonemeId;
typedef int64_t SpeakerId;
typedef std::map<Phoneme, std::vector<PhonemeId>> PhonemeIdMap;

const PhonemeId ID_PAD = 0; // interleaved
const PhonemeId ID_BOS = 1; // beginning of sentence
const PhonemeId ID_EOS = 2; // end of sentence

const Phoneme PHONEME_PAD = U'_';
const Phoneme PHONEME_BOS = U'^';
const Phoneme PHONEME_EOS = U'$';
const Phoneme PHONEME_SEPARATOR = 0;

const float DEFAULT_LENGTH_SCALE = 1.0f;
const float DEFAULT_NOISE_SCALE = 0.667f;
const float DEFAULT_NOISE_W_SCALE = 0.8f;

const int DEFAULT_HOP_LENGTH = 256;

// onnx
Ort::Env ort_env{ORT_LOGGING_LEVEL_WARNING, "piper"};

// espeak
#define CLAUSE_INTONATION_FULL_STOP 0x00000000
#define CLAUSE_INTONATION_COMMA 0x00001000
#define CLAUSE_INTONATION_QUESTION 0x00002000
#define CLAUSE_INTONATION_EXCLAMATION 0x00003000

#define CLAUSE_TYPE_CLAUSE 0x00040000
#define CLAUSE_TYPE_SENTENCE 0x00080000

#define CLAUSE_PERIOD (40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COMMA (20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_QUESTION (40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_EXCLAMATION                                                     \
    (45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COLON (30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_SEMICOLON (30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)

struct piper_voice {
    // From config JSON file
    std::string model_path;
    std::string espeak_voice;
    int sample_rate = 0;
    int num_speakers;
    PhonemeIdMap phoneme_id_map;
    int hop_length = DEFAULT_HOP_LENGTH;
    // Default synthesis settings for the voice
    float synth_length_scale = DEFAULT_LENGTH_SCALE;
    float synth_noise_scale = DEFAULT_NOISE_SCALE;
    float synth_noise_w_scale = DEFAULT_NOISE_W_SCALE;
};

struct piper_synthesizer {
    piper_voice *voice = nullptr;

    // onnx
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions session_allocator;
    Ort::SessionOptions session_options;

    // synthesize state
    // Words without ids of their own share the time of the voiced word next to them
    struct WordGroup {
        int first_word;
        int last_word;
        int voiced_word;
    };
    struct PhonemeChunk {
        std::vector<PhonemeId> ids;
        std::vector<int> id_words;          // word of the text owning each id
        std::vector<WordGroup> word_groups; // groups whose audio ends in this chunk
    };
    std::queue<PhonemeChunk> phoneme_id_queue;
    std::vector<float> chunk_samples;
    std::vector<piper_word_alignment> chunk_words;
    std::vector<int> word_weights;           // letters and digits of each word, at least 1
    std::vector<int64_t> word_start_samples; // relative to the audio of the whole text, -1 before the word's first id
    std::vector<int64_t> word_end_samples;
    int64_t samples_done = 0;                // samples returned for the text so far
    float length_scale = DEFAULT_LENGTH_SCALE;
    float noise_scale = DEFAULT_NOISE_SCALE;
    float noise_w_scale = DEFAULT_NOISE_W_SCALE;
    SpeakerId speaker_id = 0;
};

// Get the first UTF-8 codepoint of a string
std::optional<Phoneme> get_codepoint(std::string_view s) {
    auto view = una::views::utf8(s);
    auto it = view.begin();

    if (it != view.end()) {
        return *it;
    }

    return std::nullopt;
}

#endif // PIPER_IMPL_H_
