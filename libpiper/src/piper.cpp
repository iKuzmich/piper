#include "piper.h"
#include "piper_impl.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <mutex>

#include <espeak-ng/espeak_ng.h>

#ifdef __linux__
#include <malloc.h>
#endif

using json = nlohmann::json;

// A clause as espeak phonemized it, with the text word of each space-separated phoneme group
struct ClausePhonemes
{
    std::string phonemes;
    std::vector<int> group_words;
    std::string terminator;
};

static std::atomic<bool> g_espeak_ready{false};
static std::mutex g_espeak_init_mutex;
static std::string g_espeak_voice;

static void log_exception(const char *func, const char *model_path, const char *what);
static int samples_to_ms(int64_t samples, int sample_rate);
static int synthesize_start_impl(struct piper_synthesizer *synth, const char *text, const piper_synthesize_options *options);
static int synthesize_next_impl(struct piper_synthesizer *synth, struct piper_audio_chunk *chunk);

int piper_abi_version(void)
{
    return PIPER_ABI_VERSION;
}

static bool ensure_espeak_initialized(const char *espeak_data_path, const std::string &espeak_voice)
{
    if (g_espeak_ready.load(std::memory_order_acquire))
    {
        return true;
    }

    std::lock_guard<std::mutex> lock(g_espeak_init_mutex);
    if (g_espeak_ready.load(std::memory_order_relaxed))
    {
        return true;
    }

    espeak_ng_ERROR_CONTEXT context = nullptr;
    espeak_ng_STATUS status = espeak_ng_InitializePhonemizer(espeak_data_path, espeak_voice.c_str(), &context);
    if (status != ENS_OK && status != ENS_VOICE_NOT_FOUND)
    {
        espeak_ng_PrintStatusCodeMessage(status, stderr, context);
        espeak_ng_ClearErrorContext(&context);
        return false;
    }

    g_espeak_voice = (status == ENS_OK) ? espeak_voice : std::string();

    g_espeak_ready.store(true, std::memory_order_release);
    return true;
}

static void parse_voice_config(const json &config, piper_voice &voice)
{
    voice.espeak_voice = "en-us"; // default
    if (config.contains("espeak"))
    {
        auto &espeak_obj = config["espeak"];
        if (espeak_obj.contains("voice"))
        {
            voice.espeak_voice = espeak_obj["voice"].get<std::string>();
        }
    }

    if (config.contains("audio"))
    {
        auto &audio_obj = config["audio"];
        if (audio_obj.contains("sample_rate"))
        {
            // Sample rate of generated audio in hertz
            voice.sample_rate = audio_obj["sample_rate"].get<int>();
        }
    }

    // phoneme to [id] map
    // Maps phonemes to one or more phoneme ids (required).
    if (config.contains("phoneme_id_map"))
    {
        auto &phoneme_id_map_value = config["phoneme_id_map"];
        for (auto &from_phoneme_item : phoneme_id_map_value.items())
        {
            std::string from_phoneme = from_phoneme_item.key();
            auto from_codepoint = get_codepoint(from_phoneme);
            if (!from_codepoint)
            {
                // No codepoint
                continue;
            }

            for (auto &to_id_value : from_phoneme_item.value())
            {
                PhonemeId to_id = to_id_value.get<PhonemeId>();
                voice.phoneme_id_map[*from_codepoint].push_back(to_id);
            }
        }
    }

    voice.num_speakers = config.at("num_speakers").get<SpeakerId>();

    if (config.contains("inference"))
    {
        // Overrides default inference settings
        auto inference_value = config["inference"];
        if (inference_value.contains("noise_scale"))
        {
            voice.synth_noise_scale =
                inference_value["noise_scale"].get<float>();
        }

        if (inference_value.contains("length_scale"))
        {
            voice.synth_length_scale =
                inference_value["length_scale"].get<float>();
        }

        if (inference_value.contains("noise_w"))
        {
            voice.synth_noise_w_scale =
                inference_value["noise_w"].get<float>();
        }
    }
}

static std::unique_ptr<Ort::Session> create_onnx_session(const std::string &model_path, Ort::SessionOptions &session_options)
{
    session_options.DisableCpuMemArena();
    session_options.DisableMemPattern();
    session_options.DisableProfiling();
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);

    bool is_ort_model = model_path.size() >= 4 && model_path.compare(model_path.size() - 4, 4, ".ort") == 0;

    if (is_ort_model)
    {
        session_options.AddConfigEntry("session.use_memory_mapped_ort_model", "1");
        session_options.AddConfigEntry("session.use_ort_model_bytes_for_initializers", "1");
        session_options.AddConfigEntry("session.disable_prepacking", "1");
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    }

    return std::make_unique<Ort::Session>(Ort::Session(ort_env, model_path.c_str(), session_options));
}

struct piper_synthesizer *piper_create(const char *model_path, const char *config_path, const char *espeak_data_path)
{
    piper_voice *voice = piper_create_voice(model_path, config_path, espeak_data_path);
    if (!voice)
    {
        return nullptr;
    }

    piper_synthesizer *synth = new piper_synthesizer();
    synth->voice = voice;
    synth->session = create_onnx_session(synth->voice->model_path, synth->session_options);

    return synth;
}

void piper_free(struct piper_synthesizer *synth)
{
    if (!synth)
    {
        return;
    }

    // No espeak_Terminate(): espeak-ng is process-wide, initialized once and shared by all voices

    piper_free_voice(synth->voice);
    delete synth;

#ifdef __linux__
    malloc_trim(0);
#endif
}

piper_voice *piper_create_voice(const char *model_path, const char *config_path, const char *espeak_data_path)
{
    if (!model_path)
    {
        return nullptr;
    }

    try
    {
        std::string config_path_str;
        if (!config_path)
        {
            std::string model_path_str(model_path);
            config_path_str = model_path_str + ".json";
        }
        else
        {
            config_path_str = config_path;
        }

        std::ifstream config_stream(config_path_str);
        auto config = json::parse(config_stream);

        auto voice = std::make_unique<piper_voice>();
        voice->model_path = model_path;

        parse_voice_config(config, *voice);

        if (!ensure_espeak_initialized(espeak_data_path, voice->espeak_voice))
        {
            return nullptr;
        }

        return voice.release();
    }
    catch (const std::exception &e)
    {
        log_exception("piper_create_voice", model_path, e.what());
        return nullptr;
    }
    catch (...)
    {
        log_exception("piper_create_voice", model_path, "unknown exception");
        return nullptr;
    }
}

piper_synthesizer *piper_create_synthesizer(piper_voice *voice)
{
    if (!voice)
    {
        return nullptr;
    }

    try
    {
        auto synth = std::make_unique<piper_synthesizer>();
        synth->voice = voice;
        synth->session = create_onnx_session(voice->model_path, synth->session_options);

        return synth.release();
    }
    catch (const std::exception &e)
    {
        log_exception("piper_create_synthesizer", voice->model_path.c_str(), e.what());
        return nullptr;
    }
    catch (...)
    {
        log_exception("piper_create_synthesizer", voice->model_path.c_str(), "unknown exception");
        return nullptr;
    }
}

void piper_free_voice(piper_voice *voice)
{
    delete voice;
}

void piper_free_synthesizer(piper_synthesizer *synth)
{
    if (!synth)
    {
        return;
    }

    delete synth;

#ifdef __linux__
    malloc_trim(0);
#endif
}

piper_synthesize_options piper_default_synthesize_options(piper_synthesizer *synth)
{
    return piper_default_synthesize_options2(synth && synth->voice ? synth->voice : nullptr);
}

piper_synthesize_options piper_default_synthesize_options2(piper_voice *voice)
{
    piper_synthesize_options options;
    options.speaker_id = 0;
    options.length_scale = DEFAULT_LENGTH_SCALE;
    options.noise_scale = DEFAULT_NOISE_SCALE;
    options.noise_w_scale = DEFAULT_NOISE_W_SCALE;

    if (voice)
    {
        options.length_scale = voice->synth_length_scale;
        options.noise_scale = voice->synth_noise_scale;
        options.noise_w_scale = voice->synth_noise_w_scale;
    }

    return options;
}

int piper_synthesize_start(struct piper_synthesizer *synth, const char *text, const piper_synthesize_options *options)
{
    try
    {
        return synthesize_start_impl(synth, text, options);
    }
    catch (const std::exception &e)
    {
        log_exception("piper_synthesize_start", synth->voice->model_path.c_str(), e.what());
        return PIPER_ERR_GENERIC;
    }
    catch (...)
    {
        log_exception("piper_synthesize_start", synth->voice->model_path.c_str(), "unknown exception");
        return PIPER_ERR_GENERIC;
    }
}

int piper_synthesize_next(struct piper_synthesizer *synth, struct piper_audio_chunk *chunk)
{
    try
    {
        return synthesize_next_impl(synth, chunk);
    }
    catch (const std::exception &e)
    {
        log_exception("piper_synthesize_next", synth->voice->model_path.c_str(), e.what());
        return PIPER_ERR_GENERIC;
    }
    catch (...)
    {
        log_exception("piper_synthesize_next", synth->voice->model_path.c_str(), "unknown exception");
        return PIPER_ERR_GENERIC;
    }
}

static void split_words(std::string_view text, std::vector<size_t> &word_starts, std::vector<int> &word_weights)
{
    word_starts.clear();
    word_weights.clear();

    size_t start = 0;
    while (true)
    {
        size_t end = text.find(' ', start);
        std::string_view word = text.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);

        int letters = 0;
        for (char32_t c : una::ranges::utf8_view{word})
        {
            if (una::codepoint::is_alphabetic(c) || una::codepoint::is_numeric(c))
            {
                letters++;
            }
        }

        // empty words (consecutive spaces) take no time
        word_starts.push_back(start);
        word_weights.push_back(word.empty() ? 0 : std::max(letters, 1));

        if (end == std::string_view::npos)
        {
            break;
        }
        start = end + 1;
    }
}

// Phoneme ids of a sentence and the word owning each one: phonemes belong to the word of their group, spaces and
// terminators (pauses) to the word before them, BOS to the sentence's first word and EOS to its last.
// word carries the current word across sentences; -1 until the text's first group.
static void build_ids(const PhonemeIdMap &phoneme_id_map, const std::vector<ClausePhonemes> &clauses, int &word, std::vector<PhonemeId> &ids, std::vector<int> &id_words)
{
    auto push_ids = [&](Phoneme phoneme)
    {
        auto ids_for_phoneme = phoneme_id_map.find(phoneme);
        if (ids_for_phoneme == phoneme_id_map.end())
        {
            // Not in the map: skipped, as in upstream Piper
            return;
        }
        for (auto id : ids_for_phoneme->second)
        {
            ids.push_back(id);
            id_words.push_back(word);
            ids.push_back(ID_PAD);
            id_words.push_back(word);
        }
    };

    // BOS group, owned by the first word once its group is reached
    ids.push_back(ID_BOS);
    id_words.push_back(word);
    ids.push_back(ID_PAD);
    id_words.push_back(word);
    bool first_group = true;

    auto set_word = [&](int group_word)
    {
        word = group_word;
        if (first_group)
        {
            id_words[0] = word;
            id_words[1] = word;
            first_group = false;
        }
    };

    // Filter out (lang) switch (flags).
    // These surround words from languages other than the current voice.
    bool in_lang_flag = false;
    for (const auto &clause : clauses)
    {
        size_t group = 0;
        if (!clause.group_words.empty())
        {
            set_word(clause.group_words[0]);
        }

        auto phonemes_norm = una::norm::to_nfd_utf8(clause.phonemes);
        for (char32_t phoneme : una::ranges::utf8_view{phonemes_norm})
        {
            if (in_lang_flag)
            {
                if (phoneme == U')')
                {
                    // End of (lang) switch
                    in_lang_flag = false;
                }
            }
            else if (phoneme == U'(')
            {
                // Start of (lang) switch
                in_lang_flag = true;
            }
            else
            {
                push_ids(phoneme);

                // espeak writes one space per word start, so each space opens the next group
                if (phoneme == U' ' && group + 1 < clause.group_words.size())
                {
                    set_word(clause.group_words[++group]);
                }
            }
        }

        for (char32_t phoneme : una::ranges::utf8_view{clause.terminator})
        {
            push_ids(phoneme);
        }
    }

    // EOS group
    ids.push_back(ID_EOS);
    id_words.push_back(word);
}

// Every word is reported once, with the chunk holding the last id of its group's voiced word. Words without ids
// (merged by espeak into the previous word, punctuation only, empty) join the voiced word before them, or the first
// voiced word at the text start, and share its time by letter count.
static void plan_word_groups(std::vector<piper_synthesizer::PhonemeChunk> &chunks, int num_words)
{
    // ids before the text's first group (no word yet) belong to the first word that has a group
    int first_word = -1;
    for (const auto &chunk : chunks)
    {
        auto found = std::find_if(chunk.id_words.begin(), chunk.id_words.end(), [](int w)
                                  { return w >= 0; });
        if (found != chunk.id_words.end())
        {
            first_word = *found;
            break;
        }
    }

    std::vector<int> last_chunk(num_words, -1);
    for (size_t c = 0; c < chunks.size(); c++)
    {
        for (int &w : chunks[c].id_words)
        {
            if (w < 0)
            {
                w = std::max(first_word, 0);
            }
            last_chunk[w] = (int)c;
        }
    }

    int emit_chunk = 0;
    int w = 0;
    while (w < num_words)
    {
        int voiced = w;
        while (voiced < num_words && last_chunk[voiced] < 0)
        {
            voiced++;
        }
        if (voiced == num_words)
        {
            // no audio in the whole text
            break;
        }

        int end = voiced + 1;
        while (end < num_words && last_chunk[end] < 0)
        {
            end++;
        }

        // never report a group before an earlier one
        emit_chunk = std::max(emit_chunk, last_chunk[voiced]);
        chunks[emit_chunk].word_groups.push_back({w, end - 1, voiced});
        w = end;
    }
}

// Word timings of the groups ending in this chunk; times are rounded from absolute sample positions, so they don't drift
static void emit_word_groups(piper_synthesizer *synth, const std::vector<piper_synthesizer::WordGroup> &groups)
{
    int sample_rate = synth->voice->sample_rate;

    for (const auto &group : groups)
    {
        int64_t start = synth->word_start_samples[group.voiced_word];
        int64_t span = synth->word_end_samples[group.voiced_word] - start;
        if (start < 0)
        {
            // fewer durations than ids: zero length at the end of the audio so far
            start = synth->samples_done;
            span = 0;
        }

        int total_weight = 0;
        for (int w = group.first_word; w <= group.last_word; w++)
        {
            total_weight += synth->word_weights[w];
        }

        // an all-empty group still gives its span to the voiced word
        auto weight_of = [&](int w)
        {
            return (total_weight > 0) ? synth->word_weights[w] : (w == group.voiced_word ? 1 : 0);
        };
        int weight_sum = std::max(total_weight, 1);

        int weight_before = 0;
        for (int w = group.first_word; w <= group.last_word; w++)
        {
            int offset_ms = samples_to_ms(start + span * weight_before / weight_sum, sample_rate);
            weight_before += weight_of(w);
            int end_ms = samples_to_ms(start + span * weight_before / weight_sum, sample_rate);

            synth->chunk_words.push_back({w, offset_ms, end_ms - offset_ms});
        }
    }
}

static int samples_to_ms(int64_t samples, int sample_rate)
{
    return (int)((samples * 1000 + sample_rate / 2) / sample_rate);
}

static int synthesize_start_impl(struct piper_synthesizer *synth, const char *text, const piper_synthesize_options *options)
{
    if (!synth || !text)
    {
        return PIPER_ERR_GENERIC;
    }

    if (g_espeak_voice != synth->voice->espeak_voice)
    {
        g_espeak_voice.clear();

        if (espeak_ng_SetPhonemizerVoiceByName(synth->voice->espeak_voice.c_str()) != ENS_OK)
        {
            return PIPER_ERR_GENERIC;
        }

        g_espeak_voice = synth->voice->espeak_voice;
    }

    // Clear state
    while (!synth->phoneme_id_queue.empty())
    {
        synth->phoneme_id_queue.pop();
    }
    synth->chunk_samples.clear();

    piper_synthesize_options default_options;
    if (!options)
    {
        default_options = piper_default_synthesize_options(synth);
        options = &default_options;
    }

    synth->length_scale = options->length_scale;
    synth->noise_scale = options->noise_scale;
    synth->noise_w_scale = options->noise_w_scale;
    synth->speaker_id = options->speaker_id;

    std::string_view text_view{text};
    std::vector<size_t> word_starts;
    split_words(text_view, word_starts, synth->word_weights);
    int num_words = (int)word_starts.size();

    synth->word_start_samples.assign(num_words, -1);
    synth->word_end_samples.assign(num_words, 0);
    synth->chunk_words.clear();
    synth->samples_done = 0;

    // word containing a source pointer espeak returned
    auto word_of = [&](const char *source)
    {
        size_t offset = (source > text) ? std::min((size_t)(source - text), text_view.size()) : 0;
        return (int)(std::upper_bound(word_starts.begin(), word_starts.end(), offset) - word_starts.begin()) - 1;
    };

    // phonemize
    std::vector<std::vector<ClausePhonemes>> sentences(1);
    const void *text_ptr = text;
    while (text_ptr != nullptr)
    {
        int terminator = 0;
        std::string terminator_str = "";

        espeak_PHONEME_WORDS phoneme_words = {0, nullptr};
        const char *phonemes = espeak_TextToPhonemesWithTerminator(&text_ptr, espeakCHARS_AUTO, espeakPHONEMES_IPA, &terminator, &phoneme_words);

        if (!phonemes)
        {
            return PIPER_ERR_GENERIC;
        }

        ClausePhonemes clause;
        clause.phonemes = phonemes;
        for (int i = 0; i < phoneme_words.count; i++)
        {
            clause.group_words.push_back(word_of(phoneme_words.sources[i]));
        }

        // Categorize terminator
        terminator &= 0x000FFFFF;

        if (terminator == CLAUSE_PERIOD)
        {
            terminator_str = ".";
        }
        else if (terminator == CLAUSE_QUESTION)
        {
            terminator_str = "?";
        }
        else if (terminator == CLAUSE_EXCLAMATION)
        {
            terminator_str = "!";
        }
        else if (terminator == CLAUSE_COMMA)
        {
            terminator_str = ", ";
        }
        else if (terminator == CLAUSE_COLON)
        {
            terminator_str = ": ";
        }
        else if (terminator == CLAUSE_SEMICOLON)
        {
            terminator_str = "; ";
        }

        clause.terminator = terminator_str;
        sentences.back().push_back(std::move(clause));

        if ((terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE)
        {
            sentences.emplace_back();
        }
    }

    // phonemes to ids
    std::vector<piper_synthesizer::PhonemeChunk> chunks;
    int word = -1;
    for (const auto &sentence : sentences)
    {
        bool empty = std::all_of(sentence.begin(), sentence.end(), [](const ClausePhonemes &clause)
                                 { return clause.phonemes.empty() && clause.terminator.empty(); });
        if (empty)
        {
            continue;
        }

        piper_synthesizer::PhonemeChunk chunk;
        build_ids(synth->voice->phoneme_id_map, sentence, word, chunk.ids, chunk.id_words);
        chunks.push_back(std::move(chunk));
    }

    plan_word_groups(chunks, num_words);

    for (auto &chunk : chunks)
    {
        synth->phoneme_id_queue.push(std::move(chunk));
    }

    return PIPER_OK;
}

static int synthesize_next_impl(struct piper_synthesizer *synth, struct piper_audio_chunk *chunk)
{
    if (!synth)
    {
        return PIPER_ERR_GENERIC;
    }

    if (!chunk)
    {
        return PIPER_ERR_GENERIC;
    }

    // Clear data from previous call
    synth->chunk_samples.clear();
    synth->chunk_words.clear();

    chunk->samples = nullptr;
    chunk->num_samples = 0;
    chunk->is_last = false;
    chunk->words = nullptr;
    chunk->num_words = 0;

    if (synth->phoneme_id_queue.empty())
    {
        // Empty final chunk
        chunk->is_last = true;
        return PIPER_DONE;
    }

    // Process next list of phoneme ids
    auto next_chunk = std::move(synth->phoneme_id_queue.front());
    synth->phoneme_id_queue.pop();
    auto &next_ids = next_chunk.ids;

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Allocate
    std::vector<int64_t> phoneme_id_lengths{(int64_t)next_ids.size()};
    std::vector<float> scales{synth->noise_scale, synth->length_scale, synth->noise_w_scale};

    std::vector<Ort::Value> input_tensors;
    std::vector<int64_t> phoneme_ids_shape{1, (int64_t)next_ids.size()};
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, next_ids.data(), next_ids.size(), phoneme_ids_shape.data(),
        phoneme_ids_shape.size()));

    std::vector<int64_t> phoneme_id_lengths_shape{
        (int64_t)phoneme_id_lengths.size()};
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, phoneme_id_lengths.data(), phoneme_id_lengths.size(),
        phoneme_id_lengths_shape.data(), phoneme_id_lengths_shape.size()));

    std::vector<int64_t> scales_shape{(int64_t)scales.size()};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, scales.data(), scales.size(), scales_shape.data(),
        scales_shape.size()));

    // Add speaker id.
    // NOTE: These must be kept outside the "if" below to avoid being
    // deallocated.
    std::vector<int64_t> speaker_id{(int64_t)synth->speaker_id};
    std::vector<int64_t> speaker_id_shape{(int64_t)speaker_id.size()};

    if (synth->voice->num_speakers > 1)
    {
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, speaker_id.data(), speaker_id.size(),
            speaker_id_shape.data(), speaker_id_shape.size()));
    }

    // From export_onnx.py
    std::array<const char *, 4> input_names = {"input", "input_lengths", "scales", "sid"};

    // Get all output names
    std::vector<std::string> output_names_strs = synth->session->GetOutputNames();
    std::vector<const char *> output_names;
    for (const auto &name : output_names_strs)
    {
        output_names.push_back(name.c_str());
    }

    // Infer
    auto output_tensors = synth->session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names.data(),
        output_names.size());

    if ((output_tensors.size() < 1) || (!output_tensors.front().IsTensor()))
    {
        return PIPER_ERR_GENERIC;
    }

    auto audio_shape = output_tensors.front()
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();
    chunk->num_samples = audio_shape[audio_shape.size() - 1];

    const float *audio_tensor_data = output_tensors.front()
                                         .GetTensorData<float>();
    synth->chunk_samples.resize(chunk->num_samples);
    std::copy(audio_tensor_data, audio_tensor_data + chunk->num_samples, synth->chunk_samples.begin());
    chunk->samples = synth->chunk_samples.data();
    chunk->is_last = synth->phoneme_id_queue.empty();

    int64_t chunk_start = synth->samples_done;
    synth->samples_done += (int64_t)chunk->num_samples;

    // Phoneme durations (frames per id) give word timings
    if (output_tensors.size() > 1 && synth->voice->sample_rate > 0)
    {
        auto durations_shape = output_tensors[1]
                                   .GetTensorTypeAndShapeInfo()
                                   .GetShape();
        const float *durations = output_tensors[1].GetTensorData<float>();
        size_t num_ids = std::min((size_t)durations_shape[durations_shape.size() - 1], next_chunk.id_words.size());
        int hop_length = synth->voice->hop_length;

        // id positions are scaled so the chunk's words cover exactly its audio
        int64_t total_samples = 0;
        for (size_t i = 0; i < num_ids; i++)
        {
            total_samples += (int64_t)(durations[i] * hop_length);
        }

        int64_t num_samples = (int64_t)chunk->num_samples;
        int64_t position = 0;
        for (size_t i = 0; i < num_ids; i++)
        {
            int64_t id_start = chunk_start + (total_samples > 0 ? position * num_samples / total_samples : 0);
            position += (int64_t)(durations[i] * hop_length);
            int64_t id_end = chunk_start + (total_samples > 0 ? position * num_samples / total_samples : 0);

            int w = next_chunk.id_words[i];
            if (synth->word_start_samples[w] < 0)
            {
                synth->word_start_samples[w] = id_start;
            }
            synth->word_end_samples[w] = id_end;
        }

        emit_word_groups(synth, next_chunk.word_groups);
        chunk->words = synth->chunk_words.data();
        chunk->num_words = synth->chunk_words.size();
    }

    // Clean up
    for (std::size_t i = 0; i < output_tensors.size(); i++)
    {
        Ort::detail::OrtRelease(output_tensors[i].release());
    }

    for (std::size_t i = 0; i < input_tensors.size(); i++)
    {
        Ort::detail::OrtRelease(input_tensors[i].release());
    }

    return PIPER_OK;
}

// stdout is reserved for the worker MODEL_* protocol
static void log_exception(const char *func, const char *model_path, const char *what)
{
    fprintf(stderr, "[libpiper] %s failed for %s: %s\n", func, model_path, what);
}
