#ifndef PIPER_H_
#define PIPER_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <uchar.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PIPER_OK 0
#define PIPER_DONE 1
#define PIPER_ERR_GENERIC -1

/**
 * \brief Text-to-speech synthesizer.
 */
typedef struct piper_synthesizer piper_synthesizer;

/**
 * \brief Parsed, lightweight voice configuration
 */
typedef struct piper_voice piper_voice;

/**
 * \brief Version of the structs exchanged with the library.
 *
 * Bumped on every layout change; bindings compare it with \ref piper_abi_version.
 */
#define PIPER_ABI_VERSION 2

/**
 * \brief Timing of one word of the synthesized text.
 *
 * Words are the ' '-separated atoms of the text given to \ref piper_synthesize_start
 * (consecutive spaces give empty words), so word_index matches a plain split on ' '.
 */
typedef struct piper_word_alignment {
  /**
   * \brief Index of the word in the text.
   */
  int word_index;

  /**
   * \brief Start of the word in milliseconds, relative to the audio of the whole text.
   */
  int offset_ms;

  /**
   * \brief Duration in milliseconds, including the pause after the word.
   *
   * offset_ms + duration_ms is the offset_ms of the next word.
   */
  int duration_ms;
} piper_word_alignment;

/**
 * \brief Chunk of synthesized audio samples.
 */
typedef struct piper_audio_chunk {
  /**
   * \brief Raw samples returned from the voice model.
   */
  const float *samples;

  /**
   * \brief Number of samples in the audio chunk.
   */
  size_t num_samples;

  /**
   * \brief True if this is the last audio chunk.
   */
  bool is_last;

  /**
   * \brief Words whose audio ends in this chunk, in ascending word_index.
   *
   * Over all chunks every word is reported exactly once. Words without audio of
   * their own (merged by espeak into a neighbour, punctuation only, empty) share
   * the neighbour's time, split by letter count (empty words get none). Empty
   * when the voice model has no phoneme durations output.
   */
  const piper_word_alignment *words;

  /**
   * \brief Number of entries in words.
   */
  size_t num_words;
} piper_audio_chunk;

/**
 * \brief Options for synthesis.
 *
 * \sa \ref piper_default_synthesize_options
 */
typedef struct piper_synthesize_options {
  /**
   * \brief Id of speaker to use (multi-speaker models only).
   *
   * Id 0 is the first speaker.
   */
  int speaker_id;

  /**
   * \brief How fast the text is spoken.
   *
   * A length scale of 0.5 means to speak twice as fast.
   * A length scale of 2.0 means to speak twice as slow.
   * The default is 1.0.
   */
  float length_scale;

  /**
   * \brief Controls how much noise is added during synthesis.
   *
   * The best value depends on the voice.
   * For single speaker models, a value of 0.667 is usually good.
   * For multi-speaker models, a value of 0.333 is usually good.
   */
  float noise_scale;

  /**
   * \brief Controls how much phonemes vary in length during synthesis.
   *
   * The best value depends on the voice.
   * For single speaker models, a value of 0.8 is usually good.
   * For multi-speaker models, a value of 0.333 is usually good.
   */
  float noise_w_scale;
} piper_synthesize_options;

/**
 * \brief Get the struct version the library was built with.
 *
 * \return PIPER_ABI_VERSION of the library.
 */
int piper_abi_version(void);

/**
 * \brief Create a Piper text-to-speech synthesizer from a voice model.
 *
 * \param model_path path to ONNX voice model file.
 *
 * \param config_path path to JSON voice config file or NULL if it's the
 * model_path + .json.
 *
 * \param espeak_data_path path to the espeak-ng data
 * directory.
 *
 * \return a Piper text-to-speech synthesizer for the voice model.
 */
piper_synthesizer *piper_create(const char *model_path, const char *config_path,
                                const char *espeak_data_path);

/**
 * \brief Parse a Piper voice model's configuration without creating an ONNX Runtime session.
 *
 * Unlike \ref piper_create, this does not load the ONNX model — only the
 * JSON config (espeak voice, audio settings, phoneme_id_map, default
 * inference scales) is parsed. Call \ref piper_load_model to obtain a
 * synthesizer with a live ONNX Runtime session for actual inference.
 *
 * \param model_path path to ONNX voice model file.
 *
 * \param config_path path to JSON voice config file or NULL if it's the model_path + .json.
 *
 * \param espeak_data_path path to the espeak-ng data directory.
 *
 * \return a parsed voice configuration, or NULL on failure.
 */
piper_voice *piper_create_voice(const char *model_path, const char *config_path, const char *espeak_data_path);

/**
 * \brief Free resources for Piper synthesizer.
 *
 * Does not call espeak_Terminate() — espeak-ng is initialized once for the
 * life of the process and shared across all voices.
 *
 * \param synth Piper synthesizer.
 */
void piper_free(piper_synthesizer *synth);

/**
 * \brief Free a voice configuration created by \ref piper_create_voice.
 *
 * Does not call espeak_Terminate() — espeak-ng is initialized once for the
 * life of the process and shared across all voices.
 *
 * \param voice Voice configuration.
 */
void piper_free_voice(piper_voice *voice);

/**
 * \brief Get the default synthesis options for a Piper synthesizer.
 *
 * \param synth Piper synthesizer.
 *
 * \return synthesis options from voice config.
 */
piper_synthesize_options
piper_default_synthesize_options(piper_synthesizer *synth);

/**
 * \brief Get the default synthesis options for a voice configuration.
 *
 * \param voice Voice configuration.
 *
 * \return synthesis options from voice config.
 */
piper_synthesize_options
piper_default_synthesize_options2(piper_voice *voice);

/**
 * \brief Load the ONNX Runtime session for a voice, for immediate
 * synthesis.
 *
 * \param voice Voice configuration from \ref piper_create_voice.
 *
 * \return a Piper text-to-speech synthesizer bound to the voice, or NULL on
 * failure.
 */
piper_synthesizer *piper_create_synthesizer(piper_voice *voice);

/**
 * \brief Unload the ONNX Runtime session created by \ref piper_create_synthesizer.
 *
 * Releases the session and per-call working buffers only — the voice
 * configuration it was bound to is untouched and may be reused with another
 * call to \ref piper_create_synthesizer.
 *
 * \param synth Piper synthesizer.
 */
void piper_free_synthesizer(piper_synthesizer *synth);

/**
 * \brief Start text-to-speech synthesis.
 *
 * \param synth Piper synthesizer.
 *
 * \param text text to synthesize into audio.
 *
 * \param options synthesis options or NULL for defaults.
 *
 * \sa \ref piper_synthesize_next
 *
 * \return PIPER_OK or error code.
 */
int piper_synthesize_start(piper_synthesizer *synth, const char *text,
                           const piper_synthesize_options *options);

/**
 * \brief Synthesize next chunk of audio.
 *
 * \param synth Piper synthesizer.
 *
 * \param chunk audio chunk to fill.
 *
 * piper_synthesize_start must be called before this function.
 * Each call to piper_synthesize_next will fill the audio chunk, invalidating
 * the memory of the previous chunk.
 * The final audio chunk will have is_last = true.
 * A return value of PIPER_DONE indicates that synthesis is complete.
 *
 * \sa \ref piper_synthesize_start
 *
 * \return PIPER_DONE when complete, otherwise PIPER_OK or error code.
 */
int piper_synthesize_next(piper_synthesizer *synth, piper_audio_chunk *chunk);

#ifdef __cplusplus
}
#endif

#endif // PIPER_H_
