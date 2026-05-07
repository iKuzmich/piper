/**
 * piper_test - interactive debug harness for libpiper
 *
 * Usage:
 *   piper_test <model.onnx> <espeak-ng-data> <text> [output.raw]
 *
 * Play output:
 *   aplay -r 22050 -c 1 -f FLOAT_LE -t raw output.raw
 *
 * Environment overrides:
 *   PIPER_LENGTH_SCALE   float  speech rate (0.5 = 2x faster, 2.0 = 2x slower)
 *   PIPER_NOISE_SCALE    float  synthesis noise
 *   PIPER_NOISE_W_SCALE  float  phoneme length variation
 *   PIPER_SPEAKER_ID     int    speaker index for multi-speaker models
 */

#include <piper.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model.onnx> <espeak-ng-data> <text> [output.raw]\n", prog);
    fprintf(stderr, "  model.onnx.json is assumed to be <model.onnx>.json unless\n");
    fprintf(stderr, "  the file is present at a custom location.\n");
    fprintf(stderr, "\nEnvironment overrides:\n");
    fprintf(stderr, "  PIPER_LENGTH_SCALE   float  (default: voice config)\n");
    fprintf(stderr, "  PIPER_NOISE_SCALE    float  (default: voice config)\n");
    fprintf(stderr, "  PIPER_NOISE_W_SCALE  float  (default: voice config)\n");
    fprintf(stderr, "  PIPER_SPEAKER_ID     int    (default: 0)\n");
}

static void section(const char *title) {
    printf("\n--- %s ---\n", title);
}

// Print char32_t phoneme array as readable text.
// Separators (U+0000) are shown as '|'; non-ASCII codepoints as [U+XXXX].
static void print_phonemes(const char32_t *ph, size_t n) {
    if (!ph || n == 0) {
        printf("(empty)");
        return;
    }
    for (size_t i = 0; i < n; i++) {
        char32_t cp = ph[i];
        if (cp == 0) {
            printf("|");
        } else if (cp >= 0x20 && cp < 0x7F) {
            printf("%c", static_cast<char>(cp));
        } else {
            printf("[U+%04X]", static_cast<unsigned>(cp));
        }
    }
}

// Print the per-phoneme alignment table for one chunk.
static void print_alignment_table(const piper_audio_chunk &c) {
    if (c.num_alignments == 0) {
        printf("  (no alignment data)\n");
        return;
    }

    // Walk phonemes array and alignments in lock-step.
    // phonemes layout: [p, p, 0,  p, p, 0, ...] — groups of N codepoints then 0.
    // alignments and phoneme_ids are indexed per-group-element.
    size_t ph_pos = 0;
    size_t id_pos = 0;

    printf("  %-12s %8s %8s\n", "phoneme", "id", "samples");
    printf("  %-12s %8s %8s\n", "-------", "--", "-------");

    while (ph_pos < c.num_phonemes && id_pos < c.num_alignments) {
        // Collect this group's codepoint (all repeated copies are the same).
        char32_t cp = c.phonemes[ph_pos];
        char label[32];
        if (cp == 0) break;
        if (cp >= 0x20 && cp < 0x7F) {
            snprintf(label, sizeof(label), "%c", static_cast<char>(cp));
        } else {
            snprintf(label, sizeof(label), "U+%04X", static_cast<unsigned>(cp));
        }

        // Count how many identical codepoints are in this group (before the 0).
        size_t group_size = 0;
        size_t peek = ph_pos;
        while (peek < c.num_phonemes && c.phonemes[peek] != 0) {
            group_size++;
            peek++;
        }
        // Skip past the separator.
        size_t next_ph = peek + 1;

        // Print one row per id in this group.
        for (size_t g = 0; g < group_size / 2 && id_pos < c.num_alignments; g++) {
            int id  = (id_pos < c.num_phoneme_ids) ? c.phoneme_ids[id_pos] : -1;
            int dur = c.alignments[id_pos];
            printf("  %-12s %8d %8d\n", g == 0 ? label : "", id, dur);
            id_pos++;
            // skip the paired PAD id
            if (id_pos < c.num_alignments) {
                id_pos++;
            }
        }

        ph_pos = next_ph;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char *model_path      = argv[1];
    const char *espeak_data     = argv[2];
    const char *text            = argv[3];
    const char *output_path     = (argc > 4) ? argv[4] : "output.raw";

    printf("=== piper_test ===\n");
    printf("model:       %s\n", model_path);
    printf("espeak-data: %s\n", espeak_data);
    printf("text:        %s\n", text);
    printf("output:      %s\n", output_path);

    // ------------------------------------------------------------------
    // Create synthesizer
    // ------------------------------------------------------------------
    section("piper_create");
    piper_synthesizer *synth = piper_create(model_path, nullptr, espeak_data);
    if (!synth) {
        fprintf(stderr, "FAIL: piper_create returned null\n");
        return 1;
    }
    printf("OK\n");

    // ------------------------------------------------------------------
    // Default options
    // ------------------------------------------------------------------
    section("piper_default_synthesize_options");
    piper_synthesize_options opts = piper_default_synthesize_options(synth);
    printf("  speaker_id:    %d\n",   opts.speaker_id);
    printf("  length_scale:  %.4f\n", opts.length_scale);
    printf("  noise_scale:   %.4f\n", opts.noise_scale);
    printf("  noise_w_scale: %.4f\n", opts.noise_w_scale);

    // Apply environment overrides.
    const char *env_len   = getenv("PIPER_LENGTH_SCALE");
    const char *env_noise = getenv("PIPER_NOISE_SCALE");
    const char *env_noisew = getenv("PIPER_NOISE_W_SCALE");
    const char *env_sid   = getenv("PIPER_SPEAKER_ID");
    if (env_len)    opts.length_scale   = static_cast<float>(atof(env_len));
    if (env_noise)  opts.noise_scale    = static_cast<float>(atof(env_noise));
    if (env_noisew) opts.noise_w_scale  = static_cast<float>(atof(env_noisew));
    if (env_sid)    opts.speaker_id     = atoi(env_sid);

    if (env_len || env_noise || env_noisew || env_sid) {
        printf("Effective options after env overrides:\n");
        printf("  speaker_id:    %d\n",   opts.speaker_id);
        printf("  length_scale:  %.4f\n", opts.length_scale);
        printf("  noise_scale:   %.4f\n", opts.noise_scale);
        printf("  noise_w_scale: %.4f\n", opts.noise_w_scale);
    }

    // ------------------------------------------------------------------
    // Start synthesis
    // ------------------------------------------------------------------
    section("piper_synthesize_start");
    int rc = piper_synthesize_start(synth, text, &opts);
    if (rc != PIPER_OK) {
        fprintf(stderr, "FAIL: piper_synthesize_start returned %d\n", rc);
        piper_free(synth);
        return 1;
    }
    printf("OK\n");

    // ------------------------------------------------------------------
    // Open output file
    // ------------------------------------------------------------------
    FILE *out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "FAIL: cannot open output file: %s\n", output_path);
        piper_free(synth);
        return 1;
    }

    // ------------------------------------------------------------------
    // Iterate chunks
    // ------------------------------------------------------------------
    size_t total_samples = 0;
    int    chunk_idx     = 0;
    int    sample_rate   = 0;
    piper_audio_chunk chunk;

    while ((rc = piper_synthesize_next(synth, &chunk)) == PIPER_OK) {
        sample_rate = chunk.sample_rate;
        section("chunk");
        printf("  index:          %d\n",   chunk_idx);
        printf("  sample_rate:    %d Hz\n", chunk.sample_rate);
        printf("  num_samples:    %zu  (%.3f s)\n",
               chunk.num_samples,
               chunk.sample_rate > 0 ? (double)chunk.num_samples / chunk.sample_rate : 0.0);
        printf("  is_last:        %s\n",   chunk.is_last ? "true" : "false");

        printf("  phonemes (%zu):  ", chunk.num_phonemes);
        print_phonemes(chunk.phonemes, chunk.num_phonemes);
        printf("\n");

        if (chunk.phonemes_aligned && chunk.num_phonemes_aligned > 0) {
            printf("  phonemes_aligned (%zu): ", chunk.num_phonemes_aligned);
            print_phonemes(chunk.phonemes_aligned, chunk.num_phonemes_aligned);
            printf("\n");
        }

        if (chunk.num_phoneme_ids > 0) {
            printf("  phoneme_ids (%zu): ", chunk.num_phoneme_ids);
            size_t show = chunk.num_phoneme_ids < 30 ? chunk.num_phoneme_ids : 30;
            for (size_t i = 0; i < show; i++) {
                printf("%d ", chunk.phoneme_ids[i]);
            }
            if (chunk.num_phoneme_ids > 30) printf("...");
            printf("\n");
        }

        if (chunk.num_alignments > 0) {
            printf("  alignment table (phoneme -> id -> samples):\n");
            print_alignment_table(chunk);
        }

        fwrite(chunk.samples, sizeof(float), chunk.num_samples, out);
        total_samples += chunk.num_samples;
        chunk_idx++;
    }

    fclose(out);

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    section("summary");
    if (rc == PIPER_DONE) {
        printf("  status:        OK (PIPER_DONE)\n");
        printf("  chunks:        %d\n",   chunk_idx);
        printf("  total_samples: %zu\n",  total_samples);
        if (sample_rate > 0) {
            printf("  total_audio:   %.3f s at %d Hz\n",
                   (double)total_samples / sample_rate, sample_rate);
        }
        printf("  output:        %s\n",   output_path);
        if (sample_rate > 0) {
            printf("\nPlay: aplay -r %d -c 1 -f FLOAT_LE -t raw %s\n",
                   sample_rate, output_path);
        }
    } else {
        fprintf(stderr, "FAIL: piper_synthesize_next returned %d\n", rc);
    }

    piper_free(synth);
    return rc == PIPER_DONE ? 0 : 1;
}
