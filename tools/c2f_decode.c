/*
 * c2f_decode - decode a raw Codec2 3200 file (8 bytes per 20 ms frame) with FARGAN
 *
 *   c2f_decode in.c2 out.wav        16 kHz mono WAV
 *   c2f_decode in.c2 out.raw        16 kHz s16 raw (any name not ending in .wav)
 *   c2f_decode in.c2 out.wav -b     also print the decoding speed
 *
 * Use "-" for stdin or stdout (raw only on stdout).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "c2fargan.h"

static void put32(FILE *f, uint32_t v) { for (int i = 0; i < 4; i++) fputc((v >> (8 * i)) & 0xff, f); }
static void put16(FILE *f, uint16_t v) { fputc(v & 0xff, f); fputc(v >> 8, f); }

static void wav_header(FILE *f, uint32_t nsamples)
{
    fwrite("RIFF", 1, 4, f); put32(f, 36 + 2 * nsamples); fwrite("WAVEfmt ", 1, 8, f);
    put32(f, 16); put16(f, 1); put16(f, 1); put32(f, C2F_SAMPLE_RATE); put32(f, 2 * C2F_SAMPLE_RATE);
    put16(f, 2); put16(f, 16); fwrite("data", 1, 4, f); put32(f, 2 * nsamples);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s in.c2 out.wav|out.raw [-b]\n", argv[0]);
        return 1;
    }
    int bench = argc > 3 && !strcmp(argv[3], "-b");
    size_t n = strlen(argv[2]);
    int wav = n > 4 && !strcmp(argv[2] + n - 4, ".wav");
    FILE *fi = strcmp(argv[1], "-") ? fopen(argv[1], "rb") : stdin;
    FILE *fo = strcmp(argv[2], "-") ? fopen(argv[2], "wb") : stdout;
    if (!fi || !fo) { perror("open"); return 1; }
    if (wav) wav_header(fo, 0);

    c2f_decoder *dec = c2f_create();
    uint8_t bits[C2F_BYTES_PER_FRAME];
    int16_t pcm[C2F_SAMPLES_PER_FRAME];
    uint32_t frames = 0;
    clock_t t0 = clock();
    while (fread(bits, 1, sizeof(bits), fi) == sizeof(bits)) {
        c2f_decode(dec, pcm, bits);
        fwrite(pcm, sizeof(int16_t), C2F_SAMPLES_PER_FRAME, fo);
        frames++;
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    if (wav) { fseek(fo, 0, SEEK_SET); wav_header(fo, frames * C2F_SAMPLES_PER_FRAME); }
    if (bench && frames)
        fprintf(stderr, "%u frames (%.1f s of audio) in %.2f s CPU: %.1f%% of real time, %.2f ms per frame\n",
                frames, frames * 0.02, secs, 100.0 * secs / (frames * 0.02), 1000.0 * secs / frames);
    c2f_destroy(dec);
    if (fi != stdin) fclose(fi);
    if (fo != stdout) fclose(fo);
    return 0;
}
