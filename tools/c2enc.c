/* c2enc - encode 8 kHz s16 raw PCM into raw Codec2 3200 frames (8 bytes per 20 ms) */
#include <stdio.h>
#include "codec2_mod.h"

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s in_8k.raw out.c2\n", argv[0]); return 1; }
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) { perror("open"); return 1; }
    static codec2_t c2;
    codec2_init(&c2);
    int16_t s[CODEC2_SAMPLES_PER_FRAME];
    uint8_t b[CODEC2_BYTES_PER_FRAME];
    while (fread(s, 2, CODEC2_SAMPLES_PER_FRAME, fi) == CODEC2_SAMPLES_PER_FRAME) {
        codec2_encode(&c2.encoder, b, s);
        fwrite(b, 1, sizeof b, fo);
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
