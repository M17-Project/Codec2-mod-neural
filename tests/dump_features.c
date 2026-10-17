/* dump_features - write the 20 FARGAN features per 10 ms for a raw .c2 file (test helper) */
#include <stdio.h>
#include "../src/c2f_features.h"
int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s in.c2 out.f32\n", argv[0]); return 1; }
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) { perror("open"); return 1; }
    c2f_params_state st;
    c2f_params_init(&st);
    uint8_t bits[8];
    float feat[2][C2F_NB_FEATURES];
    while (fread(bits, 1, 8, fi) == 8) {
        c2f_frame_features(&st, feat, bits);
        fwrite(feat, sizeof(float), 2 * C2F_NB_FEATURES, fo);
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
