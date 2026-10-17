/*
 * c2feat - Codec2 3200 -> FARGAN conditioning features
 *
 * Runs the Codec2-mod 3200 encoder over 8 kHz s16 PCM, optionally flips
 * channel bits at a given BER, then dequantises/interpolates the parameters
 * exactly like codec2_decode() does and writes one 36-float row per 10 ms.
 * Optionally also writes the classic Codec2 decoder output (for reference
 * listening and for measuring the codec delay).
 *
 * Feature row layout (FARGAN expects 36 floats, first 20 used):
 *   0      loudness term (FARGAN derives its gain from this)
 *   1..10  decoded LSPs, deviation from uniform spacing, x4
 *   11     voicing bit (0/1)
 *   12..17 LPC cepstrum c1..c6 of the decoded all-pole envelope
 *   18     pitch, FARGAN encoding: log2(256/P16) - 1.5, P16 = period @16 kHz
 *   19     voicing as a "pitch correlation" proxy (+0.5 / -0.3)
 *   20..35 zero (LPC slot, unused by FARGAN training)
 *
 * Usage: c2feat in_8k.s16 out.f32 [ber] [seed] [classic_out_8k.s16] [bits_in.c2]
 *   If bits_in.c2 is given, the encoder is skipped and the 8-byte frames are
 *   read from that file instead (in_8k.s16 may then be "-").
 *
 * LGPL-2.1 (derivative of Codec2 / Codec2-mod).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "codec2_mod.h"
#include "quantise.h"
#include "interp.h"
#include "lpc.h"

#define NB_FEAT 36

static uint64_t rng_state = 88172645463325252ULL;
static double urand(void)
{
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (rng_state >> 11) * (1.0 / 9007199254740992.0);
}

static void lpc_cepstrum(const float *ak, float *c, int n)
{
    /* ak[0]=1, A(z)=sum ak[k] z^-k ; cepstrum of 1/A(z) */
    for (int m = 1; m <= n; m++) {
        double acc = (m <= LPC_ORD) ? -ak[m] : 0.0;
        for (int k = 1; k < m; k++)
            if (m - k <= LPC_ORD) acc -= ((double)k / m) * c[k - 1] * ak[m - k];
        c[m - 1] = (float)acc;
    }
}

static void write_row(FILE *f, float e, const float *lsp, int voiced, float Wo)
{
    float row[NB_FEAT] = {0};
    float ak[LPC_ORD + 1], cep[6];
    lsp_to_lpc((float *)lsp, ak);
    lpc_cepstrum(ak, cep, 6);

    /* loudness: 10*log10(e) mapped to roughly LPCNet c0 range */
    row[0] = 0.3f * (10.0f * log10f(e > 1e-3f ? e : 1e-3f) - 20.0f);
    for (int i = 0; i < LPC_ORD; i++)
        row[1 + i] = 4.0f * (lsp[i] - (i + 1) * (float)M_PI / (LPC_ORD + 1));
    row[11] = voiced ? 1.0f : 0.0f;
    for (int i = 0; i < 6; i++) row[12 + i] = cep[i];
    float p16 = 2.0f * TWO_PI / Wo;            /* period in 16 kHz samples */
    if (p16 < 32.0f) p16 = 32.0f;
    if (p16 > 254.0f) p16 = 254.0f;  /* FARGAN gather needs period <= 254 */
    row[18] = log2f(256.0f / p16) - 1.5f;
    row[19] = voiced ? 0.5f : -0.3f;
    fwrite(row, sizeof(float), NB_FEAT, f);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s in_8k.s16|- out.f32 [ber] [seed] [classic_out.s16|-] [bits_in.c2]\n", argv[0]);
        return 1;
    }
    double ber = argc > 3 ? atof(argv[3]) : 0.0;
    if (argc > 4) rng_state ^= strtoull(argv[4], NULL, 10) * 0x9E3779B97F4A7C15ULL + 1;
    FILE *fin = NULL, *fbits = NULL;
    if (argc > 6) fbits = fopen(argv[6], "rb");
    else fin = fopen(argv[1], "rb");
    FILE *fout = fopen(argv[2], "wb");
    FILE *fcl = (argc > 5 && strcmp(argv[5], "-")) ? fopen(argv[5], "wb") : NULL;
    if ((!fin && !fbits) || !fout) { perror("open"); return 1; }

    static codec2_t c2;
    codec2_init(&c2);

    /* parameter-decoder state, mirrors codec2_decoder_t */
    model_t prev_model = {0};
    prev_model.Wo = TWO_PI / P_MAX;
    prev_model.L = M_PI / prev_model.Wo;
    float prev_lsps[LPC_ORD], prev_e = 1.0f;
    for (int i = 0; i < LPC_ORD; i++) prev_lsps[i] = i * M_PI / (LPC_ORD + 1);

    int16_t speech[CODEC2_SAMPLES_PER_FRAME], dec[CODEC2_SAMPLES_PER_FRAME];
    uint8_t bits[CODEC2_BYTES_PER_FRAME];
    long frames = 0, flipped = 0;

    for (;;) {
        if (fbits) {
            if (fread(bits, 1, CODEC2_BYTES_PER_FRAME, fbits) != CODEC2_BYTES_PER_FRAME) break;
        } else {
            if (fread(speech, sizeof(int16_t), CODEC2_SAMPLES_PER_FRAME, fin) != CODEC2_SAMPLES_PER_FRAME) break;
            codec2_encode(&c2.encoder, bits, speech);
        }
        if (ber > 0)
            for (int b = 0; b < 64; b++)
                if (urand() < ber) { bits[b >> 3] ^= (uint8_t)(0x80 >> (b & 7)); flipped++; }

        if (fcl) {
            codec2_decode(&c2.decoder, dec, bits);
            fwrite(dec, sizeof(int16_t), CODEC2_SAMPLES_PER_FRAME, fcl);
        }

        /* --- same unpack/interp as codec2_decode() --- */
        model_t model[2] = {0};
        float lsps[2][LPC_ORD], e[2];
        int idx[LPC_ORD];
        unsigned int nbit = 0;
        model[0].voiced = unpack(bits, &nbit, 1);
        model[1].voiced = unpack(bits, &nbit, 1);
        model[1].Wo = decode_Wo(unpack(bits, &nbit, WO_BITS), WO_BITS);
        model[1].L = M_PI / model[1].Wo;
        e[1] = decode_energy(unpack(bits, &nbit, E_BITS), E_BITS);
        for (int i = 0; i < LSPD_SCALAR_INDEXES; i++) idx[i] = unpack(bits, &nbit, 5);
        decode_lspds_scalar(&lsps[1][0], idx);
        interp_Wo(&model[0], &prev_model, &model[1]);
        e[0] = interp_energy(prev_e, e[1]);
        interpolate_lsp(&lsps[0][0], prev_lsps, &lsps[1][0]);

        for (int i = 0; i < 2; i++)
            write_row(fout, e[i], lsps[i], model[i].voiced, model[i].Wo);

        prev_model = model[1];
        prev_e = e[1];
        memcpy(prev_lsps, lsps[1], sizeof(prev_lsps));
        frames++;
    }
    fprintf(stderr, "c2feat: %ld frames (%ld feature rows), %ld bits flipped\n", frames, 2 * frames, flipped);
    if (fin) fclose(fin);
    if (fbits) fclose(fbits);
    fclose(fout);
    if (fcl) fclose(fcl);
    return 0;
}
