#!/usr/bin/env python3
r"""
prep_opus.py - build a FARGAN training set the way Opus' dump_data -train does,
               but with Codec2 3200 parameters as the conditioning features.

Reproduces opus/dnn/dump_data.c (training mode), per sequence:
  * a random position in the raw 16 kHz s16 corpus (tts_speech_negative_16k.sw),
    SEQUENCE_LENGTH = 2000 frames (20 s)
  * 2nd-order high-pass (b=[1,-2,1], a=[1,-1.99599,0.996]) + random 2nd-order
    response (rand_resp: each of numerator/denominator is flat 2/3 of the time,
    else a random complex or real pole/zero pair)
  * speech gain 10^((-30 + U{0..39})/20), random polarity (50 %), x0.01 (5 %),
    silence (1 %), normalised by the weighted RMS to 3000 (int16 scale)
  * optional noise (--noise): 50 % of sequences, gain 10^((-40+U25+U15)/20)
    (x0.03 in 1/12 of those), own HP + random response
  * xn = x + n; clip to +-32767 (25 %); the TARGET IS xn - exactly what the
    features are computed from (Opus never asks the vocoder to denoise)
  * random sequence length between 1000 and 2000 frames
Then: xn -> 8 kHz -> Codec2-mod encoder -> parameter decoder -> 36-float rows
(c2feat), and the 16 kHz target delayed by the Codec2 delay, pre-emphasised 0.85.

Output: <out>/features.f32, <out>/target.s16 (for opus/dnn/torch/fargan/train_fargan.py)
"""
import argparse, os, sys, tempfile, subprocess, shutil, glob
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))
C2FEAT = os.path.join(HERE, 'c2feat')
NB_FEAT = 36
CODEC_DELAY_8K = 160            # classic Codec2 3200 output lags its input by 20 ms
FRAME = 160                     # 10 ms at 16 kHz
SEQ_FRAMES = 2000               # dump_data SEQUENCE_LENGTH
HP_B, HP_A = [1., -2., 1.], [1., -1.99599, 0.99600]
W_B, W_A = [1., -2., 1.], [1., -1.89, 0.895]   # weighted_rms filter


def rand_filt(rng):
    """dump_data.c rand_filt(): returns [c1, c2] of 1 + c1 z^-1 + c2 z^-2"""
    if rng.integers(3) != 0:
        return [0., 0.]
    if rng.random() - .5 > 0:
        r = rng.random(); r = .7 * r * r
        th = rng.random(); th = np.pi * th * th
        return [-2 * r * np.cos(th), r * r]
    r0, r1 = 1.4 * (rng.random() - .5), 1.4 * (rng.random() - .5)
    return [-r0 - r1, r0 * r1]


def rand_resp(rng):
    a, b = rand_filt(rng), rand_filt(rng)
    return [1.] + b, [1.] + a


def weighted_rms(x):
    y = ss.lfilter(W_B, W_A, x)
    return 0.9506 * np.sqrt((1e-15 + np.sum(y * y)) / len(x))


_W = {}


def _init(corpus, noise, opt, shard_dir):
    _W['speech'] = np.memmap(corpus, dtype=np.int16, mode='r')
    _W['noise'] = np.memmap(noise, dtype=np.int16, mode='r') if noise else None
    _W['opt'], _W['dir'] = opt, shard_dir


def job(k):
    opt = _W['opt']
    rng = np.random.default_rng(opt.seed * 1000003 + k)
    S = _W['speech']
    n_samp = SEQ_FRAMES * FRAME
    pos = int(rng.integers(0, len(S) - n_samp))
    x = np.asarray(S[pos:pos + n_samp], dtype=np.float64)

    # --- gains, exactly as dump_data.c ---
    g = 10 ** ((-30 + int(rng.integers(40))) / 20)
    if rng.integers(2): g = -g
    if rng.integers(20) == 0: g *= .01
    if rng.integers(100) == 0: g = 0.
    ng = 10 ** ((-40 + 25 * rng.random() + 15 * rng.random()) / 20)
    if rng.integers(2) != 0: ng = 0.
    if rng.integers(12) == 0: ng *= 0.03
    ng *= g
    b_noise, a_noise = rand_resp(rng)
    b_sig, a_sig = rand_resp(rng)

    x = ss.lfilter(HP_B, HP_A, x)
    x = ss.lfilter(b_sig, a_sig, x)
    x *= g * 3000. / (1 + weighted_rms(x))

    xn = x
    N = _W['noise']
    if N is not None and ng != 0:
        npos = int(rng.integers(0, len(N) - n_samp))
        nz = np.asarray(N[npos:npos + n_samp], dtype=np.float64)
        nz = ss.lfilter(HP_B, HP_A, nz)
        nz = ss.lfilter(b_noise, a_noise, nz)
        nz *= ng * 3000. / (1 + weighted_rms(nz))
        xn = x + nz
    if rng.integers(4) == 0:
        xn = np.clip(xn, -32767., 32767.)
    xn = np.clip(np.floor(.5 + xn), -32767, 32767)          # int16, input == target

    seq = min(SEQ_FRAMES, SEQ_FRAMES // 2 + int(rng.integers(SEQ_FRAMES // 2 + 1)))
    xn = xn[:seq * FRAME] / 32768.

    # --- Codec2 features from the 8 kHz version of xn ---
    x8 = ss.resample_poly(xn, 1, 2)
    with tempfile.TemporaryDirectory(dir=opt.tmp) as td:
        fin, ffe = os.path.join(td, 'in.s16'), os.path.join(td, 'f.f32')
        np.clip(np.round(x8 * 32768), -32768, 32767).astype(np.int16).tofile(fin)
        subprocess.run([C2FEAT, fin, ffe, '0', str(k)], check=True, stderr=subprocess.DEVNULL)
        feat = np.fromfile(ffe, dtype=np.float32).reshape(-1, NB_FEAT)

    # --- target: same signal at 16 kHz, delayed by the codec, pre-emphasised ---
    tgt = np.concatenate([np.zeros(2 * CODEC_DELAY_8K), xn])
    nrows = min(len(feat), len(tgt) // FRAME)
    tgt = ss.lfilter([1, -0.85], [1], tgt[:nrows * FRAME])
    tgt = np.clip(np.round(tgt * 32768), -32768, 32767).astype(np.int16)
    feat = feat[:nrows]

    base = os.path.join(_W['dir'], f'shard{os.getpid()}')
    with open(base + '.s16', 'ab') as f:
        tgt.tofile(f)
    with open(base + '.f32', 'ab') as f:
        feat.tofile(f)
    return nrows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('corpus', help='raw 16 kHz s16 speech, e.g. tts_speech_negative_16k.sw')
    ap.add_argument('out', help='output directory')
    ap.add_argument('--sequences', type=int, default=20000, help='number of 20 s sequences (dump_data: 20000)')
    ap.add_argument('--noise', default=None, help='raw 16 kHz s16 noise file (dump_data -train <noise>); '
                                                  'noise goes into input AND target, as in Opus')
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tmp', default=None, help='scratch dir (fast local disk)')
    opt = ap.parse_args()

    if not os.access(C2FEAT, os.X_OK):
        sys.exit(f'{C2FEAT} not found - run make first')
    for p in [opt.corpus] + ([opt.noise] if opt.noise else []):
        if not os.path.isfile(p):
            sys.exit(f'error: {p} not found')
    hours_in = os.path.getsize(opt.corpus) / 2 / 16000 / 3600
    print(f'corpus {opt.corpus}: {hours_in:.1f} h; {opt.sequences} sequences of up to 20 s; '
          f'noise: {opt.noise or "off"}; {opt.jobs} workers', flush=True)
    os.makedirs(opt.out, exist_ok=True)

    shard_dir = tempfile.mkdtemp(prefix='prep_opus_', dir=opt.tmp)
    rows = 0
    try:
        with Pool(opt.jobs, initializer=_init, initargs=(opt.corpus, opt.noise, opt, shard_dir)) as pool:
            for k, n in enumerate(pool.imap_unordered(job, range(opt.sequences), chunksize=8)):
                rows += n
                if k % 500 == 0:
                    print(f'  {k+1}/{opt.sequences}  {rows/360000:.1f} h', flush=True)
        total = 0
        with open(os.path.join(opt.out, 'features.f32'), 'wb') as ff, \
             open(os.path.join(opt.out, 'target.s16'), 'wb') as ft:
            for b in sorted(glob.glob(os.path.join(shard_dir, '*.f32'))):
                b = b[:-4]
                nf = os.path.getsize(b + '.f32') // (NB_FEAT * 4)
                nt = os.path.getsize(b + '.s16') // (FRAME * 2)
                if nf != nt:
                    sys.exit(f'error: shard {b} misaligned ({nf} vs {nt})')
                for ext, dst in (('.f32', ff), ('.s16', ft)):
                    with open(b + ext, 'rb') as src:
                        shutil.copyfileobj(src, dst, 16 << 20)
                total += nf
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
    if total == 0:
        sys.exit('error: no data produced')
    print(f'done: {total} frames = {total/360000:.1f} h -> {opt.out}')


if __name__ == '__main__':
    main()
