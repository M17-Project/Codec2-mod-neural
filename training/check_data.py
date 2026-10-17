#!/usr/bin/env python3
"""
check_data.py - sanity-check FARGAN training data built by prep.py / 1_data.sh

  python3 check_data.py DIR [DIR ...]       (each DIR holds features.f32 + target.s16)
  e.g. python3 check_data.py /workspace/data/parts/* /workspace/data

For each set it reports
  * hours, and whether features and targets have the same number of frames
  * ALIGNMENT: correlation between the loudness feature (row[0], from the Codec2
    bitstream) and the measured energy of the matching 10 ms target frame, at
    lags -3..+3 frames. Healthy data peaks at lag 0 or +-1 with r ~ 0.75-0.95.
    A peak elsewhere, or r < 0.5, means features and audio do not belong together.
  * PITCH: median decoded pitch (Hz) - adult speech is ~85-255 Hz
  * listen/<name>.wav : 10 s of the target audio (de-emphasised, 16 kHz) to listen to
"""
import os, sys
import numpy as np
import scipy.signal as ss

NB = 36


def check(d, out_dir, rng):
    ff, ft = os.path.join(d, 'features.f32'), os.path.join(d, 'target.s16')
    if not (os.path.isfile(ff) and os.path.isfile(ft)):
        return
    nf = os.path.getsize(ff) // (NB * 4)
    nt = os.path.getsize(ft) // (160 * 2)
    name = os.path.basename(os.path.normpath(d))
    F = np.memmap(ff, dtype=np.float32, mode='r', shape=(nf, NB))
    T = np.memmap(ft, dtype=np.int16, mode='r', shape=(nt, 160))
    n = min(nf, nt)
    status = 'OK' if nf == nt else f'MISMATCH ({nf} feature rows vs {nt} target frames)'

    # 20 random windows of 3000 frames (30 s) each
    W, lags = 3000, range(-3, 4)
    rs = {l: [] for l in lags}
    pitches = []
    for _ in range(20):
        s = int(rng.integers(10, max(11, n - W - 10)))
        f = np.asarray(F[s:s + W])
        t = np.asarray(T[s - 3:s + W + 3], dtype=np.float32).reshape(-1) / 32768
        t = ss.lfilter([1.], [1., -.85], t).reshape(-1, 160)
        e = 10 * np.log10(np.mean(t ** 2, axis=1) + 1e-10)          # frame energy, dB
        loud = f[:, 0]
        voiced = f[:, 11] > 0.5
        p16 = 256.0 / 2 ** (f[:, 18] + 1.5)
        pitches += list(16000.0 / p16[voiced])
        for l in lags:
            ee = e[3 + l:3 + l + W]
            m = ee > -60                                              # skip digital silence
            if m.sum() > 100:
                rs[l].append(np.corrcoef(loud[m], ee[m])[0, 1])
    r = {l: float(np.nanmean(v)) if v else float('nan') for l, v in rs.items()}
    best = max(r, key=lambda l: (r[l] if r[l] == r[l] else -9))
    pitch = float(np.median(pitches)) if pitches else float('nan')
    align = 'OK' if abs(best) <= 1 and r[0] > 0.5 else 'BAD'   # +-1 frame is normal (20 ms energy interpolation)
    print(f'{name:14s} {n/360000:7.1f} h  frames:{status:4s}  align:{align} '
          f'(r@0={r[0]:.2f}, best lag {best:+d} r={r[best]:.2f})  median pitch {pitch:5.0f} Hz')

    # 10 s of target audio to listen to
    s = int(rng.integers(0, max(1, n - 1000)))
    t = np.asarray(T[s:s + 1000], dtype=np.float32).reshape(-1) / 32768
    t = ss.lfilter([1.], [1., -.85], t)
    try:
        import soundfile as sf
        os.makedirs(out_dir, exist_ok=True)
        sf.write(os.path.join(out_dir, f'{name}.wav'), np.clip(t, -.99, .99), 16000, subtype='PCM_16')
    except Exception as ex:
        print('   (could not write listening sample:', ex, ')')


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rng = np.random.default_rng(0)
    out = os.path.join(os.getcwd(), 'listen')
    for d in sys.argv[1:]:
        check(d, out, rng)
    print(f'listening samples (target audio, 16 kHz): {out}/')


if __name__ == '__main__':
    main()
