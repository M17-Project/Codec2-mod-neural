#!/usr/bin/env python3
"""
synth.py - decode Codec2 3200 with a trained FARGAN model.

  python3 synth.py model.pth input output.wav [--ber 0.01]

input is either a raw Codec2 3200 bitstream (*.c2, 8 bytes per 20 ms) or
speech (wav/flac/raw 8 kHz s16), which is encoded with Codec2-mod first.
Output is 16 kHz.  A classic Codec2 decode is also written next to it
(<output>_classic.wav, 8 kHz) for A/B listening.
"""
import argparse, os, sys, subprocess, tempfile
import numpy as np
import scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))
_cands = [os.path.join(HERE, *up, 'opus', 'dnn', 'torch', 'fargan') for up in (('..', '..'), ('..',))]
OPUS_FARGAN = os.environ.get('FARGAN_DIR') or next((c for c in _cands if os.path.isdir(c)), _cands[0])
sys.path.insert(0, OPUS_FARGAN)
import torch, fargan  # noqa: E402
import soundfile as sf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('input')
ap.add_argument('output')
ap.add_argument('--ber', type=float, default=0.0, help='simulate channel bit errors')
opt = ap.parse_args()

c2feat = os.path.join(HERE, 'c2feat')
with tempfile.TemporaryDirectory() as td:
    ffe, fcl = os.path.join(td, 'f.f32'), os.path.join(td, 'cl.s16')
    if opt.input.endswith('.c2'):
        subprocess.run([c2feat, '-', ffe, str(opt.ber), '1', fcl, opt.input], check=True)
    else:
        if opt.input.endswith(('.raw', '.s16', '.sw')):
            x8 = np.fromfile(opt.input, dtype=np.int16)
        else:
            x, fs = sf.read(opt.input, dtype='float32', always_2d=True)
            x = x.mean(1)
            g = np.gcd(int(fs), 8000)
            x8 = ss.resample_poly(x, 8000 // g, int(fs) // g)
            x8 = np.clip(np.round(x8 * 32768), -32768, 32767).astype(np.int16)
        fin = os.path.join(td, 'in.s16'); x8.tofile(fin)
        subprocess.run([c2feat, fin, ffe, str(opt.ber), '1', fcl], check=True)
    feat = np.fromfile(ffe, dtype=np.float32).reshape(1, -1, 36)
    classic = np.fromfile(fcl, dtype=np.int16)

ck = torch.load(opt.model, map_location='cpu', weights_only=False)
model = fargan.FARGAN(*ck['model_args'], **ck['model_kwargs'])
model.load_state_dict(ck['state_dict'], strict=False)
model.eval()

f = feat[:, :, :20]
periods = np.round(np.clip(256. / 2 ** (f[:, :, 18] + 1.5), 32, 254)).astype('int')
with torch.no_grad():
    sig, _ = model(torch.tensor(f), torch.tensor(periods), f.shape[1] - 4)
sig = sig.numpy().flatten()
sig = ss.lfilter([1.], [1., -.85], sig)          # undo training pre-emphasis
sig = np.concatenate([np.zeros(3 * 160), sig])   # FARGAN output starts at feature frame 3
sf.write(opt.output, np.clip(sig, -.99, .99), 16000, subtype='PCM_16')
base = os.path.splitext(opt.output)[0]
sf.write(base + '_classic.wav', classic, 8000, subtype='PCM_16')
print(f'wrote {opt.output} (FARGAN, 16 kHz) and {base}_classic.wav (classic Codec2, 8 kHz)')
