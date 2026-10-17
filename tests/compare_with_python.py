#!/usr/bin/env python3
"""
compare_with_python.py - check the C decoder against the PyTorch reference

  python3 tests/compare_with_python.py model.pth speech.c2 [build_dir]

Decodes the same Codec2 file with build/c2f_decode (C) and training/synth.py
(PyTorch, the model the C weights were exported from) and reports
  lag   must be 160 samples (the C decoder runs 10 ms behind synth.py)
  LSD   log spectral distance, expect 0.5 to 1.5 dB on speech
  SNR   waveform SNR, for information only

FARGAN is autoregressive, so tiny numerical differences grow into different
waveforms with the same spectrum. The PyTorch model also adds a small amount
of noise inside the network even at inference. That noise alone changes the
waveform SNR to between 7 and 22 dB on speech, so SNR says little here. A wrong
weight, feature or timing shows up as an LSD of several dB.
"""
import os, subprocess, sys, tempfile
import numpy as np
import scipy.signal as ss
import soundfile as sf

if len(sys.argv) < 3:
    sys.exit(__doc__)
model, c2 = sys.argv[1], sys.argv[2]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build = sys.argv[3] if len(sys.argv) > 3 else os.path.join(root, 'build')

with tempfile.TemporaryDirectory() as td:
    fc, fp = os.path.join(td, 'c.wav'), os.path.join(td, 'py.wav')
    subprocess.run([os.path.join(build, 'c2f_decode'), c2, fc], check=True)
    subprocess.run([sys.executable, os.path.join(root, 'training', 'synth.py'), model, c2, fp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    c, _ = sf.read(fc)
    p, _ = sf.read(fp)

seg = slice(8000, min(len(c), len(p)) - 1000)
best = max(range(0, 481), key=lambda l: np.dot(c[seg.start + l:seg.stop + l][:len(p[seg])], p[seg][:len(c) - seg.start - l]))
a = c[best:]
n = min(len(a), len(p))
a, b = a[16000:n], p[16000:n]          # skip the first second (start-up differs)
corr = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))
snr = 10 * np.log10(np.sum(b ** 2) / np.sum((a - b) ** 2))
_, _, A = ss.stft(a, 16000, nperseg=512)
_, _, B = ss.stft(b, 16000, nperseg=512)
LA, LB = 20 * np.log10(np.abs(A) + 1e-5), 20 * np.log10(np.abs(B) + 1e-5)
act = LB.max(0) > LB.max() - 50
lsd = np.mean(np.sqrt(np.mean((LA - LB)[:, act] ** 2, 0)))
print(f'lag {best} samples, LSD {lsd:.2f} dB, SNR {snr:.1f} dB, corr {corr:.4f}')
ok = best == 160 and lsd < 2.0
print('OK' if ok else 'MISMATCH')
sys.exit(0 if ok else 1)
