#!/usr/bin/env bash
# 0_setup.sh - run on EVERY new pod (the container disk starts empty, /workspace persists)
# Expects this repository at $W/Codec2-mod-neural. Clones Codec2-mod and Opus next to it.
set -euo pipefail
W=${W:-/workspace}
C2F=$(cd "$(dirname "$0")/.." && pwd)          # .../Codec2-mod-neural/training
OPUS_COMMIT=${OPUS_COMMIT:-503d81b138d76621aae4b12786e90de48aa8db3a}   # the commit the released model was trained with
cd "$W"
apt-get update -qq
apt-get install -y -qq build-essential git libsndfile1 sox ffmpeg tmux wget aria2 >/dev/null
pip install -q soundfile scipy tqdm numpy matplotlib pyyaml   # keep the pod's own torch (no torchaudio)
[ -d Codec2-mod ] || git clone -q https://github.com/M17-Project/Codec2-mod.git
if [ ! -d opus ]; then
  git init -q opus
  git -C opus remote add origin https://github.com/xiph/opus.git
  git -C opus fetch -q --depth 1 origin "$OPUS_COMMIT"
  git -C opus checkout -q FETCH_HEAD
fi
echo "opus at $(git -C opus rev-parse HEAD)"
python3 - <<'PY'
import re
F = 'opus/dnn/torch/fargan/'
s = open(F + 'stft_loss.py').read()
open(F + 'stft_loss.py', 'w').write(re.sub(r'^import torchaudio', '# import torchaudio  (unused)', s, flags=re.M))
# stage-2 discriminator only needs torchaudio.functional.amplitude_to_DB -> pure-torch fallback
D = 'opus/dnn/torch/osce/models/fd_discriminator.py'
s = open(D).read()
if 'amplitude_to_DB fallback' not in s:
    s = re.sub(r'^import torchaudio\n', '''try:
    import torchaudio
except ImportError:  # amplitude_to_DB fallback (same maths as torchaudio)
    import types
    def _a2db(x, multiplier, amin, db_multiplier, top_db=None):
        x_db = multiplier * torch.log10(torch.clamp(x, min=amin)) - multiplier * db_multiplier
        if top_db is not None:
            sh = x_db.size(); c = sh[-3] if x_db.dim() > 2 else 1
            x_db = x_db.reshape(-1, c, sh[-2], sh[-1])
            x_db = torch.max(x_db, (x_db.amax(dim=(-3, -2, -1)) - top_db).view(-1, 1, 1, 1)).reshape(sh)
        return x_db
    torchaudio = types.SimpleNamespace(functional=types.SimpleNamespace(amplitude_to_DB=_a2db))
''', s, count=1, flags=re.M)
    open(D, 'w').write(s)
# stage-2 resume: --start-epoch N continues from fargan_adv_N.pth (generator + discriminator;
# lr is constant in stage 2, only the Adam moments restart)
A = F + 'adv_train_fargan.py'
s = open(A).read()
if '--start-epoch' not in s:
    s = s.replace("training_group.add_argument('--epochs'",
                  "training_group.add_argument('--start-epoch', type=int, default=0, help='resume after this adv epoch')\n"
                  "training_group.add_argument('--epochs'", 1)
    s = s.replace("checkpoint['state_dict']    = model.state_dict()",
                  "if args.start_epoch > 0 and 'disc_sate_dict' in checkpoint:\n"
                  "    disc.load_state_dict(checkpoint['disc_sate_dict']); print('resumed discriminator')\n"
                  "checkpoint['state_dict']    = model.state_dict()", 1)
    s = s.replace("for epoch in range(1, epochs + 1):", "for epoch in range(args.start_epoch + 1, epochs + 1):", 1)
    s = s.replace("if epoch == 1 and i == 400:",
                  "if (epoch == 1 and i == 400) or (args.start_epoch > 0 and epoch == args.start_epoch + 1 and i == 0):", 1)
    assert s.count('start_epoch') == 4, 'adv_train_fargan.py resume patch did not apply'
    open(A, 'w').write(s)
for f in ('train_fargan.py', 'adv_train_fargan.py'):
    s = open(F + f).read()
    # more loader workers + 'fork' (Python >= 3.14 defaults to forkserver, which copies the dataset per worker)
    s = re.sub(r"num_workers=\d+(, multiprocessing_context='fork')?(, prefetch_factor=\d+, persistent_workers=True)?",
               "num_workers=16, multiprocessing_context='fork'", s)
    open(F + f, 'w').write(s)
    print(f, '->', re.search(r'DataLoader\(.*\)', s).group(0))
PY
make -s -C "$C2F" C2MOD="$W/Codec2-mod"
lscpu | grep 'Model name' | sed 's/^/CPU: /'
python3 -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
echo "setup OK"
