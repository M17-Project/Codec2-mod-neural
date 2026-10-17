# Training

This folder reproduces the released model. It follows the FARGAN recipe from
Opus (opus/dnn/dump_data.c in training mode and opus/dnn/torch/fargan/README.md).
The only change is that the conditioning features come from Codec2 3200
instead of LPCNet.

| | Opus | here |
|---|---|---|
| corpus | tts_speech_negative_16k.sw (media.xiph.org) | same |
| sampling | 20000 random sequences of 10 to 20 s | same |
| augmentation | high pass, random filter, random gain, polarity, clipping, optional noise | same, the target is always the exact signal the features come from |
| features | LPCNet | Codec2 3200 decoded parameters (c2feat) |
| stage 1 | train_fargan.py, 400 epochs, batch 4096, lr 0.002 | same |
| stage 2 | adv_train_fargan.py, 50 epochs, batch 160, lr 2e-6, reg weight 5 | same |

Lesson from earlier attempts: never train with noisy features and a clean
target. The vocoder cannot denoise and the result sounds like a pipe.

## Files

| file | what |
|---|---|
| c2feat.c, Makefile | Codec2-mod encoder plus parameter decoder, writes the features. Must stay identical to src/c2f_features.c |
| prep_opus.py | builds the training set the way Opus dump_data does |
| check_data.py | hours, feature and audio alignment, pitch sanity, listening samples |
| synth.py | decodes a .c2 or speech file with a .pth checkpoint (PyTorch reference) |
| compare_adv.sh | decodes one file with several checkpoints for listening |
| export_c.sh | writes model/fargan_data.c and .h from a checkpoint |
| runpod/0_setup.sh to 3_adv.sh | the cloud training pipeline |

## RunPod

Pod: PyTorch template, network volume at /workspace, container disk of at least
60 GB. The training loop is limited by single thread CPU speed, so pick a host
with a fast desktop class CPU (Ryzen 7950X or 9950X). Stage 2 needs more than
24 GB of GPU memory at batch 160 (32 GB is enough).

    cd /workspace
    git clone https://github.com/M17-Project/Codec2-mod-neural.git
    bash Codec2-mod-neural/training/runpod/0_setup.sh    # on every new pod
    tmux new -s work
    bash /workspace/Codec2-mod-neural/training/runpod/1_data.sh    # corpus and training set
    bash /workspace/Codec2-mod-neural/training/runpod/2_train.sh   # stage 1
    bash /workspace/Codec2-mod-neural/training/runpod/3_adv.sh     # stage 2

0_setup.sh clones Codec2-mod and Opus next to the repository. Opus is pinned to
the commit the released model was trained with (OPUS_COMMIT can override it).
It also patches the Opus training scripts: more data loader workers, no
torchaudio dependency and a resume option for stage 2.

Options are environment variables:

* 1_data.sh: SEQUENCES=20000, NOISE_FILE=/path/noise16k.sw (off by default)
* 2_train.sh: EPOCHS=400 BATCH=4096 LR=0.002, RESUME=N continues after epoch N
* 3_adv.sh: EPOCHS=50 BATCH=160, INIT=checkpoint, RESUME=N continues after adv epoch N

Checkpoints go to /workspace/runs/opus_v1/checkpoints. The container disk is
wiped when the pod stops. 2_train.sh copies the training data back
automatically. For 3_adv.sh copy it by hand:

    mkdir -p /data/opus && cp /workspace/data/opus/{features.f32,target.s16} /data/opus/

## What the released model took

* training set: 30 million frames (83 hours) built from the 409 hour corpus
* stage 1 on a Ryzen 7950X pod: about 2 minutes per epoch, 14.5 hours in total
* stage 2 on an RTX 5090 with a Ryzen 9950X: about 44 minutes per epoch, 37 hours in total
* stage 2 trains only the discriminator for the first 400 batches, so the first minutes look faster than the rest

## Listening and export

    python3 synth.py fargan_adv_50.pth speech.wav out.wav    # out.wav and out_classic.wav
    bash compare_adv.sh speech.wav                           # adv 30, 40 and 50 side by side
    bash export_c.sh fargan_adv_50.pth                       # new model/fargan_data.c

After an export rebuild the library and run tests/compare_with_python.py.
