#!/usr/bin/env python3
"""
Procedural sound design for FischVogel's End Boss - "a soul trapped in a block".

Everything is synthesised from scratch (no samples), so the pack carries no
third-party audio:
  * a small cascade formant synthesiser (Klatt style) speaks phoneme strings,
    voiced or whispered - whispers that *almost* form words ("let me out",
    "help me", "so cold", "free us"), a chant ("we are still here"),
  * high, raspy formant voices with pitch glides for wails and shrieks,
  * crystal "glass" partials, heartbeat thumps, buzzing laser tones,
  * a simple exponential-noise reverb, plus reversed reverb pre-swells.

Output: mono Ogg Vorbis (Minecraft only positions mono sounds) at 44.1 kHz,
written to  <out>/assets/fv_endboss/sounds/...  plus a sounds.json and
lang/en_us.json next to them.

    python3 make_sounds.py <pack_root>       (pack_root = .../Nexo/pack)

Deterministic: fixed random seed, so re-running gives identical audio.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
RNG = np.random.default_rng(20261001)


# --------------------------------------------------------------------------
#  basic helpers
# --------------------------------------------------------------------------
def n_of(sec):
    return int(round(sec * SR))


def noise(n):
    return RNG.standard_normal(n)


def lowpass(x, fc, order=2):
    b, a = signal.butter(order, min(fc / (SR / 2), 0.99), 'low')
    return signal.lfilter(b, a, x)


def highpass(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), 'high')
    return signal.lfilter(b, a, x)


def bandpass(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], 'band')
    return signal.lfilter(b, a, x)


def smooth(x, sec):
    w = max(1, n_of(sec))
    k = np.hanning(w * 2 + 1)
    k /= k.sum()
    pad = np.pad(x, (w, w), mode='edge')
    return np.convolve(pad, k, mode='same')[w:-w]


def fade(x, fin=0.01, fout=0.05):
    x = x.copy()
    a, b = n_of(fin), n_of(fout)
    if a:
        x[:a] *= np.linspace(0, 1, a) ** 2
    if b:
        x[-b:] *= np.linspace(1, 0, b) ** 2
    return x


def pad_to(x, n):
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x))])


def mix(*parts):
    """mix([(signal, offset_sec, gain), ...])"""
    end = max(n_of(o) + len(s) for s, o, g in parts)
    out = np.zeros(end)
    for s, o, g in parts:
        i = n_of(o)
        out[i:i + len(s)] += g * s
    return out


def softclip(x, drive):
    return np.tanh(drive * x) / np.tanh(drive)


def trim_tail(x, floor_db=-58.0, keep=0.08):
    """Cut the inaudible end of a reverb tail (below floor_db of the peak)."""
    env = smooth(np.abs(x), 0.02)
    thr = np.max(env) * 10 ** (floor_db / 20)
    idx = np.nonzero(env > thr)[0]
    if len(idx) == 0:
        return x
    end = min(len(x), idx[-1] + n_of(keep))
    y = x[:end].copy()
    k = min(len(y), n_of(keep))
    y[-k:] *= np.linspace(1, 0, k)
    return y


def normalize(x, peak_db=-1.0, rms_db=None):
    x = trim_tail(x)
    x = x - np.mean(x)
    if rms_db is not None:
        rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        x = x * (10 ** (rms_db / 20) / rms)
    pk = np.max(np.abs(x)) + 1e-12
    lim = 10 ** (peak_db / 20)
    if pk > lim:
        x = x * (lim / pk)
    return x


# --------------------------------------------------------------------------
#  reverb: exponentially decaying, darkening noise tail
# --------------------------------------------------------------------------
def make_ir(decay=0.45, length=1.6, bright=6000.0, predelay=0.012):
    n = n_of(length)
    t = np.arange(n) / SR
    ir = noise(n) * np.exp(-t / decay)
    # darker as it decays: two filtered copies cross-faded
    dark = lowpass(ir, bright * 0.35)
    brt = lowpass(ir, bright)
    w = np.exp(-t / (decay * 0.6))
    ir = brt * w + dark * (1 - w)
    ir = np.concatenate([np.zeros(n_of(predelay)), ir])
    ir /= np.sqrt(np.sum(ir ** 2))
    return ir


def reverb(x, wet=0.3, decay=0.45, length=1.6, bright=6000.0):
    ir = make_ir(decay, length, bright)
    y = signal.fftconvolve(x, ir)[: len(x) + len(ir) - 1]
    dry = pad_to(x, len(y))
    return (1 - wet) * dry + wet * y * 1.2


def reverse_swell(x, decay=0.5, length=1.4):
    """Reverb the sound, reverse it: a ghostly swell that leads into it."""
    ir = make_ir(decay, length, 5000.0)
    y = signal.fftconvolve(x, ir)
    return y[::-1]


# --------------------------------------------------------------------------
#  formant synthesis
# --------------------------------------------------------------------------
#  formants (F1..F5, Hz) - roughly an adult female / child voice
VOWELS = {
    'a': (850, 1220, 2810, 3800, 4600),
    'e': (600, 2200, 2900, 3900, 4600),
    'i': (310, 2790, 3310, 4100, 4800),
    'o': (600, 900, 2800, 3700, 4500),
    'u': (370, 950, 2670, 3600, 4500),
    '@': (550, 1650, 2700, 3700, 4500),   # schwa
    'l': (350, 1100, 2700, 3700, 4500),
    'r': (450, 1200, 1650, 3500, 4400),
    'm': (260, 1000, 2200, 3500, 4400),
    'n': (260, 1700, 2600, 3600, 4400),
    'w': (330, 750, 2400, 3500, 4400),
}
BW = (90, 110, 160, 220, 280)
# relative loudness of each sound
LOUD = {'a': 1.0, 'e': 0.95, 'i': 0.85, 'o': 1.0, 'u': 0.85, '@': 0.8,
        'l': 0.55, 'r': 0.6, 'm': 0.3, 'n': 0.3, 'w': 0.55, 'h': 0.35}
FRIC = {  # band (Hz) and gain of fricative noise
    's': (4200, 9500, 0.55), 'S': (2000, 5500, 0.6), 'f': (1500, 8000, 0.25),
    'z': (4200, 9500, 0.3), 'v': (1500, 6000, 0.2),
}
STOPS = {  # burst band, gain, closure length
    't': (3000, 9000, 0.7, 0.045), 'k': (1500, 4000, 0.7, 0.05),
    'p': (300, 3000, 0.5, 0.05), 'd': (2500, 7000, 0.45, 0.035),
    'g': (1200, 3500, 0.45, 0.04), 'b': (200, 2500, 0.35, 0.04),
}


def cascade(x, F, B, block=128):
    """Time-varying cascade of 2-pole resonators, unity gain at DC."""
    out = x
    for j in range(F.shape[1]):
        res = np.empty_like(out)
        zi = np.zeros(2)
        for s in range(0, len(out), block):
            e = min(s + block, len(out))
            f = float(F[s:e, j].mean())
            bw = float(B[s:e, j].mean())
            f = min(f, SR * 0.45)
            r = np.exp(-np.pi * bw / SR)
            th = 2 * np.pi * f / SR
            a1, a2 = -2 * r * np.cos(th), r * r
            b0 = 1 + a1 + a2
            res[s:e], zi = signal.lfilter([b0], [1, a1, a2], out[s:e], zi=zi)
        out = res
    return out


def glottal(f0, breath=0.05, tilt=1.0):
    """Band-limited sawtooth-like voicing for a per-sample f0 track."""
    n = len(f0)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    src = np.zeros(n)
    kmax = int(SR * 0.45 / max(60.0, f0.min()))
    for k in range(1, kmax + 1):
        mask = (k * f0 < SR * 0.45).astype(float)
        if not mask.any():
            break
        src += mask * np.sin(k * phase) / k ** tilt
    src /= np.max(np.abs(src)) + 1e-9
    if breath:
        src += breath * highpass(noise(n), 1200) * (0.6 + 0.4 * np.sin(phase) ** 2)
    return src


def f0_track(n, base, glide=None, vib_rate=6.0, vib_depth=0.03, jitter=0.012):
    t = np.arange(n) / SR
    if glide is None:
        f = np.full(n, float(base))
    else:  # glide = list of (time_fraction, multiplier)
        xs = [g[0] for g in glide]
        ys = [g[1] for g in glide]
        f = base * np.interp(t / t[-1], xs, ys)
    rate = vib_rate * (1 + 0.08 * smooth(noise(n), 0.3))
    vib = np.sin(2 * np.pi * np.cumsum(rate) / SR + RNG.uniform(0, 6.28))
    jit = smooth(noise(n), 0.02)
    jit /= np.max(np.abs(jit)) + 1e-9
    return f * (1 + vib_depth * vib) * (1 + jitter * jit)


def speak(phones, voiced=True, f0=200.0, glide=None, fscale=1.0, breath=0.06,
          vib_depth=0.025, vib_rate=5.5, bw_scale=1.0, tilt=1.0):
    """phones: list of (symbol, seconds). ' ' = pause.
    Vowels / liquids / nasals go through the formant cascade (voiced, or
    whispered when voiced=False); fricatives and stop bursts are filtered
    noise added on top."""
    total = sum(d for _, d in phones)
    n = n_of(total) + n_of(0.2)
    F = np.zeros((n, 5))
    B = np.zeros((n, 5))
    amp = np.zeros(n)
    extra = np.zeros(n)
    pos = 0
    last_v = '@'
    # look-ahead so 'h' and stops take the colour of the next vowel
    syms = [p for p, _ in phones]
    for idx, (p, d) in enumerate(phones):
        m = n_of(d)
        seg = slice(pos, pos + m)
        nxt = next((s for s in syms[idx + 1:] if s in VOWELS), last_v)
        if p in VOWELS:
            F[seg] = np.array(VOWELS[p]) * fscale
            B[seg] = np.array(BW) * bw_scale
            amp[seg] = LOUD[p]
            last_v = p
        else:
            F[seg] = np.array(VOWELS[nxt]) * fscale
            B[seg] = np.array(BW) * bw_scale * 1.6
            if p == 'h':
                amp[seg] = 0.0
                e = np.hanning(m * 2)[:m] if m > 1 else np.ones(m)
                hn = cascade(noise(m), np.tile(np.array(VOWELS[nxt]) * fscale, (m, 1)),
                             np.tile(np.array(BW) * 1.8, (m, 1)))
                extra[seg] += LOUD['h'] * 0.6 * hn / (np.max(np.abs(hn)) + 1e-9) * e
            elif p in FRIC:
                lo, hi, g = FRIC[p]
                fr = bandpass(noise(m), lo, hi, 3)
                e = np.minimum(1, np.minimum(np.arange(m), np.arange(m)[::-1]) / max(1, n_of(0.03)))
                extra[seg] += g * fr / (np.max(np.abs(fr)) + 1e-9) * e
            elif p in STOPS:
                lo, hi, g, clo = STOPS[p]
                c = min(m, n_of(clo))
                bl = m - c
                if bl > 0:
                    bu = bandpass(noise(bl), lo, hi, 2) * np.exp(-np.arange(bl) / n_of(0.012))
                    extra[pos + c:pos + m] += g * bu / (np.max(np.abs(bu)) + 1e-9)
            # ' ' (pause): silence
        pos += m
    # tail
    F[pos:] = F[pos - 1] if pos > 0 else np.array(VOWELS['@'])
    B[pos:] = B[pos - 1] if pos > 0 else np.array(BW)
    # coarticulation: formants glide, loudness rises and falls smoothly
    for j in range(5):
        F[:, j] = smooth(F[:, j], 0.035)
        B[:, j] = smooth(B[:, j], 0.035)
    amp = smooth(amp, 0.025)
    if voiced:
        src = glottal(f0_track(n, f0, glide, vib_rate, vib_depth), breath, tilt)
    else:
        src = noise(n) * 0.5 + 0.5 * lowpass(noise(n), 2500)
    v = cascade(src * amp, F, B)
    v /= np.max(np.abs(v)) + 1e-9
    if not voiced:
        v = highpass(v, 350)
    out = v + 0.8 * extra
    return out / (np.max(np.abs(out)) + 1e-9)


# --------------------------------------------------------------------------
#  textures
# --------------------------------------------------------------------------
def glass_ping(dur=0.6, base=2300.0, partials=(1.0, 1.41, 2.03, 2.72, 3.45), decay=0.12):
    n = n_of(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for i, p in enumerate(partials):
        f = base * p * (1 + RNG.uniform(-0.01, 0.01))
        y += np.sin(2 * np.pi * f * t + RNG.uniform(0, 6.28)) * np.exp(-t / (decay / (1 + 0.4 * i))) / (1 + i * 0.6)
    return y / (np.max(np.abs(y)) + 1e-9)


def crack(dur=0.25):
    """Something brittle splitting: a burst of clicks over a noise snap."""
    n = n_of(dur)
    t = np.arange(n) / SR
    y = highpass(noise(n), 1500) * np.exp(-t / 0.03)
    for _ in range(14):
        i = int(RNG.uniform(0, 0.6) ** 2 * n)
        L = n_of(0.004)
        if i + L < n:
            y[i:i + L] += RNG.uniform(0.5, 1.2) * np.hanning(L) * RNG.choice([-1, 1])
    y += 0.4 * glass_ping(dur, RNG.uniform(1800, 2600), decay=0.08)
    return y / (np.max(np.abs(y)) + 1e-9)


def thump(f_start=75.0, f_end=42.0, dur=0.22, click=0.25):
    n = n_of(dur)
    t = np.arange(n) / SR
    f = f_end + (f_start - f_end) * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * (1 - np.exp(-t / 0.004)) * np.exp(-t / 0.075)
    c = lowpass(noise(n), 900) * np.exp(-t / 0.006) * click
    return y + c


def saw(f, n, harm_tilt=1.0):
    ph = 2 * np.pi * np.cumsum(np.full(n, f) if np.isscalar(f) else f) / SR
    y = np.zeros(n)
    fmin = f if np.isscalar(f) else float(np.min(f))
    for k in range(1, int(SR * 0.45 / max(fmin, 20)) + 1):
        if k > 200:
            break
        ff = f * k
        mask = (ff < SR * 0.45) if not np.isscalar(ff) else (1.0 if ff < SR * 0.45 else 0.0)
        y += mask * np.sin(k * ph) / k ** harm_tilt
    return y / (np.max(np.abs(y)) + 1e-9)


# --------------------------------------------------------------------------
#  the sounds
# --------------------------------------------------------------------------
def whisper(phones, fscale=1.0, seed_shift=0):
    a = speak(phones, voiced=False, fscale=fscale, bw_scale=1.3)
    # a second, fainter soul a moment behind - "more than one of them"
    b = speak(phones, voiced=False, fscale=fscale * 1.12, bw_scale=1.4)
    pre = reverse_swell(a * 0.6, decay=0.35, length=0.9)
    pre = pre[-n_of(0.45):]
    body = mix((a, 0.0, 1.0), (b, 0.075, 0.45))
    x = mix((pre, 0.0, 0.35), (body, 0.42, 1.0))
    x = reverb(x, wet=0.28, decay=0.35, length=1.0, bright=7000)
    return normalize(fade(x, 0.02, 0.15), -1.5, -19)


def wail(phones, f0, glide, rasp=1.8, crack_at=None, length_pad=0.6):
    v1 = speak(phones, voiced=True, f0=f0, glide=glide, fscale=1.18, breath=0.12,
               vib_depth=0.035, vib_rate=6.5, tilt=0.9)
    v2 = speak(phones, voiced=True, f0=f0 * 1.012, glide=glide, fscale=1.24, breath=0.15,
               vib_depth=0.04, vib_rate=5.7, tilt=0.9)
    v = softclip(v1 + 0.7 * v2, rasp)
    v = highpass(v, 250)
    parts = [(v, 0.0, 1.0)]
    if crack_at is not None:
        parts.append((crack(0.3), crack_at, 0.5))
    x = mix(*parts)
    x = pad_to(x, len(x) + n_of(length_pad))
    x = reverb(x, wet=0.33, decay=0.55, length=1.8, bright=6500)
    return normalize(fade(x, 0.01, 0.4), -1.0, -16)


def make_all():
    S = {}
    # ---- whispers: almost words --------------------------------------------
    S['soul/whisper1'] = whisper([('l', .06), ('e', .13), ('t', .07), (' ', .03), ('m', .07),
                                  ('i', .22), (' ', .1), ('a', .16), ('u', .13), ('t', .07)])
    S['soul/whisper2'] = whisper([('h', .08), ('e', .16), ('l', .07), ('p', .07), (' ', .05),
                                  ('m', .06), ('i', .28)], fscale=1.05)
    S['soul/whisper3'] = whisper([('s', .16), ('o', .24), (' ', .06), ('k', .06), ('o', .17),
                                  ('l', .09), ('d', .06), (' ', .05), ('i', .1), ('n', .06),
                                  ('s', .12), ('a', .12), ('i', .1), ('d', .05)], fscale=0.97)
    S['soul/whisper4'] = whisper([('f', .12), ('r', .07), ('i', .26), (' ', .08), ('@', .13),
                                  ('s', .24)], fscale=1.08)

    # ---- wails / screams ---------------------------------------------------
    S['soul/wail1'] = wail([('a', .55), ('a', .5), ('i', .5), ('@', .25)], 560,
                           [(0, .85), (.25, 1.25), (.55, 1.5), (.8, 1.2), (1, .8)], crack_at=0.0)
    S['soul/wail2'] = wail([('o', .3), (' ', .07), ('o', .3), (' ', .06), ('a', .75), ('u', .3)], 640,
                           [(0, 1.0), (.2, 1.1), (.4, 1.05), (.55, 1.45), (.8, 1.3), (1, .75)],
                           rasp=2.2)
    S['soul/wail3'] = wail([('e', .25), ('a', .45), ('i', .7)], 700,
                           [(0, .8), (.3, 1.3), (.6, 1.7), (1, 1.55)], rasp=2.6, crack_at=0.05)

    # ---- shrieks: short and sharp -------------------------------------------
    for k, (ph, f0, gl) in enumerate([
            ([('i', .38), ('e', .14)], 820, [(0, .8), (.35, 1.45), (1, 1.15)]),
            ([('a', .2), ('i', .32)], 900, [(0, .9), (.4, 1.5), (1, 1.2)])], 1):
        v = wail(ph, f0, gl, rasp=3.0, crack_at=0.0, length_pad=0.3)
        S[f'soul/shriek{k}'] = normalize(fade(v, 0.005, 0.2), -1.0, -15)

    # ---- gasps: a sharp inhale ---------------------------------------------
    for k, vowel in enumerate(['a', 'o'], 1):
        n = n_of(0.42)
        t = np.arange(n) / SR
        env = (t / t[-1]) ** 1.8 * (1 - np.exp(-(t[-1] - t) / 0.012))
        F = np.tile(np.array(VOWELS[vowel]) * 1.1, (n, 1))
        F[:, 1] *= np.linspace(0.9, 1.15, n)
        g = cascade(noise(n), F, np.tile(np.array(BW) * 2.2, (n, 1)))
        g = highpass(g, 500) * env
        g = mix((g, 0.0, 1.0), (glass_ping(0.3, 3100, decay=0.05), 0.38, 0.12))
        S[f'soul/gasp{k}'] = normalize(fade(reverb(g, 0.2, 0.25, 0.7), 0.002, 0.08), -1.5, -18)

    # ---- chants: a voice that almost speaks ---------------------------------
    def chant(phones, f0, glide):
        low = speak(phones, voiced=True, f0=f0, glide=glide, fscale=1.0, breath=0.18,
                    vib_depth=0.012, vib_rate=4.5)
        high = speak(phones, voiced=True, f0=f0 * 2, glide=glide, fscale=1.15, breath=0.2,
                     vib_depth=0.02, vib_rate=5.1)
        wh = speak(phones, voiced=False, fscale=1.05, bw_scale=1.3)
        body = mix((low, 0.0, 0.8), (high, 0.012, 0.35), (wh, 0.03, 0.5))
        pre = reverse_swell(body * 0.5, 0.6, 1.4)[-n_of(0.7):]
        x = mix((pre, 0.0, 0.45), (body, 0.66, 1.0))
        x = pad_to(x, len(x) + n_of(0.5))
        x = reverb(x, wet=0.4, decay=0.7, length=2.2, bright=5000)
        return normalize(fade(x, 0.03, 0.4), -1.0, -17)

    S['soul/chant1'] = chant([('w', .08), ('i', .2), (' ', .05), ('a', .22), ('r', .1), (' ', .06),
                              ('s', .1), ('t', .05), ('i', .12), ('l', .1), (' ', .05), ('h', .07),
                              ('i', .22), ('r', .16)], 215, [(0, 1.0), (.7, 0.97), (1, 0.88)])
    S['soul/chant2'] = chant([('l', .07), ('e', .15), ('t', .06), (' ', .03), ('@', .1), ('s', .13),
                              (' ', .06), ('g', .05), ('o', .3), ('u', .24)], 196,
                             [(0, 1.0), (.5, 1.04), (1, 0.9)])

    # ---- heartbeat inside crystal -------------------------------------------
    def knock(dur=0.12):
        k = bandpass(noise(n_of(dur)), 110, 420, 2)
        return k * np.exp(-np.arange(n_of(dur)) / n_of(0.022))
    # a deep thump plus its octave and a dull knock, so the beat is still
    # heard on laptop speakers and earbuds that cannot play 50 Hz
    lub = thump(78, 44, 0.24) + 0.7 * thump(156, 92, 0.24, 0.0)
    dub = thump(70, 40, 0.24, 0.15) + 0.6 * thump(140, 84, 0.24, 0.0)
    hb = mix((lub, 0.0, 1.0), (dub, 0.19, 0.75),
             (knock(), 0.0, 1.1), (knock(), 0.19, 0.8),
             (glass_ping(0.5, 2600, decay=0.09), 0.005, 0.05),
             (glass_ping(0.5, 2900, decay=0.08), 0.195, 0.035))
    hb = pad_to(hb, n_of(0.75))
    S['soul/heartbeat'] = normalize(fade(reverb(hb, 0.15, 0.25, 0.6, 3000), 0.001, 0.1), -1.0, -14)

    # ---- laser charge: light and voices pulled into the eye -----------------
    dur = 2.4
    n = n_of(dur)
    t = np.arange(n) / SR
    f = 160 * (5.0 ** (t / dur) ** 1.6)
    tone = sum(saw(f * d, n, 1.1) for d in (1.0, 1.007, 0.993)) / 3
    cut = 400 + 7000 * (t / dur) ** 2
    out = np.zeros(n)
    blk = 1024
    zi = None
    for s in range(0, n, blk):
        e = min(s + blk, n)
        b, a = signal.butter(2, min(cut[s] / (SR / 2), 0.95), 'low')
        if zi is None:
            zi = signal.lfilter_zi(b, a) * 0
        out[s:e], zi = signal.lfilter(b, a, tone[s:e], zi=zi)
    choir = speak([('a', dur * 0.5), ('i', dur * 0.5)], voiced=True, f0=330,
                  glide=[(0, 0.9), (1, 2.0)], fscale=1.15, breath=0.2, vib_depth=0.04)
    choir = pad_to(choir, n)
    crackle = np.zeros(n)
    for _ in range(160):
        i = int((RNG.uniform(0, 1) ** 0.5) * (n - 200))
        crackle[i:i + 120] += RNG.uniform(0.2, 1) * np.hanning(120) * RNG.choice([-1, 1])
    crackle = highpass(crackle, 2500)
    env = (t / dur) ** 0.8
    x = (0.55 * out + 0.35 * choir * env + 0.25 * crackle) * env
    x = mix((x, 0.0, 1.0), (glass_ping(0.4, 3300, decay=0.06), dur - 0.06, 0.4))
    S['laser/charge'] = normalize(fade(reverb(x, 0.2, 0.3, 0.8), 0.05, 0.03), -1.0, -16)

    # ---- laser hum: a buzzing beam with a voice screaming inside it ---------
    dur = 1.6
    n = n_of(dur)
    t = np.arange(n) / SR
    buzz = saw(110, n, 1.0) * 0.6 + saw(113.5, n, 1.0) * 0.5 + saw(220.7, n, 1.2) * 0.3
    buzz = lowpass(buzz, 3200)
    sizzle = bandpass(noise(n), 3000, 9000, 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 31 * t))
    voice = speak([('a', dur)], voiced=True, f0=440, glide=[(0, 1.0), (.5, 1.03), (1, 1.0)],
                  fscale=1.2, breath=0.25, vib_depth=0.05, vib_rate=7.0)
    voice = pad_to(voice, n)
    x = 0.55 * buzz + 0.12 * sizzle + 0.34 * voice
    x = x * (1 + 0.15 * np.sin(2 * np.pi * 7.5 * t))
    S['laser/hum'] = normalize(fade(x, 0.06, 0.12), -1.0, -15)

    # ---- death: a chorus of every voice it ever held ------------------------
    dur = 3.2
    voices = []
    for k, (f0, vow) in enumerate([(330, 'a'), (415, 'o'), (494, 'a'), (587, 'e'), (660, 'o'),
                                   (740, 'a'), (880, 'i')]):
        ph = [(vow, dur * 0.45), ('o', dur * 0.3), ('u', dur * 0.25)]
        v = speak(ph, voiced=True, f0=f0, glide=[(0, 1.0), (.4, 1.06), (1, 0.55)],
                  fscale=1.1 + 0.03 * k, breath=0.18, vib_depth=0.03 + 0.004 * k,
                  vib_rate=5 + 0.4 * k)
        voices.append((v, RNG.uniform(0, 0.12), 0.5 + 0.1 * RNG.uniform()))
    chorus = mix(*voices)
    n = len(chorus)
    t = np.arange(n) / SR
    drone = np.sin(2 * np.pi * 55 * t + 0.4 * np.sin(2 * np.pi * 0.7 * t)) * 0.5
    ws = whisper([('l', .06), ('e', .13), ('t', .07), ('m', .07), ('i', .22), ('a', .16), ('u', .13),
                  ('t', .07)], 1.15)
    x = mix((chorus, 0.0, 1.0), (drone * np.minimum(1, t / 0.5), 0.0, 0.35), (ws, 0.6, 0.35))
    x = pad_to(x, len(x) + n_of(0.8))
    x = reverb(x, 0.45, 0.9, 2.6, 5000)
    S['death/chorus'] = normalize(fade(x, 0.05, 0.6), -1.0, -16)

    # ---- the freed souls drift away -----------------------------------------
    dur = 2.6
    n = n_of(dur)
    t = np.arange(n) / SR
    wind = np.zeros(n)
    for k in range(3):
        c = 600 + 1800 * (0.5 + 0.5 * np.sin(2 * np.pi * (0.35 + 0.12 * k) * t + k))
        seg = np.zeros(n)
        zi = None
        for s in range(0, n, 1024):
            e = min(s + 1024, n)
            b, a = signal.butter(2, [max(c[s] * 0.7, 80) / (SR / 2), min(c[s] * 1.3, 18000) / (SR / 2)], 'band')
            if zi is None:
                zi = np.zeros(max(len(a), len(b)) - 1)
            seg[s:e], zi = signal.lfilter(b, a, noise(e - s), zi=zi)
        wind += seg
    oo = sum(pad_to(speak([('u', dur * 0.6), ('o', dur * 0.4)], voiced=True, f0=f0,
                          glide=[(0, 1.0), (1, 1.12)], fscale=1.2, breath=0.35,
                          vib_depth=0.03), n) for f0 in (880, 1046, 1174))
    sparkle = np.zeros(n)
    for _ in range(9):
        p = glass_ping(0.4, RNG.uniform(2600, 4200), decay=0.07)
        i = int(RNG.uniform(0.05, 0.8) * n)
        sparkle[i:i + len(p)] += pad_to(p, len(sparkle[i:i + len(p)])) * RNG.uniform(0.3, 0.8)
    env = np.minimum(1, t / 0.4) * np.exp(-np.maximum(0, t - 1.2) / 0.7)
    x = (0.5 * wind / (np.max(np.abs(wind)) + 1e-9) + 0.35 * oo / 3 + 0.25 * sparkle) * env
    x = reverb(x, 0.45, 0.8, 2.0, 7000)
    S['death/souls'] = normalize(fade(x, 0.05, 0.5), -1.0, -18)
    return S


# --------------------------------------------------------------------------
#  sounds.json / lang
# --------------------------------------------------------------------------
EVENTS = {
    # event: (files, subtitle, attenuation_distance, extra pitch variants)
    'soul.whisper': (['soul/whisper1', 'soul/whisper2', 'soul/whisper3', 'soul/whisper4'],
                     'Soul whispers', 16, [0.9]),
    'soul.wail': (['soul/wail1', 'soul/wail2', 'soul/wail3'], 'Soul wails', 32, []),
    'soul.shriek': (['soul/shriek1', 'soul/shriek2'], 'Soul shrieks', 32, []),
    'soul.gasp': (['soul/gasp1', 'soul/gasp2'], 'Soul gasps', 16, []),
    'soul.chant': (['soul/chant1', 'soul/chant2'], 'Soul speaks', 24, []),
    'soul.heartbeat': (['soul/heartbeat'], 'Heart beats', 16, []),
    'laser.charge': (['laser/charge'], 'Light gathers', 32, []),
    'laser.hum': (['laser/hum'], 'Beam hums', 24, []),
    'death.chorus': (['death/chorus'], 'Voices cry out', 48, []),
    'death.souls': (['death/souls'], 'Souls drift free', 32, []),
}


def sounds_json():
    out = {}
    for ev, (files, sub, att, pitches) in EVENTS.items():
        entries = []
        for f in files:
            entries.append({'name': f'fv_endboss:{f}', 'attenuation_distance': att})
            for p in pitches:
                entries.append({'name': f'fv_endboss:{f}', 'pitch': p, 'attenuation_distance': att})
        out[ev] = {'subtitle': f'subtitles.fv_endboss.{ev}', 'sounds': entries}
    return out


def lang_json():
    return {f'subtitles.fv_endboss.{ev}': sub for ev, (_, sub, _, _) in EVENTS.items()}


def write_ogg(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav = tmp.name
    sf.write(wav, x.astype(np.float32), SR, subtype='PCM_16')
    # bitexact: no random Ogg stream serial, so the same audio gives the same file
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-ac', '1', '-ar', str(SR),
                    '-c:a', 'libvorbis', '-q:a', '5', '-map_metadata', '-1',
                    '-fflags', '+bitexact', '-flags:a', '+bitexact', path], check=True)
    os.unlink(wav)


def main():
    pack = sys.argv[1] if len(sys.argv) > 1 else 'pack'
    base = os.path.join(pack, 'assets', 'fv_endboss')
    sounds = make_all()
    for name, x in sounds.items():
        write_ogg(os.path.join(base, 'sounds', name + '.ogg'), x)
    with open(os.path.join(base, 'sounds.json'), 'w', newline='\n') as f:
        json.dump(sounds_json(), f, indent=2)
        f.write('\n')
    os.makedirs(os.path.join(base, 'lang'), exist_ok=True)
    with open(os.path.join(base, 'lang', 'en_us.json'), 'w', newline='\n') as f:
        json.dump(lang_json(), f, indent=2)
        f.write('\n')
    # previews for checking: wav copies are not shipped
    if len(sys.argv) > 2:
        prev = sys.argv[2]
        os.makedirs(prev, exist_ok=True)
        for name, x in sounds.items():
            sf.write(os.path.join(prev, name.replace('/', '_') + '.wav'), x.astype(np.float32), SR)
    print(f'{len(sounds)} sounds written to {base}')


if __name__ == '__main__':
    main()
