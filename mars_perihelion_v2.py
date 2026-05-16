"""
mars_perihelion_v2.py
=====================

"Perihelion Drift" — Mission (Austin Yang, Mikul Saravanan,
Nicholas Tan, Steven Huang).
Aerotech (melodic-festival house lineage), 124 BPM, A minor, ~3:20.
For the Perihelion Festival, Airlock Atrium, Arcadia Planitia,
Mars (2073) — The Pressure Commons Era.

v2 changes from v1 — the production sophistication pass:
  1. SIDECHAIN COMPRESSION — bass and pads duck to the kick. The
     defining "pump and breathe" of festival house. Every track on
     Afterlife uses this; v1 didn't.
  2. AUX REVERB ON LEAD — lead synth gets its own huge reverb send,
     separate from the master bus. Lead sounds like it's playing in a
     cathedral while the kick stays tight and dry.
  3. LAYERED DROPS — drop 2 introduces a counter-melody arp + vocal
     chops + crash, on top of all of drop 1's elements. Drop 2 earns
     its bigger payoff instead of being a copy of drop 1.
  4. PITCHED VOCAL CHOPS — wordless 'ah' / 'oh' synthesized via formant
     filtering. Used sparingly: a few sustains in the breakdown, a few
     accents in drop 2. Worldbuilding: the colony refuses to add lyrics
     because language itself is contested between Earth-English and the
     emerging Mars patois — only the human voice without words.
  5. REAL CHORD PROGRESSION — Am - F - C - G (i - VI - III - VII).
     Anyma-style melodic-house progression. Lead hook varies per chord.

Worldbuilding choices preserved from v1:
  - 124 BPM (slower than Earth festival's 126-128 — 0.38g overcorrection)
  - Continuous 60Hz dome drone under everything (life-support hum)
  - Mid-bass forward, sub de-emphasized (CO2-thin air doesn't carry sub)
  - Long aux reverb (200m central dome)
  - Atmospheric intro before drop (ritual entry from the corridors)

Run:
    python mars_perihelion_v2.py
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal


# ============================================================
# CONSTANTS
# ============================================================
SAMPLE_RATE       = 44100
BPM               = 124
BEATS_PER_BAR     = 4
STEPS_PER_BAR     = 16
SECONDS_PER_BEAT  = 60 / BPM
SECONDS_PER_BAR   = SECONDS_PER_BEAT * BEATS_PER_BAR
SECONDS_PER_STEP  = SECONDS_PER_BAR / STEPS_PER_BAR


# ============================================================
# NOTE → FREQUENCY
# ============================================================
NOTE_NAMES = ['c', 'c#', 'd', 'eb', 'e', 'f', 'f#', 'g', 'g#', 'a', 'bb', 'b']
ALIASES    = {'db': 'c#', 'd#': 'eb', 'gb': 'f#', 'ab': 'g#', 'a#': 'bb'}


def note_to_freq(s):
    s = s.lower().strip()
    if s in ('.', '_'):
        return None
    for i, ch in enumerate(s):
        if ch.isdigit() or ch == '-':
            pitch, octave = s[:i], int(s[i:])
            break
    else:
        pitch, octave = s, 4
    pitch = ALIASES.get(pitch, pitch)
    semitone = NOTE_NAMES.index(pitch)
    midi = (octave + 1) * 12 + semitone
    return 440 * 2 ** ((midi - 69) / 12)


# ============================================================
# PATTERN DSL
# ============================================================
class Pattern:
    def __init__(self, pattern_str, bars=1):
        tokens = pattern_str.split()
        steps_total = STEPS_PER_BAR * bars
        if len(tokens) < steps_total:
            tokens += ['.'] * (steps_total - len(tokens))
        else:
            tokens = tokens[:steps_total]
        self.events = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in ('.', '_'):
                i += 1
                continue
            dur = 1
            j = i + 1
            while j < len(tokens) and tokens[j] == '_':
                dur += 1
                j += 1
            self.events.append((i, tok, dur))
            i += 1
        self.bars = bars


def pattern(s, bars=1):
    return Pattern(s, bars)


# ============================================================
# DSP PRIMITIVES
# ============================================================
def adsr(n, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
    a = max(1, int(attack  * SAMPLE_RATE))
    d = max(1, int(decay   * SAMPLE_RATE))
    r = max(1, int(release * SAMPLE_RATE))
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a),
        np.linspace(1, sustain, d),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def lowpass(audio, cutoff_hz, order=4):
    nyq = SAMPLE_RATE / 2
    cutoff_hz = min(cutoff_hz, nyq * 0.99)
    sos = signal.butter(order, cutoff_hz / nyq, btype='low', output='sos')
    return signal.sosfilt(sos, audio)


def highpass(audio, cutoff_hz, order=4):
    nyq = SAMPLE_RATE / 2
    cutoff_hz = min(cutoff_hz, nyq * 0.99)
    sos = signal.butter(order, cutoff_hz / nyq, btype='high', output='sos')
    return signal.sosfilt(sos, audio)


def bandpass(audio, low_hz, high_hz, order=2):
    nyq = SAMPLE_RATE / 2
    high_hz = min(high_hz, nyq * 0.99)
    sos = signal.butter(order, [low_hz / nyq, high_hz / nyq],
                        btype='band', output='sos')
    return signal.sosfilt(sos, audio)


def saturate(audio, drive=2.0):
    return np.tanh(audio * drive) / np.tanh(drive)


def reverb_master(audio, decay=0.6, mix=0.15):
    """Master-bus reverb (room sound)."""
    delays = [int(SAMPLE_RATE * d) for d in
              [0.029, 0.037, 0.041, 0.043, 0.067, 0.089]]
    out = np.zeros_like(audio)
    for d in delays:
        gain = decay ** (d / SAMPLE_RATE)
        delayed = np.zeros_like(audio)
        delayed[d:] = audio[:-d] * gain
        out += delayed
    return audio * (1 - mix) + out * (mix / len(delays)) * 3


def aux_reverb(audio, decay=0.85, mix=0.55):
    """Heavy aux reverb for lead — much bigger than master.
       The 'cathedral on top of the dry signal' effect."""
    delays = [int(SAMPLE_RATE * d) for d in
              [0.029, 0.037, 0.041, 0.043, 0.067, 0.089, 0.113, 0.149, 0.193]]
    out = np.zeros_like(audio)
    for d in delays:
        gain = decay ** (d / SAMPLE_RATE)
        delayed = np.zeros_like(audio)
        delayed[d:] = audio[:-d] * gain
        out += delayed
    out = out / len(delays) * 4
    # smooth highs slightly
    out = lowpass(out, 6000, order=2)
    return audio * (1 - mix) + out * mix


def sidechain(audio, kick_pat, num_bars, depth=0.65, release_ms=140):
    """Duck audio in sync with a kick pattern. The defining festival
       house production move — bass and pads breathe with the kick."""
    n = len(audio)
    envelope = np.ones(n)
    repeats = num_bars // kick_pat.bars
    release_s = int(release_ms / 1000 * SAMPLE_RATE)

    for rep in range(repeats):
        bar_offset = rep * kick_pat.bars * SECONDS_PER_BAR
        for step_idx, tok, _ in kick_pat.events:
            if tok != 'x':
                continue
            start_t = bar_offset + step_idx * SECONDS_PER_STEP
            start_s = int(start_t * SAMPLE_RATE)
            if start_s >= n:
                break
            end_s = min(start_s + release_s, n)
            duck_len = end_s - start_s
            # exponential recovery from (1-depth) back to 1
            duck = (1 - depth) + depth * (np.linspace(0, 1, duck_len) ** 2)
            envelope[start_s:end_s] = np.minimum(envelope[start_s:end_s], duck)
    return audio * envelope


# ============================================================
# INSTRUMENTS
# ============================================================
def kick(_=None):
    """Tech-house kick: pitched sweep + transient + saturation."""
    dur = 0.4
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, False)
    freq = 80 * np.exp(-t * 12) + 50
    body = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    click_n = int(0.005 * SAMPLE_RATE)
    body[:click_n] += np.random.uniform(-1, 1, click_n) * np.linspace(1, 0, click_n) * 0.6
    env = adsr(n, attack=0.001, decay=0.05, sustain=0.3, release=0.3)
    return saturate(body * env, drive=1.8) * 0.95


def sub_bass(freq, dur):
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, False)
    fund = np.sin(2 * np.pi * freq * t)
    mid  = np.sin(2 * np.pi * freq * 2 * t) * 0.5
    saw  = signal.sawtooth(2 * np.pi * freq * t) * 0.3
    saw  = lowpass(saw, freq * 4)
    sig  = fund + mid + saw
    env  = adsr(n, attack=0.005, decay=0.05, sustain=0.85, release=0.05)
    return saturate(sig * env, drive=1.4) * 0.45


def lead_synth(freq, dur):
    """Wide detuned saw lead."""
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, False)
    saw1 = signal.sawtooth(2 * np.pi * freq          * t)
    saw2 = signal.sawtooth(2 * np.pi * freq * 1.005  * t)
    saw3 = signal.sawtooth(2 * np.pi * freq * 0.995  * t)
    sig  = (saw1 + saw2 + saw3) / 3
    sig  = lowpass(sig, min(freq * 8, 6500))
    env  = adsr(n, attack=0.005, decay=0.15, sustain=0.65, release=0.1)
    return sig * env * 0.25


def pluck(freq, dur):
    """Short plucky synth for counter-melody arp."""
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, False)
    saw = signal.sawtooth(2 * np.pi * freq * t)
    sq  = np.sign(np.sin(2 * np.pi * freq * t)) * 0.3
    sig = saw + sq
    sig = lowpass(sig, min(freq * 6, 4000))
    env = adsr(n, attack=0.001, decay=0.12, sustain=0.0, release=0.08)
    return sig * env * 0.16


def pad(freq, dur):
    """Lush evolving pad (one voice — call multiple times for chord)."""
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, False)
    sig = np.zeros(n)
    for harm, amp in [(1, 1.0), (2, 0.5), (3, 0.25), (5, 0.12)]:
        sig += np.sin(2 * np.pi * freq * harm           * t) * amp
        sig += np.sin(2 * np.pi * freq * harm * 1.004   * t) * amp * 0.6
    sig /= 4
    sig = lowpass(sig, 1800)
    env = adsr(n, attack=0.4, decay=0.3, sustain=0.85, release=0.6)
    lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    return sig * env * lfo * 0.11


def vocal_chop(freq, dur, vowel='ah'):
    """Wordless 'ah' / 'oh' via formant filtering of a sawtooth source.
       Used sparingly — Mars-born colonists' voices, no language."""
    formants = {
        'ah': [(700, 1.0), (1220, 0.7), (2600, 0.4)],
        'oh': [(450, 1.0), (800,  0.6), (2830, 0.25)],
    }
    f_set = formants[vowel]
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, False)
    # vibrato
    vibrato = 1 + 0.018 * np.sin(2 * np.pi * 5.5 * t)
    phase = np.cumsum(2 * np.pi * freq * vibrato / SAMPLE_RATE)
    source = signal.sawtooth(phase)
    # subtle breath
    source += np.random.uniform(-1, 1, n) * 0.04
    # apply formant bandpass stack
    out = np.zeros(n)
    for f_freq, f_amp in f_set:
        bw = max(80, f_freq * 0.18)
        try:
            filtered = bandpass(source, max(50, f_freq - bw),
                                f_freq + bw, order=2)
            out += filtered * f_amp
        except Exception:
            continue
    # slow attack, natural release — vocal-like envelope
    env = adsr(n, attack=0.06, decay=0.15, sustain=0.7, release=0.35)
    return out * env * 0.22


def hihat_closed(_=None):
    n = int(0.05 * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n)
    noise = highpass(noise, 7000)
    env = adsr(n, attack=0.001, decay=0.02, sustain=0.0, release=0.01)
    return noise * env * 0.16


def hihat_open(_=None):
    n = int(0.22 * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n)
    noise = highpass(noise, 6000)
    env = adsr(n, attack=0.001, decay=0.05, sustain=0.3, release=0.18)
    return noise * env * 0.13


def clap(_=None):
    n = int(0.18 * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n)
    noise = highpass(noise, 1500)
    noise = lowpass(noise, 4500)
    env = np.zeros(n)
    for off_ms, gain in [(0, 1.0), (12, 0.7), (24, 0.5), (40, 1.0)]:
        off = int(off_ms * SAMPLE_RATE / 1000)
        if off < n:
            sub = adsr(n - off, attack=0.001, decay=0.025,
                       sustain=0.0, release=0.06)
            env[off:off + len(sub)] += sub * gain
    return noise * env * 0.32


def snare(_=None):
    """Snare for fills."""
    n = int(0.1 * SAMPLE_RATE)
    t = np.linspace(0, 0.1, n, False)
    body = np.sin(2 * np.pi * 200 * t)
    noise = highpass(np.random.uniform(-1, 1, n), 1500)
    sig = body * 0.5 + noise * 0.7
    env = adsr(n, attack=0.001, decay=0.04, sustain=0.0, release=0.05)
    return sig * env * 0.22


def crash(_=None):
    """Long crash cymbal for drop transitions."""
    n = int(1.8 * SAMPLE_RATE)
    noise = highpass(np.random.uniform(-1, 1, n), 4000)
    shimmer = lowpass(highpass(np.random.uniform(-1, 1, n), 2000), 8000) * 0.3
    sig = noise + shimmer
    env = adsr(n, attack=0.001, decay=0.4, sustain=0.25, release=1.3)
    return sig * env * 0.18


def dome_drone(dur, freq=60):
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, False)
    sig  = np.sin(2 * np.pi * freq         * t)
    sig += np.sin(2 * np.pi * freq * 2     * t) * 0.3
    sig += np.sin(2 * np.pi * freq * 3     * t) * 0.1
    sig *= 1 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
    return sig * 0.05


def riser(dur):
    """Layered riser: filtered noise sweep + tonal upward sweep."""
    n = int(dur * SAMPLE_RATE)
    t = np.linspace(0, dur, n, False)
    # noise component, opening filter
    noise = highpass(np.random.uniform(-1, 1, n), 1200)
    noise_env = np.linspace(0, 1, n) ** 2.2
    # tonal component, pitch sweep up
    sweep_freq = 200 * np.exp(t / dur * 3)  # 200Hz → ~4kHz
    tonal = np.sin(2 * np.pi * np.cumsum(sweep_freq) / SAMPLE_RATE) * 0.5
    tonal_env = np.linspace(0, 1, n) ** 2
    return (noise * noise_env * 0.18) + (tonal * tonal_env * 0.12)


def wind_texture(dur):
    n = int(dur * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n)
    noise = lowpass(noise, 800)
    noise = highpass(noise, 100)
    t = np.linspace(0, dur, n, False)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t)
    return noise * mod * 0.07


# ============================================================
# RENDER PATTERNS → AUDIO
# ============================================================
def render_drum(pat, drum_func, num_bars):
    out = np.zeros(int(num_bars * SECONDS_PER_BAR * SAMPLE_RATE))
    repeats = num_bars // pat.bars
    for rep in range(repeats):
        bar_offset = rep * pat.bars * SECONDS_PER_BAR
        for step_idx, tok, _ in pat.events:
            if tok != 'x':
                continue
            start = int((bar_offset + step_idx * SECONDS_PER_STEP) * SAMPLE_RATE)
            hit = drum_func()
            end = min(start + len(hit), len(out))
            out[start:end] += hit[:end - start]
    return out


def render_pitched(pat, synth_func, num_bars, default_dur=None, **kwargs):
    out = np.zeros(int(num_bars * SECONDS_PER_BAR * SAMPLE_RATE))
    repeats = num_bars // pat.bars
    for rep in range(repeats):
        bar_offset = rep * pat.bars * SECONDS_PER_BAR
        for step_idx, tok, dur_steps in pat.events:
            f = note_to_freq(tok)
            if f is None:
                continue
            start = int((bar_offset + step_idx * SECONDS_PER_STEP) * SAMPLE_RATE)
            dur_s = default_dur or (dur_steps * SECONDS_PER_STEP * 0.95)
            note = synth_func(f, dur_s, **kwargs)
            end = min(start + len(note), len(out))
            out[start:end] += note[:end - start]
    return out


# ============================================================
# PATTERNS — A minor, chord progression: Am - F - C - G
# ============================================================

# DRUMS
kick_4floor   = pattern("x . . . x . . . x . . . x . . .")
hat_offbeat   = pattern(". . x . . . x . . . x . . . x .")
hat_open_pat  = pattern(". . . . . . x . . . . . . . x .")
clap_pat      = pattern(". . . . x . . . . . . . x . . .")
snare_fill    = pattern(". . . . . . . . x . x . x x x x")  # last-bar fill

# BASS — 4-bar riff following Am - F - C - G chord progression
bass_main = pattern(
    "a2 . . . a2 . e2 . a2 . . c3 . . e2 . "    # Am
    "f2 . . . f2 . c3 . f2 . . a2 . . c3 . "    # F
    "c2 . . . c2 . g2 . c2 . . e2 . . g2 . "    # C
    "g2 . . . g2 . d3 . g2 . . b2 . . d3 . ",   # G
    bars=4,
)

# LEAD HOOK — 4-bar phrase, varies per chord. THIS is the song's hook.
lead_hook = pattern(
    "a4 . . c5 . e5 . a5 . . g5 . e5 . . . "    # Am: ascending then resolve
    "a4 . . c5 . f5 . a5 . . g5 . f5 . . . "    # F:  same shape, F-color
    "g4 . . c5 . e5 . g5 . . e5 . c5 . . . "    # C:  brighter, descends
    "g4 . . b4 . d5 . g5 . . f5 . d5 . . . ",   # G:  tension to resolve
    bars=4,
)

# COUNTER-MELODY ARP — busy 16th notes, only enters in drop 2
counter_arp = pattern(
    "a3 e4 a4 e4 a3 e4 a4 e4 a3 e4 a4 e4 a3 e4 a4 e4 "    # Am
    "f3 c4 a4 c4 f3 c4 a4 c4 f3 c4 a4 c4 f3 c4 a4 c4 "    # F
    "c3 g3 e4 g3 c3 g3 e4 g3 c3 g3 e4 g3 c3 g3 e4 g3 "    # C
    "g2 d3 b3 d3 g2 d3 b3 d3 g2 d3 b3 d3 g2 d3 b3 d3 ",   # G
    bars=4,
)

# PAD VOICES — three voices stack into chord (root, third, fifth)
pad_root = pattern(
    "a3 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "f3 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "c3 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "g3 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ",
    bars=4,
)
pad_third = pattern(
    "c4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "a3 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "e4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "b3 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ",
    bars=4,
)
pad_fifth = pattern(
    "e4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "c4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "g4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
    "d4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ",
    bars=4,
)

# VOCAL CHOPS — sparse. Long sustains for breakdown, short stabs for drop 2.
vox_breakdown = pattern(
    "a4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "    # long Ah on root
    ". . . . . . . . . . . . . . . . "
    "f4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "    # long Ah on F
    ". . . . . . . . . . . . . . . . ",
    bars=4,
)
vox_drop2 = pattern(
    ". . . . . . . . . . . . a5 . . . "    # one stab per bar, syncopated
    ". . . . . . . . . . . . a5 . . . "
    ". . . . . . . . . . . . g5 . . . "
    ". . . . . . . . . . . . e5 . . . ",
    bars=4,
)


# ============================================================
# ARRANGEMENT
# ============================================================
def render_song():
    sections = []

    # ---------- INTRO (16 bars, atmospheric) ----------
    print("  intro...")
    nb = 16
    s = np.zeros(int(nb * SECONDS_PER_BAR * SAMPLE_RATE))
    s += dome_drone(nb * SECONDS_PER_BAR)
    s += wind_texture(nb * SECONDS_PER_BAR)
    # pad enters bar 4 (3 voices = chord)
    pad_voices = (
        render_pitched(pad_root,  pad, nb - 4) +
        render_pitched(pad_third, pad, nb - 4) +
        render_pitched(pad_fifth, pad, nb - 4)
    )
    pad_off = int(4 * SECONDS_PER_BAR * SAMPLE_RATE)
    s[pad_off:pad_off + len(pad_voices)] += pad_voices
    # sparse hat from bar 8
    hat_a = render_drum(hat_offbeat, hihat_closed, nb - 8) * 0.5
    hat_off = int(8 * SECONDS_PER_BAR * SAMPLE_RATE)
    s[hat_off:hat_off + len(hat_a)] += hat_a
    sections.append(s)

    # ---------- BUILD 1 (8 bars) ----------
    print("  build 1...")
    nb = 8
    s = np.zeros(int(nb * SECONDS_PER_BAR * SAMPLE_RATE))
    s += dome_drone(nb * SECONDS_PER_BAR)
    pad_voices = (
        render_pitched(pad_root,  pad, nb) +
        render_pitched(pad_third, pad, nb) +
        render_pitched(pad_fifth, pad, nb)
    )
    s += pad_voices
    s += render_drum(hat_offbeat, hihat_closed, nb)
    s += render_drum(kick_4floor, kick, nb) * 0.65
    # big riser last 4 bars
    riser_a = riser(4 * SECONDS_PER_BAR)
    riser_off = int(4 * SECONDS_PER_BAR * SAMPLE_RATE)
    s[riser_off:riser_off + len(riser_a)] += riser_a
    sections.append(s)

    # ---------- DROP 1 (16 bars, sidechained, with lead) ----------
    print("  drop 1...")
    nb = 16
    s = np.zeros(int(nb * SECONDS_PER_BAR * SAMPLE_RATE))
    # crash on the first beat
    crash_hit = crash()
    s[:len(crash_hit)] += crash_hit * 0.7
    # drone (always)
    s += dome_drone(nb * SECONDS_PER_BAR)
    # drums
    s += render_drum(kick_4floor, kick, nb)
    s += render_drum(hat_offbeat, hihat_closed, nb)
    s += render_drum(hat_open_pat, hihat_open, nb)
    s += render_drum(clap_pat, clap, nb)
    # bass + pad — SIDECHAINED to kick
    bass_a = render_pitched(bass_main, sub_bass, nb)
    bass_a = sidechain(bass_a, kick_4floor, nb, depth=0.55, release_ms=130)
    s += bass_a
    pad_voices = (
        render_pitched(pad_root,  pad, nb) +
        render_pitched(pad_third, pad, nb) +
        render_pitched(pad_fifth, pad, nb)
    )
    pad_voices = sidechain(pad_voices, kick_4floor, nb, depth=0.7, release_ms=160)
    s += pad_voices * 0.85
    # lead enters bar 4, with AUX REVERB
    lead_a = render_pitched(lead_hook, lead_synth, nb - 4)
    lead_a = aux_reverb(lead_a, decay=0.85, mix=0.45)
    lead_off = int(4 * SECONDS_PER_BAR * SAMPLE_RATE)
    s[lead_off:lead_off + len(lead_a)] += lead_a
    sections.append(s)

    # ---------- BREAKDOWN (16 bars, atmospheric + vocals) ----------
    print("  breakdown...")
    nb = 16
    s = np.zeros(int(nb * SECONDS_PER_BAR * SAMPLE_RATE))
    s += dome_drone(nb * SECONDS_PER_BAR)
    s += wind_texture(nb * SECONDS_PER_BAR) * 1.4
    # pad chord (no sidechain — no kick)
    pad_voices = (
        render_pitched(pad_root,  pad, nb) +
        render_pitched(pad_third, pad, nb) +
        render_pitched(pad_fifth, pad, nb)
    )
    s += pad_voices * 1.1
    # filtered lead echoes (heavy reverb, low-passed)
    lead_a = render_pitched(lead_hook, lead_synth, nb)
    lead_a = lowpass(lead_a, 1500)
    lead_a = aux_reverb(lead_a, decay=0.9, mix=0.7)
    s += lead_a * 0.55
    # VOCAL CHOPS — long sustains (bars 4-7 and 12-15)
    vox_a = render_pitched(vox_breakdown, vocal_chop, nb)
    vox_a = aux_reverb(vox_a, decay=0.85, mix=0.5)
    s += vox_a
    sections.append(s)

    # ---------- BUILD 2 (8 bars, biggest build) ----------
    print("  build 2...")
    nb = 8
    s = np.zeros(int(nb * SECONDS_PER_BAR * SAMPLE_RATE))
    s += dome_drone(nb * SECONDS_PER_BAR)
    pad_voices = (
        render_pitched(pad_root,  pad, nb) +
        render_pitched(pad_third, pad, nb) +
        render_pitched(pad_fifth, pad, nb)
    )
    s += pad_voices
    s += render_drum(hat_offbeat, hihat_closed, nb)
    # kick comes back light from bar 4
    kick_back = render_drum(kick_4floor, kick, 4) * 0.5
    kick_off = int(4 * SECONDS_PER_BAR * SAMPLE_RATE)
    s[kick_off:kick_off + len(kick_back)] += kick_back
    # long layered riser across full 8 bars
    riser_a = riser(nb * SECONDS_PER_BAR)
    s += riser_a
    # snare fill last bar (accelerating)
    fill_a = render_drum(snare_fill, snare, 1)
    fill_off = int(7 * SECONDS_PER_BAR * SAMPLE_RATE)
    s[fill_off:fill_off + len(fill_a)] += fill_a
    sections.append(s)

    # ---------- DROP 2 (24 bars, biggest payoff) ----------
    print("  drop 2...")
    nb = 24
    s = np.zeros(int(nb * SECONDS_PER_BAR * SAMPLE_RATE))
    # crash at start
    crash_hit = crash()
    s[:len(crash_hit)] += crash_hit
    # drone
    s += dome_drone(nb * SECONDS_PER_BAR)
    # full drums
    s += render_drum(kick_4floor, kick, nb)
    s += render_drum(hat_offbeat, hihat_closed, nb)
    s += render_drum(hat_open_pat, hihat_open, nb)
    s += render_drum(clap_pat, clap, nb)
    # bass + pad sidechained
    bass_a = render_pitched(bass_main, sub_bass, nb)
    bass_a = sidechain(bass_a, kick_4floor, nb, depth=0.55, release_ms=130)
    s += bass_a
    pad_voices = (
        render_pitched(pad_root,  pad, nb) +
        render_pitched(pad_third, pad, nb) +
        render_pitched(pad_fifth, pad, nb)
    )
    pad_voices = sidechain(pad_voices, kick_4floor, nb, depth=0.7, release_ms=160)
    s += pad_voices * 0.85
    # lead from bar 0 (no wait this time — full energy from start)
    lead_a = render_pitched(lead_hook, lead_synth, nb)
    lead_a = aux_reverb(lead_a, decay=0.85, mix=0.4)
    s += lead_a
    # NEW: counter-melody arp from bar 8 (the "lift" moment)
    counter_a = render_pitched(counter_arp, pluck, nb - 8)
    counter_a = sidechain(counter_a, kick_4floor, nb - 8,
                          depth=0.4, release_ms=100)
    counter_off = int(8 * SECONDS_PER_BAR * SAMPLE_RATE)
    s[counter_off:counter_off + len(counter_a)] += counter_a
    # NEW: vocal chop accents from bar 8 onward
    vox_a = render_pitched(vox_drop2, vocal_chop, nb - 8)
    vox_a = aux_reverb(vox_a, decay=0.8, mix=0.45)
    vox_off = int(8 * SECONDS_PER_BAR * SAMPLE_RATE)
    s[vox_off:vox_off + len(vox_a)] += vox_a
    sections.append(s)

    # ---------- OUTRO (12 bars, fadeout) ----------
    print("  outro...")
    nb = 12
    s = np.zeros(int(nb * SECONDS_PER_BAR * SAMPLE_RATE))
    s += dome_drone(nb * SECONDS_PER_BAR)
    s += wind_texture(nb * SECONDS_PER_BAR)
    pad_voices = (
        render_pitched(pad_root,  pad, nb) +
        render_pitched(pad_third, pad, nb) +
        render_pitched(pad_fifth, pad, nb)
    )
    s += pad_voices
    # last filtered lead phrase
    lead_a = render_pitched(lead_hook, lead_synth, nb)
    lead_a = lowpass(lead_a, 1200)
    lead_a = aux_reverb(lead_a, decay=0.9, mix=0.6)
    s += lead_a * 0.5
    # exponential fade
    fade = np.linspace(1, 0, len(s)) ** 1.8
    s *= fade
    sections.append(s)

    # ---------- MASTER BUS ----------
    master = np.concatenate(sections)
    print(f"  applying master bus...")
    master = reverb_master(master, decay=0.6, mix=0.13)
    master = saturate(master, drive=1.15)
    master = master / np.max(np.abs(master)) * 0.95
    return master


# ============================================================
# WRITE OUT
# ============================================================
if __name__ == "__main__":
    print("Rendering 'Perihelion Drift' v2 by Mission...")
    audio = render_song()
    audio_int = (audio * 32767).astype(np.int16)
    wavfile.write("perihelion_drift_v2.wav", SAMPLE_RATE, audio_int)
    duration = len(audio) / SAMPLE_RATE
    print(f"\n  → perihelion_drift_v2.wav  "
          f"({duration:.1f}s, {duration / 60:.2f}min)")
