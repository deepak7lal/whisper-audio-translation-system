import sounddevice as sd
import numpy as np
import threading
import queue
import time
import collections
import noisereduce as nr
import webrtcvad
from faster_whisper import WhisperModel
import os

# ============================================================
#  CONFIG — tweak these to your machine + environment
# ============================================================
SAMPLE_RATE       = 16000
CPU_THREADS       = os.cpu_count()   # use all cores
CHUNK_DURATION    = 6                # seconds of audio per transcription window
OVERLAP_DURATION  = 1.2              # seconds of overlap to avoid cut-off words
CHUNK_SIZE        = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_SIZE      = int(SAMPLE_RATE * OVERLAP_DURATION)

# WebRTC VAD aggressiveness: 0 (least aggressive) → 3 (most aggressive)
# 1 works well for normal office environments
# raise to 2-3 if you have a lot of background noise
VAD_AGGRESSIVENESS = 1

# RMS silence gate — skip chunks quieter than this (prevents hallucinations)
SILENCE_RMS_THRESHOLD = 0.008

# Noise reduction strength: 0.0 (off) → 1.0 (max)
# 0.7 is a good balance — too high removes speech too
NOISE_REDUCTION_STRENGTH = 0.7

# Whisper model: "tiny", "base", "small" — small is the sweet spot for CPU realtime
MODEL_SIZE = "small"
# ============================================================

print(f"⚙️  Loading Whisper '{MODEL_SIZE}' on CPU with {CPU_THREADS} threads...")
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=CPU_THREADS,
    num_workers=2,
)
print("✅ Model loaded.\n")

audio_queue  = queue.Queue()
stop_event   = threading.Event()

# WebRTC VAD works on 10/20/30ms frames of 16-bit PCM
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
VAD_FRAME_MS      = 30                              # 30ms frames
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 samples


# ── helpers ──────────────────────────────────────────────────

def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def has_speech_webrtcvad(audio_f32: np.ndarray) -> bool:
    """
    Returns True if WebRTC VAD detects speech in at least 20% of frames.
    More reliable than a simple RMS threshold.
    """
    pcm_i16 = (audio_f32 * 32767).astype(np.int16)
    speech_frames = 0
    total_frames  = 0
    for start in range(0, len(pcm_i16) - VAD_FRAME_SAMPLES, VAD_FRAME_SAMPLES):
        frame = pcm_i16[start:start + VAD_FRAME_SAMPLES].tobytes()
        total_frames += 1
        if vad.is_speech(frame, SAMPLE_RATE):
            speech_frames += 1
    if total_frames == 0:
        return False
    return (speech_frames / total_frames) > 0.20   # at least 20% speech frames


def reduce_noise(audio: np.ndarray) -> np.ndarray:
    """
    Uses the first 0.3s of the chunk as a noise profile to reduce background noise.
    Works best when the speaker doesn't start talking immediately.
    """
    noise_sample = audio[:int(SAMPLE_RATE * 0.3)]
    denoised = nr.reduce_noise(
        y=audio,
        sr=SAMPLE_RATE,
        y_noise=noise_sample,
        prop_decrease=NOISE_REDUCTION_STRENGTH,
        stationary=False,    # handles non-stationary noise (fans, traffic, etc.)
    )
    return denoised.astype(np.float32)


def deduplicate(new_text: str, prev_text: str) -> str:
    """
    Strips leading words from new_text that overlap with the end of prev_text.
    Prevents the same phrase being printed twice due to overlapping audio chunks.
    """
    if not prev_text:
        return new_text
    prev_words = prev_text.lower().split()
    new_words  = new_text.split()
    # Try to find overlap of up to 8 words
    for overlap in range(min(8, len(prev_words), len(new_words)), 0, -1):
        if prev_words[-overlap:] == [w.lower() for w in new_words[:overlap]]:
            new_words = new_words[overlap:]
            break
    return " ".join(new_words)


# ── threads ──────────────────────────────────────────────────

def mic_thread():
    """Captures microphone audio and pushes float32 mono chunks into the queue."""
    print("🎙️  Microphone active — speak now...\n")

    def callback(indata, frames, time_info, status):
        if status:
            print(f"⚠️  sounddevice status: {status}", flush=True)
        audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=1024,
        callback=callback,
    ):
        while not stop_event.is_set():
            time.sleep(0.05)


def transcribe_thread():
    """
    Accumulates mic audio into a rolling buffer.
    When enough audio is collected:
      1. RMS gate  — skip obvious silence
      2. WebRTC VAD — skip chunks with no detected speech
      3. Noise reduction — clean up the audio
      4. Whisper transcription
      5. Deduplication — remove repeated words from overlapping chunks
    """
    buffer    = np.zeros(0, dtype=np.float32)
    last_text = ""

    while not stop_event.is_set():
        # Drain everything currently in the queue into the buffer
        try:
            while True:
                buffer = np.concatenate([buffer, audio_queue.get_nowait()])
        except queue.Empty:
            pass

        if len(buffer) < CHUNK_SIZE:
            time.sleep(0.05)
            continue

        audio_chunk = buffer[:CHUNK_SIZE]
        buffer      = buffer[CHUNK_SIZE - OVERLAP_SIZE:]   # keep overlap

        # ── Gate 1: RMS silence check (very fast) ──
        if rms(audio_chunk) < SILENCE_RMS_THRESHOLD:
            continue

        # ── Gate 2: WebRTC VAD (fast, catches non-speech noise) ──
        if not has_speech_webrtcvad(audio_chunk):
            continue

        # ── Gate 3: Noise reduction ──
        clean_audio = reduce_noise(audio_chunk)

        # ── Gate 4: Whisper transcription ──
        t0 = time.time()
        segments, _ = model.transcribe(
            clean_audio,
            language="en",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt="Transcribe the following speech accurately, preserving punctuation.",
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.25,
                min_silence_duration_ms=400,
                min_speech_duration_ms=100,
                speech_pad_ms=100,
            ),
            without_timestamps=True,
        )
        elapsed = time.time() - t0

        raw_text = "".join(s.text for s in segments).strip()
        if not raw_text:
            continue

        # ── Gate 5: Deduplication ──
        clean_text = deduplicate(raw_text, last_text)
        if not clean_text.strip():
            continue

        last_text = raw_text
        print(f"📝  {clean_text}   ⏱️ {elapsed:.2f}s", flush=True)


# ── main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    t1 = threading.Thread(target=mic_thread,      daemon=True)
    t2 = threading.Thread(target=transcribe_thread, daemon=True)
    t1.start()
    t2.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped.")
        stop_event.set()