import pyaudio
import numpy as np
import threading
import queue
import time
import os
from faster_whisper import WhisperModel

# ---------- Config ----------
SAMPLE_RATE = 16000
CHUNK_DURATION = 2        # seconds per chunk to transcribe
OVERLAP_DURATION = 0.5    # seconds of overlap for context
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_SIZE = int(SAMPLE_RATE * OVERLAP_DURATION)

AUDIO_FILE = "ENGLISH SPEECH  SUNDAR PICHAI The AI Moment (English Subtitles) - English Speeches.mp3"         # Set to a file path to transcribe a file, e.g. "audio.mp3"
                          # Set to None to use the microphone in real-time
# ----------------------------

model = WhisperModel("base", device="cpu", compute_type="int8")

audio_queue = queue.Queue()
buffer = np.zeros(0, dtype=np.float32)
stop_event = threading.Event()


def file_thread():
    """Reads an audio file, resamples to 16kHz mono float32, and pushes into queue."""
    try:
        import librosa
    except ImportError:
        print("❌ librosa not installed. Run: pip install librosa")
        stop_event.set()
        return

    print(f"📂 Loading file: {AUDIO_FILE}\n")
    audio, _ = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE, mono=True)  # auto-resamples

    # Push in 1024-sample chunks to mimic mic stream behavior
    pos = 0
    while pos < len(audio) and not stop_event.is_set():
        chunk = audio[pos:pos + 1024]
        audio_queue.put(chunk.astype(np.float32))
        pos += 1024
        time.sleep(0)  # yield to other threads

    # Signal end of file
    audio_queue.put(None)
    print("✅ File fully loaded into queue.")


def mic_thread():
    """Continuously reads from mic and pushes raw audio into queue."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024,
    )
    print("🎙️  Listening... (Ctrl+C to stop)\n")
    while not stop_event.is_set():
        data = stream.read(1024, exception_on_overflow=False)
        audio_queue.put(np.frombuffer(data, dtype=np.float32))
    stream.stop_stream()
    stream.close()
    p.terminate()


def transcribe_thread():
    """Accumulates audio chunks and transcribes when enough data is available."""
    global buffer
    while not stop_event.is_set():
        # Drain queue into buffer
        eof = False
        try:
            while True:
                chunk = audio_queue.get_nowait()
                if chunk is None:       # end-of-file sentinel
                    eof = True
                    break
                buffer = np.concatenate([buffer, chunk])
        except queue.Empty:
            pass

        if len(buffer) >= CHUNK_SIZE:
            audio_chunk = buffer[:CHUNK_SIZE]
            buffer = buffer[CHUNK_SIZE - OVERLAP_SIZE:]

            segments, _ = model.transcribe(
                audio_chunk,
                language="en",
                beam_size=1,
                best_of=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
            )
            text = "".join(s.text for s in segments).strip()
            if text:
                print(f"📝 {text}", flush=True)

        elif eof:
            # Transcribe whatever is left in the buffer
            if len(buffer) > 0:
                segments, _ = model.transcribe(
                    buffer,
                    language="en",
                    beam_size=1,
                    best_of=1,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                )
                text = "".join(s.text for s in segments).strip()
                if text:
                    print(f"📝 {text}", flush=True)
            print("\n⏹️  Transcription complete.")
            stop_event.set()
            break

        else:
            time.sleep(0.05)


if __name__ == "__main__":
    # Pick input source based on AUDIO_FILE
    if AUDIO_FILE:
        if not os.path.exists(AUDIO_FILE):
            print(f"❌ File not found: {AUDIO_FILE}")
            exit(1)
        source_thread = threading.Thread(target=file_thread, daemon=True)
    else:
        source_thread = threading.Thread(target=mic_thread, daemon=True)

    t2 = threading.Thread(target=transcribe_thread, daemon=True)
    source_thread.start()
    t2.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
        stop_event.set()
        # BEFORE (CPU)
model = WhisperModel("small", device="cpu", compute_type="int8")

# AFTER (GPU)
model = WhisperModel("small", device="cuda", compute_type="float16")
# Good balance
model = WhisperModel("small",    device="cuda", compute_type="float16")

# Best accuracy (runs well on GPU unlike CPU)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# If you get VRAM errors, drop to
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")