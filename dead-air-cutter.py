import os
import argparse
import subprocess
import tempfile
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
import numpy as np


# ---------- FFMPEG UTILS ----------

def run_ffmpeg(args):
    cmd = ["ffmpeg", "-y", *args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_audio(input_video, audio_path, sr=16000):
    # mono, 16kHz wav
    run_ffmpeg([
        "-i", input_video,
        "-ac", "1",
        "-ar", str(sr),
        audio_path
    ])


def get_audio_duration(audio_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    return float(result.stdout.strip())


# ---------- AUDIO / VAD ----------

import numpy as np
import soundfile as sf
import torch

def load_audio(audio_path, target_sr=16000):
    wav, sr = sf.read(audio_path)  # numpy array

    # mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # resample if needed
    if sr != target_sr:
        x_old = np.linspace(0, 1, num=len(wav), endpoint=False)
        num_new = int(len(wav) * target_sr / sr)
        x_new = np.linspace(0, 1, num=num_new, endpoint=False)
        wav = np.interp(x_new, x_old, wav)

        sr = target_sr

    # ensure float32
    wav = wav.astype(np.float32)

    wav_t = torch.from_numpy(wav)          # dtype=float32
    return wav_t, sr


def bandlimit_for_speech(wav, sr, low=300, high=3400, order=4):
    import numpy.fft as fft
    import numpy as np
    import torch

    x = wav.detach().cpu().numpy().astype(np.float32)
    n = len(x)

    freqs = fft.rfftfreq(n, d=1.0 / sr)
    X = fft.rfft(x)

    mask = (freqs >= low) & (freqs <= high)
    X_filtered = X * mask

    x_filtered = fft.irfft(X_filtered, n=n).astype(np.float32)

    return torch.from_numpy(x_filtered)


def load_silero_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False
    )
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    return model, get_speech_timestamps


def detect_speech_timestamps(wav, sr, threshold=0.5):
    model, get_speech_timestamps = load_silero_vad()

    print("wav dtype:", wav.dtype)
    for p in model.parameters():
        print("model param dtype:", p.dtype)
        break


    # enforce float32 for both input and model
    wav = wav.to(torch.float32)
    model = model.to(torch.float32)

    speech_ts = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=threshold,
    )
    segments = [
        (ts["start"] / sr, ts["end"] / sr)
        for ts in speech_ts
    ]
    return segments


# ---------- SEGMENT OPS ----------

def merge_close_segments(segments, min_gap):
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged = []
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        if start - cur_end <= min_gap:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def pad_and_filter_segments(segments, pre_roll, post_roll, min_len, total_dur):
    padded = []
    for start, end in segments:
        s = max(0.0, start - pre_roll)
        e = min(total_dur, end + post_roll)
        if e - s >= min_len:
            padded.append((s, e))
    return padded


def intersect_segments(seg_a, seg_b):
    # seg_a, seg_b: lists of (start, end) sorted
    i, j = 0, 0
    result = []
    while i < len(seg_a) and j < len(seg_b):
        a_start, a_end = seg_a[i]
        b_start, b_end = seg_b[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if end > start:
            result.append((start, end))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result


# ---------- CONCAT FILE + RENDER ----------

def write_concat_file(segments, concat_path, input_video_abs):
    with open(concat_path, "w", encoding="utf-8") as f:
        for start, end in segments:
            f.write(f"file '{input_video_abs}'\n")
            f.write(f"inpoint {start:.3f}\n")
            f.write(f"outpoint {end:.3f}\n")


def render_final_video(concat_path, output_video):
    run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", concat_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "aresample=async=1:first_pts=0",
        output_video,
    ])

#    run_ffmpeg([
#        "-f", "concat",
#        "-safe", "0",
#        "-i", concat_path,
#        "-c:v", "libx264",
#        "-preset", "veryfast",
#        "-crf", "18",
#        "-c:a", "aac",
#        "-b:a", "192k",
#        output_video,
#    ])



# ---------- MAIN PIPELINE ----------

def main():
    parser = argparse.ArgumentParser(
        description="Cut a video to only keep human speech segments using VAD."
    )
    parser.add_argument("input_video", help="Input video file (e.g., input.mp4)")
    parser.add_argument("output_video", help="Output video file (e.g., talk_only.mp4)")

    parser.add_argument("--pre-roll", type=float, default=0.25,
                        help="Padding before each speech segment in seconds (default: 0.25)")
    parser.add_argument("--post-roll", type=float, default=0.25,
                        help="Padding after each speech segment in seconds (default: 0.25)")
    parser.add_argument("--min-segment-len", type=float, default=0.4,
                        help="Minimum kept segment length in seconds (default: 0.4)")
    parser.add_argument("--min-gap-len", type=float, default=0.4,
                        help="Minimum gap to consider a separate segment in seconds (default: 0.4)")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="Silero VAD threshold (0–1, higher = more strict, default: 0.5)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate for VAD (default: 16000)")
    parser.add_argument("--two-pass", action="store_true",
                        help="Run a second VAD pass on band-limited (speech-focused) audio and intersect segments.")

    args = parser.parse_args()

    input_video = args.input_video
    output_video = args.output_video

    if not Path(input_video).exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_path = tmpdir / "audio.wav"

        print("Extracting audio...")
        extract_audio(input_video, str(audio_path), sr=args.sample_rate)

        print("Loading audio...")
        wav, sr = load_audio(str(audio_path), target_sr=args.sample_rate)

        print("VAD pass 1 (raw)...")
        seg1 = detect_speech_timestamps(wav, sr, threshold=args.vad_threshold)
        seg1 = merge_close_segments(seg1, args.min_gap_len)
        print(f"Pass 1 merged segments: {len(seg1)}")

        if args.two_pass:
            print("Band-limiting audio for speech focus...")
            wav_band = bandlimit_for_speech(wav, sr)

            print("VAD pass 2 (band-limited)...")
            seg2 = detect_speech_timestamps(wav_band, sr, threshold=args.vad_threshold)
            seg2 = merge_close_segments(seg2, args.min_gap_len)
            print(f"Pass 2 merged segments: {len(seg2)}")

            print("Intersecting segments from both passes...")
            seg_final = intersect_segments(seg1, seg2)
        else:
            seg_final = seg1

        seg_final = merge_close_segments(seg_final, args.min_gap_len)
        total_dur = get_audio_duration(str(audio_path))
        seg_final = pad_and_filter_segments(
            seg_final,
            args.pre_roll,
            args.post_roll,
            args.min_segment_len,
            total_dur
        )
        print(f"Final segments after padding/filter: {len(seg_final)}")

        if not seg_final:
            print("No speech segments found after processing; aborting.")
            return

        concat_path = tmpdir / "concat.txt"
        write_concat_file(seg_final, concat_path, os.path.abspath(input_video))

        print("Rendering final video...")
        render_final_video(str(concat_path), output_video)

    print("Done. Output:", output_video)


if __name__ == "__main__":
    main()

