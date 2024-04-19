#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import enum
import json
import os
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from basic_pitch_torch.model import BasicPitchTorch
import numpy as np
import librosa
import pretty_midi

from basic_pitch_torch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
from basic_pitch_torch import note_creation as infer


def frame_with_pad(x: np.array, frame_length: int, hop_size: int) -> np.array:
    """
    Extends librosa.util.frame with end padding if required, similar to 
    tf.signal.frame(pad_end=True).

    Returns:
        framed_audio: tensor with shape (n_windows, AUDIO_N_SAMPLES)
    """
    n_frames = int(np.ceil((x.shape[0] - frame_length) / hop_size)) + 1
    n_pads = (n_frames - 1) * hop_size + frame_length - x.shape[0]
    x = np.pad(x, (0, n_pads), mode="constant")
    framed_audio = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_size)
    return framed_audio


def window_audio_file(audio_original: np.array, hop_size: int) -> Tuple[np.array, List[Dict[str, int]]]:
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    audio_windowed = frame_with_pad(audio_original, AUDIO_N_SAMPLES, hop_size)
    window_times = [
        {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        for t_start in np.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
    ]
    return audio_windowed, window_times


def get_audio_input(
    audio_path: Union[pathlib.Path, str], overlap_len: int, hop_size: int
) -> Tuple[Tensor, List[Dict[str, int]], int]:
    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)
        audio_original_length: int
            length of original audio file, in frames, BEFORE padding.

    """
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)

    audio_original, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    audio_windowed, window_times = window_audio_file(audio_original, hop_size)
    return audio_windowed, window_times, original_length


def unwrap_output(output: Tensor, audio_original_length: int, n_overlapping_frames: int) -> np.array:
    """Unwrap batched model predictions to a single matrix.

    Args:
        output: array (n_batches, n_times_short, n_freqs)
        audio_original_length: length of original audio signal (in samples)
        n_overlapping_frames: number of overlapping frames in the output

    Returns:
        array (n_times, n_freqs)
    """
    raw_output = output.cpu().detach().numpy()
    if len(raw_output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        raw_output = raw_output[:, n_olap:-n_olap, :]

    output_shape = raw_output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = raw_output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length


def run_inference(
    audio_path: Union[pathlib.Path, str],
    model: nn.Module,
    debug_file: Optional[pathlib.Path] = None,
) -> Dict[str, np.array]:
    """Run the model on the input audio path.

    Args:
        audio_path: The audio to run inference on.
        model: A loaded keras model to run inference with.
        debug_file: An optional path to output debug data to. Useful for testing/verification.

    Returns:
       A dictionary with the notes, onsets and contours from model inference.
    """
    # overlap 30 frames
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    audio_windowed, _, audio_original_length = get_audio_input(audio_path, overlap_len, hop_size)
    audio_windowed = torch.from_numpy(audio_windowed).T
    if torch.cuda.is_available():
        audio_windowed = audio_windowed.cuda()

    output = model(audio_windowed)
    unwrapped_output = {k: unwrap_output(output[k], audio_original_length, n_overlapping_frames) for k in output}

    if debug_file:
        with open(debug_file, "w") as f:
            json.dump(
                {
                    "audio_windowed": audio_windowed.numpy().tolist(),
                    "audio_original_length": audio_original_length,
                    "hop_size_samples": hop_size,
                    "overlap_length_samples": overlap_len,
                    "unwrapped_output": {k: v.tolist() for k, v in unwrapped_output.items()},
                },
                f,
            )

    return unwrapped_output


class OutputExtensions(enum.Enum):
    MIDI = "mid"
    MODEL_OUTPUT_NPZ = "npz"
    MIDI_SONIFICATION = "wav"
    NOTE_EVENTS = "csv"


def verify_input_path(audio_path: Union[pathlib.Path, str]) -> None:
    """Verify that an input path is valid and can be processed

    Args:
        audio_path: Path to an audio file.

    Raises:
        ValueError: If the audio file is invalid.
    """
    if not os.path.isfile(audio_path):
        raise ValueError(f"🚨 {audio_path} is not a file path.")

    if not os.path.exists(audio_path):
        raise ValueError(f"🚨 {audio_path} does not exist.")


def verify_output_dir(output_dir: Union[pathlib.Path, str]) -> None:
    """Verify that an output directory is valid and can be processed

    Args:
        output_dir: Path to an output directory.

    Raises:
        ValueError: If the output directory is invalid.
    """
    if not os.path.isdir(output_dir):
        raise ValueError(f"🚨 {output_dir} is not a directory.")

    if not os.path.exists(output_dir):
        raise ValueError(f"🚨 {output_dir} does not exist.")


def build_output_path(
    audio_path: Union[pathlib.Path, str],
    output_directory: Union[pathlib.Path, str],
    output_type: OutputExtensions,
) -> pathlib.Path:
    """Create an output path and make sure it doesn't already exist.

    Args:
        audio_path: The original file path.
        output_directory: The directory we will output to.
        output_type: The type of output file we are creating.

    Raises:
        IOError: If the generated path already exists.

    Returns:
        A new path in the output_directory with the stem audio_path and an extension
        based on output_type.
    """
    audio_path = str(audio_path)
    if not isinstance(output_directory, pathlib.Path):
        output_directory = pathlib.Path(output_directory)

    basename, _ = os.path.splitext(os.path.basename(audio_path))

    output_path = output_directory / f"{basename}_basic_pitch.{output_type.value}"

    if output_path.exists():
        raise IOError(
            f"  🚨 {str(output_path)} already exists and would be overwritten. Skipping output files for {audio_path}."
        )

    return output_path


def save_note_events(
    note_events: List[Tuple[float, float, int, float, Optional[List[int]]]],
    save_path: Union[pathlib.Path, str],
) -> None:
    """Save note events to file

    Args:
        note_events: A list of note event tuples to save. Tuples have the format
            ("start_time_s", "end_time_s", "pitch_midi", "velocity", "list of pitch bend values")
        save_path: The location we're saving it
    """

    with open(save_path, "w") as fhandle:
        writer = csv.writer(fhandle, delimiter=",")
        writer.writerow(["start_time_s", "end_time_s", "pitch_midi", "velocity", "pitch_bend"])
        for start_time, end_time, note_number, amplitude, pitch_bend in note_events:
            row = [start_time, end_time, note_number, int(np.round(127 * amplitude))]
            if pitch_bend:
                row.extend(pitch_bend)
            writer.writerow(row)


def predict(
    audio_path: Union[pathlib.Path, str],
    model_path: Union[pathlib.Path, str] = "assets/basic_pitch_pytorch_icassp_2022.pth",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.70,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    debug_file: Optional[pathlib.Path] = None,
    midi_tempo: float = 120,
) -> Tuple[Dict[str, np.array], pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]],]:
    """Run a single prediction.

    Args:
        audio_path: File path for the audio to run inference on.
        model_or_model_path: Path to load the Keras saved model from. Can be local or on GCS.
        onset_threshold: Minimum energy required for an onset to be considered present.
        frame_threshold: Minimum energy requirement for a frame to be considered present.
        minimum_note_length: The minimum allowed note length in milliseconds.
        minimum_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        maximum_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.
        debug_file: An optional path to output debug data to. Useful for testing/verification.
    Returns:
        The model output, midi data and note events from a single prediction
    """
    model = BasicPitchTorch()
    model.load_state_dict(torch.load(str(model_path)))
    model.eval() 
    if torch.cuda.is_available():
        model.cuda()

    print(f"Predicting MIDI for {audio_path}...")

    model_output = run_inference(audio_path, model, debug_file)
    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi_data, note_events = infer.model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,  # convert to frames
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
        multiple_pitch_bends=multiple_pitch_bends,
        melodia_trick=melodia_trick,
        midi_tempo=midi_tempo,
    )

    if debug_file:
        with open(debug_file) as f:
            debug_data = json.load(f)
        with open(debug_file, "w") as f:
            json.dump(
                {
                    **debug_data,
                    "min_note_length": min_note_len,
                    "onset_thresh": onset_threshold,
                    "frame_thresh": frame_threshold,
                    "estimated_notes": [
                        (
                            float(start_time),
                            float(end_time),
                            int(pitch),
                            float(amplitude),
                            [int(b) for b in pitch_bends] if pitch_bends else None,
                        )
                        for start_time, end_time, pitch, amplitude, pitch_bends in note_events
                    ],
                },
                f,
            )

    return model_output, midi_data, note_events