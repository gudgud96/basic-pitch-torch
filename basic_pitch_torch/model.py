from typing import List
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from nnAudio.features import CQT2010v2
import numpy as np
from basic_pitch_torch.constants import *


def log_base_b(x: Tensor, base: int) -> Tensor:
    """
    Compute log_b(x)
    Args:
        x : input
        base : log base. E.g. for log10 base=10
    Returns:
        log_base(x)
    """
    numerator = torch.log(x)
    denominator = torch.log(torch.tensor([base], dtype=numerator.dtype, device=numerator.device))
    return numerator / denominator


def normalized_log(inputs: Tensor) -> Tensor:
    """
    Takes an input with a shape of either (batch, x, y, z) or (batch, y, z)
    and rescales each (y, z) to dB, scaled 0 - 1.
    Only x=1 is supported.
    This layer adds 1e-10 to all values as a way to avoid NaN math.
    """
    power = torch.square(inputs)
    log_power = 10 * log_base_b(power + 1e-10, 10)

    log_power_min = torch.amin(log_power, dim=(1, 2)).reshape(inputs.shape[0], 1, 1)
    log_power_offset = log_power - log_power_min    
    log_power_offset_max = torch.amax(log_power_offset, dim=(1, 2)).reshape(inputs.shape[0], 1, 1)
    # equivalent to TF div_no_nan
    log_power_normalized = log_power_offset / log_power_offset_max
    log_power_normalized = torch.nan_to_num(log_power_normalized, nan=0.0)

    return log_power_normalized.reshape(inputs.shape)


def get_cqt(
        inputs: Tensor, 
        n_harmonics: int, 
        use_batch_norm: bool, 
        bn_layer: nn.BatchNorm2d, 
    ):
    """Calculate the CQT of the input audio.

    Input shape: (batch, number of audio samples, 1)
    Output shape: (batch, number of frequency bins, number of time frames)

    Args:
        inputs: The audio input.
        n_harmonics: The number of harmonics to capture above the maximum output frequency.
            Used to calculate the number of semitones for the CQT.
        use_batchnorm: If True, applies batch normalization after computing the CQT

    Returns:
        The log-normalized CQT of the input audio.
    """
    n_semitones = np.min(
        [
            int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
            MAX_N_SEMITONES,
        ]
    )
    cqt_layer = CQT2010v2(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
        verbose=False,
    )
    cqt_layer.to(inputs.device)
    x = cqt_layer(inputs)
    x = torch.transpose(x, 1, 2)
    x = normalized_log(x)
    
    x = x.unsqueeze(1)
    if use_batch_norm:
        x = bn_layer(x)
    x = x.squeeze(1)
    
    return x


class HarmonicStacking(nn.Module):
    """Harmonic stacking layer

    Input shape: (n_batch, n_times, n_freqs, 1)
    Output shape: (n_batch, n_times, n_output_freqs, len(harmonics))

    n_freqs should be much larger than n_output_freqs so that information from the upper
    harmonics is captured.

    Attributes:
        bins_per_semitone: The number of bins per semitone of the input CQT
        harmonics: List of harmonics to use. Should be positive numbers.
        shifts: A list containing the number of bins to shift in frequency for each harmonic
        n_output_freqs: The number of frequency bins in each harmonic layer.
    """

    def __init__(
        self, 
        bins_per_semitone: int, 
        harmonics: List[float], 
        n_output_freqs: int,
    ):
        super().__init__()
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.n_output_freqs = n_output_freqs

        self.shifts = [
            int(round(12.0 * self.bins_per_semitone * math.log2(h))) for h in self.harmonics
        ]
    
    @torch.no_grad()
    def forward(self, x):
        # x: (batch, t, n_bins)
        hcqt = []
        for shift in self.shifts:
            if shift == 0:
                cur_cqt = x
            if shift > 0:
                cur_cqt = F.pad(x[:, :, shift:], (0, shift))
            elif shift < 0:     # sub-harmonic
                cur_cqt = F.pad(x[:, :, :shift], (-shift, 0))
            hcqt.append(cur_cqt)
        hcqt = torch.stack(hcqt, dim=1)
        hcqt = hcqt[:, :, :, :self.n_output_freqs]
        return hcqt


class BasicPitchTorch(nn.Module):
    def __init__(
        self, 
        stack_harmonics=[0.5, 1, 2, 3, 4, 5, 6, 7],
    ) -> None:
        super().__init__()
        self.stack_harmonics = stack_harmonics
        if len(stack_harmonics) > 0:
            self.hs = HarmonicStacking(
                bins_per_semitone=CONTOURS_BINS_PER_SEMITONE, 
                harmonics=stack_harmonics, 
                n_output_freqs=ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE
            )
            num_in_channels = len(stack_harmonics)
        else:
            num_in_channels = 1

        self.bn_layer = nn.BatchNorm2d(1, eps=0.001)
        self.conv_contour = nn.Sequential(
            # NOTE: in the original implementation, this part of the network should be dangling...
            # nn.Conv2d(num_in_channels, 32, kernel_size=5, padding="same"),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Conv2d(num_in_channels, 8, kernel_size=(3, 3 * 13), padding="same"),
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=5, padding="same"),
            nn.Sigmoid()
        )
        self.conv_note = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(7, 3), padding="same"),
            nn.Sigmoid()
        )
        self.conv_onset_pre = nn.Sequential(
            nn.Conv2d(num_in_channels, 32, kernel_size=5, stride=(1, 3)),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(),
        )
        self.conv_onset_post = nn.Sequential(
            nn.Conv2d(32 + 1, 1, kernel_size=3, stride=1, padding="same"),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cqt = get_cqt(
            x, 
            len(self.stack_harmonics), 
            True, 
            self.bn_layer,
        )
        if hasattr(self, "hs"):
            cqt = self.hs(cqt)
        else:
            cqt = cqt.unsqueeze(1)
                
        x_contour = self.conv_contour(cqt)

        # for strided conv, padding is different between PyTorch and TensorFlow
        # we use this equation: pad = [ (stride * (output-1)) - input + kernel ] / 2
        # (172, 264) --(1, 3)--> (172, 88), pad = ((1 * 171 - 172 + 7) / 2, (3 * 87 - 264 + 7) / 2) = (3, 2)
        # F.pad process from the last dimension, so it's (2, 2, 3, 3)
        x_contour_for_note = F.pad(x_contour, (2,2,3,3))
        x_note = self.conv_note(x_contour_for_note)
        
        # (172, 264) --(1, 3)--> (172, 88), pad = ((1 * 171 - 172 + 5) / 2, (3 * 87 - 264 + 5) / 2) = (2, 1)
        # F.pad process from the last dimension, so it's (1, 1, 2, 2)
        cqt_for_onset = F.pad(cqt, (1,1,2,2))
        x_onset_pre = self.conv_onset_pre(cqt_for_onset)
        x_onset_pre = torch.cat([x_note, x_onset_pre], dim=1)
        x_onset = self.conv_onset_post(x_onset_pre)

        outputs = {"onset": x_onset.squeeze(1), "contour": x_contour.squeeze(1), "note": x_note.squeeze(1)}
        return outputs