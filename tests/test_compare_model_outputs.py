from tensorflow import keras
from basic_pitch.models import transcription_loss
from basic_pitch import ICASSP_2022_MODEL_PATH
import librosa
import numpy as np

import torch
from basic_pitch_torch.model import BasicPitchTorch
from basic_pitch_torch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE
import matplotlib.pyplot as plt

with keras.utils.custom_object_scope({'<lambda>': transcription_loss}):
    tf_model = keras.models.load_model(ICASSP_2022_MODEL_PATH)

pt_model = BasicPitchTorch()
pt_model.load_state_dict(torch.load('assets/basic_pitch_pytorch_icassp_2022.pth'))
pt_model.eval()
if torch.cuda.is_available():
    pt_model.cuda()


def test_model_output():
    y, _ = librosa.load('test_audio/00_BN1-129-Eb_solo_mic.wav', sr=AUDIO_SAMPLE_RATE, mono=True)
    y = y[:AUDIO_N_SAMPLES].reshape(1, -1)
    y_torch = torch.tensor(y)
    if torch.cuda.is_available():
        y_torch = y_torch.cuda()

    output = tf_model.predict(y)
    contour, note, onset = output['contour'], output['note'], output['onset']

    with torch.no_grad():
        output_pt = pt_model(y_torch)
        contour_pt, note_pt, onset_pt = output_pt['contour'], output_pt['note'], output_pt['onset']

    # Uncomment if you want to save the plots
    # plt.imshow(contour.squeeze().T, aspect='auto', interpolation="none")
    # plt.savefig("contour_tf.png")
    # plt.close()
    # plt.imshow(contour_pt.squeeze().cpu().detach().numpy().T, aspect='auto', interpolation="none")
    # plt.savefig("contour_pt.png")
    # plt.close()
    # plt.imshow(np.abs(contour_pt.squeeze().cpu().detach().numpy().T - contour.squeeze().T), aspect='auto', interpolation="none")
    # plt.savefig("contour_pt_diff.png")
    # plt.close()

    # plt.imshow(note.squeeze().T, aspect='auto', interpolation="none")
    # plt.savefig("note_tf.png")
    # plt.close()
    # plt.imshow(note_pt.squeeze().cpu().detach().numpy().T, aspect='auto', interpolation="none")
    # plt.savefig("note_pt.png")
    # plt.close()

    # plt.imshow(onset.squeeze().T, aspect='auto', interpolation="none")
    # plt.savefig("onset_tf.png")
    # plt.close()
    # plt.imshow(onset_pt.squeeze().cpu().detach().numpy().T, aspect='auto', interpolation="none")
    # plt.savefig("onset_pt.png")
    # plt.close()

    diff = np.abs(contour.squeeze() - contour_pt.squeeze().cpu().detach().numpy())
    print(f"Contour abs diff - max: {np.max(diff):.4}, min: {np.min(diff):.4}, avg: {np.mean(diff):.4}")
    diff = np.abs(onset.squeeze() - onset_pt.squeeze().cpu().detach().numpy())
    print(f"Onset abs diff - max: {np.max(diff):.4}, min: {np.min(diff):.4}, avg: {np.mean(diff):.4}")
    diff = np.abs(note.squeeze() - note_pt.squeeze().cpu().detach().numpy())
    print(f"Note abs diff - max: {np.max(diff):.4}, min: {np.min(diff):.4}, avg: {np.mean(diff):.4}")

    atol = 6e-4
    assert np.allclose(contour.squeeze(), contour_pt.squeeze().cpu().detach().numpy(), atol=atol)
    assert np.allclose(onset.squeeze(), onset_pt.squeeze().cpu().detach().numpy(), atol=atol)
    assert np.allclose(note.squeeze(), note_pt.squeeze().cpu().detach().numpy(), atol=atol)