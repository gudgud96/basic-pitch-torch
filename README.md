## basic-pitch-torch

PyTorch version of Spotify's [Basic Pitch](https://github.com/spotify/basic-pitch), a lightweight audio-to-MIDI converter.
The [provided weights](https://github.com/spotify/basic-pitch/tree/main/basic_pitch/saved_models/icassp_2022/nmp) in Spotify's repo are converted using [this script](./scripts/convert_tf_to_pt.py). Hopefully this helps researchers who are more accustomed to PyTorch to re-use the pretrained model.

### Usage

For transcribing MIDI files, similar to Basic Pitch:
```python
from basic_pitch_torch.inference import predict

model_output, midi_data, note_events = predict(audio_path)
```

For loading the `nn.Module`:
```python
from basic_pitch_torch.model import BasicPitchTorch

pt_model = BasicPitchTorch()
pt_model.load_state_dict(torch.load('assets/basic_pitch_pytorch_icassp_2022.pth'))
pt_model.eval()

with torch.no_grad():
    output_pt = pt_model(y_torch)
    contour_pt, note_pt, onset_pt = output_pt['contour'], output_pt['note'], output_pt['onset']
```

### Result Validation

In `tests/` we show two levels of validation tests using a test audio from [GuitarSet](https://guitarset.weebly.com/): 

 - **On model output** 
    - Most of the discrepancies originated from float division (e.g. `normalized_log`) and error propagation further down the network. The difference should be minimal enough to be ignored during MIDI note creation.

    ```
    Contour abs diff - max: 0.0003006, min: 0.0, avg: 5.863e-06
    Onset abs diff - max: 0.0002712, min: 0.0, avg: 1.431e-05
    Note abs diff - max: 0.0002297, min: 0.0, avg: 6.6e-06
    ```
    
 - **On MIDI transcription**
    - The transcribed MIDI using both TF and PT models are identical (see `midi_data_pt.mid` and `midi_data_tf.mid`)



