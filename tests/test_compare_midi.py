from basic_pitch.inference import predict as predict_tf
from basic_pitch_torch.inference import predict as predict_pt


def test_midi_outputs():
    test_audio_path = "test_audio/00_BN1-129-Eb_solo_mic.wav"
    _, midi_data_tf, _ = predict_tf(test_audio_path)
    _, midi_data_pt, _ = predict_pt(test_audio_path)
    midi_data_tf.write("midi_data_tf.mid")
    midi_data_pt.write("midi_data_pt.mid")

    note_pt = sorted(midi_data_pt.instruments[0].notes, key = lambda x: x.start)
    note_tf = sorted(midi_data_tf.instruments[0].notes, key = lambda x: x.start)

    for i in range(len(note_pt)):
        assert note_pt[i].start == note_tf[i].start
        assert note_pt[i].end == note_tf[i].end
        assert note_pt[i].pitch == note_tf[i].pitch
        assert note_pt[i].velocity == note_tf[i].velocity
