from tensorflow import keras
import torch
from basic_pitch.models import transcription_loss
from basic_pitch import ICASSP_2022_MODEL_PATH, note_creation as infer


with keras.utils.custom_object_scope({'<lambda>': transcription_loss}):
    model = keras.models.load_model(ICASSP_2022_MODEL_PATH)


layers_to_load = {
    "batch_normalization": "bn_layer",
    "conv2d_1": "conv_contour.0",
    "batch_normalization_2": "conv_contour.1",
    "contours-reduced": "conv_contour.3",
    "conv2d_2": "conv_note.0",
    "conv2d_4": "conv_onset_pre.0",
    "batch_normalization_3": "conv_onset_pre.1",
    "conv2d_3": "conv_note.2",
    "conv2d_5": "conv_onset_post.0",
}

print("===============")
state_dict = {}
for layer_name in layers_to_load:
    pt_layer_name = layers_to_load[layer_name]
    if "batch_norm" in layer_name:
        weights = torch.tensor(model.get_layer(layer_name).gamma.numpy())
        bias = torch.tensor(model.get_layer(layer_name).beta.numpy())
        running_mean = torch.tensor(model.get_layer(layer_name).moving_mean.numpy())
        running_var = torch.tensor(model.get_layer(layer_name).moving_variance.numpy())
        state_dict[pt_layer_name + ".weight"] = weights
        state_dict[pt_layer_name + ".bias"] = bias
        state_dict[pt_layer_name + ".running_mean"] = running_mean
        state_dict[pt_layer_name + ".running_var"] = running_var
        print(pt_layer_name + ".weight", weights.shape)
        print(pt_layer_name + ".bias", bias.shape)
        print(pt_layer_name + ".running_mean", running_mean.shape)
        print(pt_layer_name + ".running_var", running_var.shape)
    
    elif "conv" in layer_name or "contours" in layer_name:
        weights = torch.tensor(model.get_layer(layer_name).kernel.numpy()).permute(3, 2, 0, 1)
        bias = torch.tensor(model.get_layer(layer_name).bias.numpy())
        state_dict[pt_layer_name + ".weight"] = weights
        state_dict[pt_layer_name + ".bias"] = bias
        print(pt_layer_name + ".weight", weights.shape)
        print(pt_layer_name + ".bias", bias.shape)
    
    print("===============")

torch.save(state_dict, "assets/basic_pitch_pytorch_icassp_2022.pth")