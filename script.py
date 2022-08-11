import torch
from efficientnet_pytorch import EfficientNet

def take_model():
  MODEL_SAVE_PATH = '/content/drive/MyDrive/model/model.npy'


  def get_model():
      model_name = "efficientnet-b0"
      model = EfficientNet.from_pretrained(model_name, num_classes=10, in_channels=13)
      return model


  model = get_model()
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
  return model
