import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from .unet import UNet

def unet_infer(img_rgb, checkpoint_path=None, device='cpu', img_size=256):

    model = UNet(n_channels=3, n_classes=1, base=32).to(device)
    if checkpoint_path:
        try:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            print("Falha ao carregar checkpoint:", e)
    model.eval()
    pil = Image.fromarray(img_rgb)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    x = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(x)[0,0].cpu().numpy()
   
    prob_resized = np.array(Image.fromarray((prob*255).astype('uint8')).resize((img_rgb.shape[1], img_rgb.shape[0])))
    prob_resized = prob_resized.astype('float32') / 255.0
    return prob_resized
