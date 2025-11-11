import torch
import torchvision
import numpy as np
import torchvision.transforms as T

def maskrcnn_infer(img_rgb, score_threshold=0.5, device='cpu'):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device).eval()
    transform = T.Compose([T.ToTensor()])
    img_t = transform(img_rgb).to(device)
    with torch.no_grad():
        outputs = model([img_t])[0]
    masks = outputs.get('masks')  
    scores = outputs.get('scores')
    results = []
    result_masks = []
    out_scores = []
    if masks is None:
        return [], []
    for i in range(len(masks)):
        s = float(scores[i].cpu().numpy())
        if s < score_threshold:
            continue
        m = masks[i,0].cpu().numpy()
        bin_mask = (m > 0.5).astype(np.uint8)
        result_masks.append(bin_mask)
        out_scores.append(s)
    return result_masks, out_scores
