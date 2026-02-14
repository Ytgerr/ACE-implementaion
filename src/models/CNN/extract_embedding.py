import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _to_tensor_normalized(patch: np.ndarray,
                          mean=IMAGENET_MEAN,
                          std=IMAGENET_STD) -> torch.Tensor:

    if patch.dtype == np.uint8:
        patch = patch.astype(np.float32) / 255.0
    else:
        patch = patch.astype(np.float32)
        if patch.max() > 1.5:   
            patch = patch / 255.0

    x = torch.from_numpy(patch).permute(2, 0, 1).contiguous().unsqueeze(0)  

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t  = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    return x

def extract_superpixel_embedding(patch: np.ndarray, model, device) -> torch.Tensor:
    model.eval()
    x = _to_tensor_normalized(patch).to(device)

    store = {}
    def hook_fn(module, inp, out):
        store["feat"] = out 

    h = model.features.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(x)
    h.remove()

    feat = store["feat"]
    emb = feat.mean(dim=(2, 3))        
    return emb.detach().cpu()
