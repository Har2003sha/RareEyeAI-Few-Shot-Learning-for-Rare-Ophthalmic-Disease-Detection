"""
Grad-CAM (Selvaraju et al., 2017) adapted for a Prototypical Network.

Instead of backpropagating a softmax class logit (as in a normal
classifier), we backpropagate the *similarity score* of the query image to
its predicted class prototype (negative squared distance in embedding
space). This highlights which regions of the retinal image most increased
the image's embedding similarity to the predicted rare-disease prototype -
the natural analogue of Grad-CAM for metric-based few-shot learning.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ml.network import get_embedding_net


def generate_gradcam(image_tensor, prototype_embedding, embedding_dim=64, img_size=None):
    net = get_embedding_net(embedding_dim)
    net.zero_grad()

    image_tensor = image_tensor.clone().requires_grad_(True)
    embedding = net(image_tensor, register_hook=True)  # (1, embedding_dim)

    # similarity score to the predicted prototype = negative squared distance
    score = -torch.sum((embedding.squeeze(0) - prototype_embedding.detach()) ** 2)
    score.backward()

    gradients = net.get_activations_gradient()          # (1, C, h, w)
    activations = net.get_activations().detach()        # (1, C, h, w)

    weights = gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pool gradients -> (1, C, 1, 1)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)

    cam = cam.squeeze().detach().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam  # 2D numpy array in [0, 1], low resolution


def _apply_colormap(cam_2d):
    """Simple 'jet'-like colormap implemented without extra dependencies."""
    r = np.clip(1.5 - np.abs(4 * cam_2d - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_2d - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_2d - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap_on_image(pil_image, cam_2d, alpha=0.45):
    size = pil_image.size  # (W, H)
    cam_img = Image.fromarray(_apply_colormap(cam_2d)).resize(size, resample=Image.BICUBIC)
    base = pil_image.convert("RGB")
    blended = Image.blend(base, cam_img, alpha=alpha)
    return blended