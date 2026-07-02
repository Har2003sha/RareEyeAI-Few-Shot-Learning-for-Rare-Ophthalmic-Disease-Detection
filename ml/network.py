"""
Embedding backbone for the few-shot pipeline.

Uses a torchvision ResNet-18 pretrained on ImageNet as the feature
extractor. Even WITHOUT any fine-tuning on ophthalmic images, ImageNet
pretraining gives the network genuinely useful low/mid-level visual
features (edges, colour blobs, texture, shape) - so embeddings of visually
different images land in meaningfully different places in feature space.

Why this matters: a randomly-initialized CNN (the previous version of this
file) produces near-random embeddings, so every query image ends up
roughly equidistant from all class prototypes and the softmax output
collapses to ~1/N for every class (e.g. ~20% for 5 classes) regardless of
the input image. Swapping in ImageNet-pretrained weights fixes this by
giving the encoder real visual features to work with, so predictions and
confidence scores actually vary meaningfully between different images.

IMPORTANT (read this before using in a clinical setting):
This backbone has NOT been fine-tuned on a real, labelled
rare-ophthalmic-disease dataset (no such public dataset ships with this
repo), and the bundled support set is procedurally generated, not real
patient data. Predictions demonstrate the *architecture and engineering
pipeline* of a few-shot rare disease detector, not real diagnosis. See
README "Training on a real dataset" section for how to plug in real data
and (optionally) fine-tuned weights.
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

torch.manual_seed(42)


class EmbeddingNet(nn.Module):
    """ResNet-18 (ImageNet-pretrained) truncated before the classification
    head, exposing the last convolutional feature map (layer4 output) for
    Grad-CAM and the globally-pooled 512-d vector as the embedding used by
    the Prototypical Network.
    """

    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()

        backbone = None
        if pretrained:
            try:
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:  # no internet / weights not cached yet
                warnings.warn(
                    "Could not download ImageNet-pretrained ResNet-18 weights "
                    f"({e}). Falling back to a randomly-initialized backbone. "
                    "Predictions will be much less differentiated until you "
                    "run this again with internet access so torch can cache "
                    "the pretrained weights (~44MB, one-time download)."
                )
        if backbone is None:
            backbone = resnet18(weights=None)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.embedding_dim = embedding_dim  # 512 for resnet18
        self._last_conv_activations = None
        self._last_conv_gradients = None

    def activations_hook(self, grad):
        self._last_conv_gradients = grad

    def forward(self, x, register_hook=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        conv_out = self.layer4(x)  # last conv feature map - used by Grad-CAM

        if register_hook and conv_out.requires_grad:
            conv_out.register_hook(self.activations_hook)
        self._last_conv_activations = conv_out

        x = self.avgpool(conv_out)
        x = x.flatten(1)
        # L2-normalize embeddings onto the unit hypersphere. This keeps
        # squared-Euclidean distances bounded in [0, 4] regardless of the
        # backbone's raw feature scale, which prevents the softmax over
        # prototype distances from saturating to near-0%/near-100% and
        # gives well-calibrated, meaningfully different confidence scores
        # across images.
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_activations_gradient(self):
        return self._last_conv_gradients

    def get_activations(self):
        return self._last_conv_activations


_model_cache = {}


def get_embedding_net(embedding_dim=512):
    if "net" not in _model_cache:
        net = EmbeddingNet(embedding_dim=embedding_dim, pretrained=True)
        net.eval()
        _model_cache["net"] = net
    return _model_cache["net"]