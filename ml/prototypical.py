# import os
# import json
# import numpy as np
# import torch
# from PIL import Image
# from torchvision import transforms

# from ml.network import get_embedding_net
# from ml.demo_support_set import RARE_DISEASE_CLASSES, ensure_support_set

# IMG_SIZE = 128

# _transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# _prototype_cache = {}


# def load_image_tensor(path_or_pil):
#     if isinstance(path_or_pil, str):
#         img = Image.open(path_or_pil).convert("RGB")
#     else:
#         img = path_or_pil.convert("RGB")
#     return _transform(img).unsqueeze(0), img


# def _embed_batch(net, tensors):
#     with torch.no_grad():
#         return net(tensors)


# def build_prototypes(support_root, embedding_dim=64, k_shot=5, force=False):
#     """Compute one prototype embedding (mean support embedding) per class."""
#     cache_key = support_root
#     if not force and cache_key in _prototype_cache:
#         return _prototype_cache[cache_key]

#     ensure_support_set(support_root, k_shot=k_shot)
#     net = get_embedding_net(embedding_dim)

#     prototypes = {}
#     per_class_embeddings = {}
#     for class_name in RARE_DISEASE_CLASSES:
#         class_dir = os.path.join(support_root, class_name.replace(" ", "_"))
#         files = sorted(
#             [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg"))]
#         )[:k_shot]
#         tensors = []
#         for f in files:
#             t, _ = load_image_tensor(os.path.join(class_dir, f))
#             tensors.append(t)
#         batch = torch.cat(tensors, dim=0)
#         embeddings = _embed_batch(net, batch)  # (k_shot, embedding_dim)
#         prototypes[class_name] = embeddings.mean(dim=0)
#         per_class_embeddings[class_name] = embeddings

#     _prototype_cache[cache_key] = (prototypes, per_class_embeddings)
#     return prototypes, per_class_embeddings


# def classify_query(image_path, support_root, embedding_dim=64, k_shot=5):
#     """Run the query image through the embedding net and classify it by
#     (negative squared euclidean) distance to each class prototype, exactly
#     as in the Prototypical Networks paper.
#     """
#     prototypes, _ = build_prototypes(support_root, embedding_dim, k_shot)
#     net = get_embedding_net(embedding_dim)

#     query_tensor, pil_img = load_image_tensor(image_path)
#     with torch.no_grad():
#         query_embedding = net(query_tensor).squeeze(0)  # (embedding_dim,)

#     distances = {}
#     for class_name, proto in prototypes.items():
#         dist = torch.sum((query_embedding - proto) ** 2).item()
#         distances[class_name] = dist

#     # Prototypical Networks classify via softmax over NEGATIVE distances
#     dist_values = np.array(list(distances.values()))
#     neg_dist = -dist_values
#     exp = np.exp(neg_dist - neg_dist.max())
#     probs = exp / exp.sum()

#     class_names = list(distances.keys())
#     prob_dict = {c: float(p) for c, p in zip(class_names, probs)}
#     predicted_class = class_names[int(np.argmax(probs))]
#     confidence = float(np.max(probs))

#     return {
#         "predicted_class": predicted_class,
#         "confidence": confidence,
#         "distances": distances,
#         "probabilities": prob_dict,
#         "query_tensor": query_tensor,
#         "pil_image": pil_img,
#     }







import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.network import get_embedding_net
from ml.demo_support_set import RARE_DISEASE_CLASSES, ensure_support_set

IMG_SIZE = 224  # standard ResNet-18 / ImageNet input resolution

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_prototype_cache = {}


def load_image_tensor(path_or_pil):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert("RGB")
    else:
        img = path_or_pil.convert("RGB")
    return _transform(img).unsqueeze(0), img


def _embed_batch(net, tensors):
    with torch.no_grad():
        return net(tensors)


def build_prototypes(support_root, embedding_dim=64, k_shot=5, force=False):
    """Compute one prototype embedding (mean support embedding) per class."""
    cache_key = support_root
    if not force and cache_key in _prototype_cache:
        return _prototype_cache[cache_key]

    ensure_support_set(support_root, k_shot=k_shot)
    net = get_embedding_net(embedding_dim)

    prototypes = {}
    per_class_embeddings = {}
    for class_name in RARE_DISEASE_CLASSES:
        class_dir = os.path.join(support_root, class_name.replace(" ", "_"))
        files = sorted(
            [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg"))]
        )[:k_shot]
        tensors = []
        for f in files:
            t, _ = load_image_tensor(os.path.join(class_dir, f))
            tensors.append(t)
        batch = torch.cat(tensors, dim=0)
        embeddings = _embed_batch(net, batch)  # (k_shot, embedding_dim)
        prototypes[class_name] = embeddings.mean(dim=0)
        per_class_embeddings[class_name] = embeddings

    _prototype_cache[cache_key] = (prototypes, per_class_embeddings)
    return prototypes, per_class_embeddings


def classify_query(image_path, support_root, embedding_dim=64, k_shot=5):
    """Run the query image through the embedding net and classify it by
    (negative squared euclidean) distance to each class prototype, exactly
    as in the Prototypical Networks paper.
    """
    prototypes, _ = build_prototypes(support_root, embedding_dim, k_shot)
    net = get_embedding_net(embedding_dim)

    query_tensor, pil_img = load_image_tensor(image_path)
    with torch.no_grad():
        query_embedding = net(query_tensor).squeeze(0)  # (embedding_dim,)

    distances = {}
    for class_name, proto in prototypes.items():
        dist = torch.sum((query_embedding - proto) ** 2).item()
        distances[class_name] = dist

    # Prototypical Networks classify via softmax over NEGATIVE distances
    dist_values = np.array(list(distances.values()))
    neg_dist = -dist_values
    exp = np.exp(neg_dist - neg_dist.max())
    probs = exp / exp.sum()

    class_names = list(distances.keys())
    prob_dict = {c: float(p) for c, p in zip(class_names, probs)}
    predicted_class = class_names[int(np.argmax(probs))]
    confidence = float(np.max(probs))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "distances": distances,
        "probabilities": prob_dict,
        "query_tensor": query_tensor,
        "pil_image": pil_img,
    }