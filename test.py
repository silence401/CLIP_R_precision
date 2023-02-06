import clip
import torch
clip_model, preprocess = clip.load('ViT-B/32',  jit=False)
print(clip_model)
print(preprocess)
