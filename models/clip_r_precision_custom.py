import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl

from clip import clip


# class CLIPRPrecision(pl.LightningModule):
class CLIPRPrecision(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model, preprocess = clip.load('ViT-B/32', jit=False)
        #clip_model, preprocess = clip.load('ViT-B/32', jit=False)
        self.clip_model = clip_model.float()
        self.preprocess = preprocess

        # visual

    def encode_image(self, image):
        
        with torch.no_grad():
            x = self.preprocess(image)
            x = x.unsqueeze(0).cuda()
            x = self.clip_model.encode_image(x)
        return x

    def encode_text(self, text_token):
        with torch.no_grad():
            x = self.clip_model.encode_text(text_token)
        return x

    def unfreeze(self):
        self.attn_pool.requires_grad_(True)
        self.layer4.requires_grad_(True)

        self.transformer_last_block.requires_grad_(True)
        self.ln_final.requires_grad_(True)
        self.text_projection.requires_grad_(True)
        self.logit_scale.requires_grad_(True)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        return image_features, text_features

    def training_step(self, batch, batch_idx):
        image, text = batch

        bs = image.size(0)

        image_features, text_features = self(image, text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        label = torch.arange(bs).long()
        label = label.to(image.device)

        loss_i = F.cross_entropy(logits_per_image, label)
        loss_t = F.cross_entropy(logits_per_text, label)

        loss = (loss_i + loss_t) / 2

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW(list(self.attn_pool.parameters()) +
                                list(self.layer4.parameters()) +
                                list(self.transformer_last_block.parameters()) +
                                list(self.ln_final.parameters()) +
                                [self.text_projection, self.logit_scale],
                                lr=lr)
        return opt

