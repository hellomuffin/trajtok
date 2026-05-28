import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import ResNetModel, ResNetConfig
import timm         


class CustomResNet(nn.Module):
    def __init__(self, model_name="resnet18", upsample_then_downsample=True, 
                 out_channel=512, pretrained=False):
        super(CustomResNet, self).__init__()
        # print(f"{model_name} backbone pretrained?", pretrained)
        # Load the ResNet model with specified configuration
        model_path = f"microsoft/{model_name}"
        config = ResNetConfig.from_pretrained(model_path)
        
        # Load model with specified configuration
        self.resnet = ResNetModel(config) if not pretrained else ResNetModel.from_pretrained(model_path, config=config)
        
        self.upsample_then_downsample = upsample_then_downsample

        # Retrieve layer output sizes from the loaded config
        hidden_sizes = self.resnet.config.hidden_sizes
        self.out_channel = out_channel
        
        # Define linear layers for each stage's output, if upsampling then downsampling
        if self.upsample_then_downsample:
            self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, out_channel) for hidden_size in hidden_sizes
            ])
        
        self.layer_norm = nn.LayerNorm((out_channel))

    def forward(self, x, pool='sum', output_size=(56,56)):
        # Get intermediate layer outputs
        outputs = self.resnet(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # List of tensors from each stage

        # Extract the desired stages
        stage_outputs = [hidden_states[i] for i in [1, 2, 3, 4]]  # Typically corresponds to stages 1-4

        # Apply linear layers if upsample_then_downsample is True
        if self.upsample_then_downsample:
            for i, feature in enumerate(stage_outputs):
                feature = rearrange(feature, 'b d w h -> b w h d')
                feature = self.linear_layers[i](feature)
                feature = self.layer_norm(feature)
                stage_outputs[i] = rearrange(feature, 'b w h d -> b d w h')

        # Resize all features to the specified output size
        resized_features = []
        for feature in stage_outputs:
            rf = F.interpolate(feature, size=output_size, mode='bilinear', align_corners=False)
            resized_features.append(rf)
        # resized_features = [F.layer_norm(feat, feat.size()[1:]) for feat in resized_features]
        # Pooling method
        if pool == 'sum':
            out_features = sum(resized_features)
        elif pool == 'concat':
            out_features = torch.cat(resized_features, dim=1)
        elif pool == 'lastlayer':
            out_features = resized_features[-1]
        else: raise NotImplementedError
        
        return out_features
    
    
    
class CLIPResNetHierFeat(nn.Module):
    """
    Extract hierarchical feature maps from the CLIP‑trained ResNet‑50,
    convert them to a common channel dim, resize them to the same H×W,
    and pool (sum / concat / last) as requested.
    """
    def __init__(
        self,
        model_name      = "hf-hub:timm/resnet50_clip.openai",
        out_channel     = 512,
        upsample_first  = True,         # same meaning as your flag
        pretrained      = True,
    ):
        super().__init__()
        
        # -------- backbone that emits intermediate feature maps ----------
        #   out_indices=(1,2,3,4) == C2, C3, C4, C5  (skip the stem C1)
        self.backbone = timm.create_model(
            model_name,
            pretrained   = pretrained,
            features_only=True,
            out_indices  = (1, 2, 3, 4),
        )
        self.upsample_first = upsample_first
        
        # channel dimensions of those four stages, e.g. [256, 512, 1024, 2048]
        hidden_sizes = self.backbone.feature_info.channels()
        
        if upsample_first:
            # 1×1 convs are cheaper than `nn.Linear` on (C, H, W) tensors
            self.proj = nn.ModuleList([
                nn.Conv2d(c, out_channel, kernel_size=1)
                for c in hidden_sizes
            ])
            self.ln = nn.LayerNorm(out_channel)

    # ------------------------------------------------------------------ #
    def forward(self, x, output_size=(56,56), pool: str = "sum"):
        """
        Args
        ----
        x    : (B, 3, 224, 224) – be sure to use CLIP’s mean/std when you
               build your preprocessing transform.
        pool : one of {"sum", "concat", "lastlayer"}.
        """
        feats = self.backbone(x)          # list of 4 tensors
        # project → layer‑norm → resize
        processed = []
        for i, f in enumerate(feats):     # f: (B, C_i, H_i, W_i)
            if self.upsample_first:
                f = self.proj[i](f)       # (B, out_channel, H_i, W_i)
                f = rearrange(f, "b c h w -> b h w c")
                f = self.ln(f)
                f = rearrange(f, "b h w c -> b c h w")
            f = F.interpolate(
                f, size=output_size,
                mode="bilinear", align_corners=False
            )
            processed.append(f)
        
        if pool == "sum":
            out = sum(processed)
        elif pool == "concat":
            out = torch.cat(processed, dim=1)   # channel dim
        elif pool == "lastlayer":
            out = processed[-1]
        else:
            raise ValueError(f"Unknown pool='{pool}'")

        return out


convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


class DINOResNetHierFeat(nn.Module):
    """DINOv3 ConvNeXt hierarchical feature extractor.

    Locating DINOv3:
      Set `TRAJTOK_DINOV3_ROOT` to the directory containing the cloned
      ``dinov3`` repo (with its ``hubconf.py``). Set
      `TRAJTOK_DINOV3_WEIGHTS_DIR` to the directory holding the official
      ConvNeXt weight files (``dinov3_convnext_{small,base}_pretrain_*.pth``).
      Both default to ``./dinov3`` and ``./dinov3`` respectively. See
      ``scripts/download_dinov3.sh`` for one-line setup.
    """

    def __init__(
        self,
        model_size     = "base",
        out_channel     = 512,
        upsample_first  = True,         # same meaning as your flag
        load_pretrained = True,
    ):
        super().__init__()
        if model_size not in ("base", "small"):
            raise NotImplementedError(f"Unknown DINOv3 model_size={model_size}")

        dinov3_root = os.environ.get("TRAJTOK_DINOV3_ROOT", "./dinov3")
        weights_dir = os.environ.get("TRAJTOK_DINOV3_WEIGHTS_DIR", dinov3_root)
        # Filenames match the official DINOv3 release (hashes are not part of
        # the API; we glob for any matching file rather than hard-coding hashes).
        import glob as _glob
        matches = _glob.glob(os.path.join(weights_dir, f"dinov3_convnext_{model_size}_pretrain_*.pth"))
        if load_pretrained and not matches:
            raise FileNotFoundError(
                f"DINOv3 ConvNeXt-{model_size} weights not found in {weights_dir}. "
                f"Expected a file matching dinov3_convnext_{model_size}_pretrain_*.pth. "
                "See README.md for download instructions."
            )
        ckpt_path = matches[0] if matches else None

        if load_pretrained:
            self.backbone = torch.hub.load(
                dinov3_root, f"dinov3_convnext_{model_size}",
                source="local", weights=ckpt_path,
            )
        else:
            # Random-init path (e.g. for meta-device construction). Caller is
            # responsible for loading weights via load_dinov3_pretrained later.
            self.backbone = torch.hub.load(
                dinov3_root, f"dinov3_convnext_{model_size}",
                source="local", pretrained=False,
            )
        self.upsample_first = upsample_first
        
        hidden_sizes = convnext_sizes[model_size]['dims']
        if upsample_first:
            # 1×1 convs are cheaper than `nn.Linear` on (C, H, W) tensors
            self.proj = nn.ModuleList([
                nn.Conv2d(c, out_channel, kernel_size=1)
                for c in hidden_sizes
            ])
            self.ln = nn.LayerNorm(out_channel)
            
            
    def forward(self, x, output_size=(56,56), pool: str = "sum"):
        """
        Args
        ----
        x    : (B, 3, 224, 224) – be sure to use CLIP’s mean/std when you
            build your preprocessing transform.
        pool : one of {"sum", "concat", "lastlayer"}.
        """
        feats = self.backbone.get_intermediate_layers(x, n=4, reshape=True)
        
        processed = []
        for i, f in enumerate(feats):     # f: (B, C_i, H_i, W_i)
            if self.upsample_first:
                f = self.proj[i](f)       # (B, out_channel, H_i, W_i)
                f = rearrange(f, "b c h w -> b h w c")
                f = self.ln(f)
                f = rearrange(f, "b h w c -> b c h w")
            f = F.interpolate(
                f, size=output_size,
                mode="bilinear", align_corners=False
            )
            processed.append(f)
        
        if pool == "sum":
            out = sum(processed)
        elif pool == "concat":
            out = torch.cat(processed, dim=1)   # channel dim
        elif pool == "lastlayer":
            out = processed[-1]
        else:
            raise ValueError(f"Unknown pool='{pool}'")

        return out
    
    
    
    
    
    
class SigLipFeat(nn.Module):
    """Alternative SigLIP backbone (vs. DINOv3). Optional — only used in
    ablation experiments; the released segmenter checkpoint uses DINOv3.

    Requires the molmo2 source on PYTHONPATH for its `VisionTransformer`
    implementation, plus a downloaded SigLIP weight file pointed at by
    TRAJTOK_SIGLIP_WEIGHTS. Skip this class if you only need DINOv3.
    """

    def __init__(
        self,
        out_channel     = 512,
    ):
        super().__init__()
        try:
            from olmo.nn.image_vit import VitConfig, VisionTransformer
        except ImportError as e:
            raise ImportError(
                "SigLipFeat requires the molmo2 package on PYTHONPATH for "
                "olmo.nn.image_vit.VisionTransformer. Install molmo2 first, "
                "or use the DINOv3 backbone (DINOResNetHierFeat) instead."
            ) from e
        siglip_weights = os.environ.get(
            "TRAJTOK_SIGLIP_WEIGHTS",
            "./checkpoints/siglip-so400m-14-384.pt",
        )
        vit_config = VitConfig(
            image_model_type="siglip",
            image_default_input_size=(378, 378),
            image_patch_size=14,
            image_pos_patch_size=14,
            image_emb_dim=1152,
            image_num_heads=16,
            image_num_key_value_heads=16,
            image_num_layers=27,
            image_head_dim=72,
            image_mlp_dim=4304,
            image_mlp_activations="gelu_pytorch_tanh",
            image_dropout_rate=0.0,
            image_num_pos=729, # no CLS token
            image_norm_eps=1e-6,
            attention_dropout=0.0,
            residual_dropout=0.0,
            initializer_range=0.02,
            resize_mode="siglip",
            normalize="siglip",
            init_path=siglip_weights,
        )
        ori_vit_layers = [-3, -9]
        self.vit_layers = []
        for layer in ori_vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(vit_config.image_num_layers + layer)
        last_layer_needed = (max(self.vit_layers)+1)

        if last_layer_needed < vit_config.image_num_layers: vit_config.image_num_layers=last_layer_needed

        self.vit_config = vit_config
        self.image_vit = vit_config.build('cuda')
        self.image_vit.reset_with_pretrained_weights()
        self.image_vit.eval()
        for param in self.image_vit.parameters():
            param.requires_grad = False
        self.projector = nn.Linear(vit_config.image_emb_dim*len(self.vit_layers), out_channel)
        
    def patchify(self, x: torch.Tensor, patch_size: int, pad: bool = False):
        """
        x: (B, 3, H, W)
        returns: (B, H'*W', 3*patch_size*patch_size) where
                H' = H//patch_size, W' = W//patch_size (if no padding)
        """
        B, C, H, W = x.shape
        assert C == 3, "Expected 3 channels"

        if (H % patch_size != 0) or (W % patch_size != 0):
            if not pad:
                raise ValueError(f"H,W must be divisible by patch_size={patch_size} "
                                f"(got H={H}, W={W}). Set pad=True to pad.")
            # pad to multiple of patch_size (right/bottom)
            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)
            H += pad_h; W += pad_w

        unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        patches = unfold(x)                     # (B, C*ps*ps, L), L = H'*W'
        patches = patches.transpose(1, 2)       # (B, L, C*ps*ps)
        return patches        
                
    def forward(self, x, output_size=(56,56), pool: str = "sum"):
        """
        Args
        ----
        x    : (B, 3, 224, 224) – be sure to use CLIP’s mean/std when you
            build your preprocessing transform.
        pool : one of {"sum", "concat", "lastlayer"}.
        """
        assert output_size[0] == self.vit_config.image_default_input_size[0] // self.vit_config.image_patch_size
        
        images = self.patchify(x, patch_size=self.vit_config.image_patch_size)
        
        with torch.no_grad(): image_features = self.image_vit(images)
        
        features = []
        for layer in self.vit_layers:  # [24,18], so the feature size will double [1, T, N, D_feature * 2 (=2304)]
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        if self.image_vit.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]
            
        projected_feature = self.projector(image_features)
        
        out = rearrange(projected_feature, 'b (h w) d -> b d h w', h=output_size[0], w=output_size[1])
        return out
    

if __name__ == "__main__":
    pass