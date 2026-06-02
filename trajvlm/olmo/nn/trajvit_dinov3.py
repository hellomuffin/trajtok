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


_DINOV3_CKPT_PATHS = {
    "base":  "/weka/prior-default/chenhaoz/home/open_videotok/dinov3/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "small": "/weka/prior-default/chenhaoz/home/open_videotok/dinov3/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
}


class DINOResNetHierFeat(nn.Module):
    def __init__(
        self,
        model_size     = "base",
        out_channel     = 512,
        upsample_first  = True,         # same meaning as your flag
        load_pretrained = False,
    ):
        """When `load_pretrained=False` (default), the ConvNeXt backbone is constructed
        with random weights and `.load_dinov3_pretrained()` should be called later
        (after meta-device materialisation). When True (legacy), pretrained DINOv3
        weights are loaded eagerly in __init__ and the whole module is moved to CUDA.
        The latter conflicts with Molmo2's meta-device trainer pipeline."""
        super().__init__()
        if model_size not in _DINOV3_CKPT_PATHS:
            raise NotImplementedError(model_size)
        self.model_size = model_size

        # Build architecture only — defer weight loading to support meta-device init.
        if load_pretrained:
            self.backbone = torch.hub.load(
                '/weka/prior-default/chenhaoz/home/open_videotok/dinov3',
                f'dinov3_convnext_{model_size}',
                source='local',
                weights=_DINOV3_CKPT_PATHS[model_size],
            ).to('cuda')
        else:
            self.backbone = torch.hub.load(
                '/weka/prior-default/chenhaoz/home/open_videotok/dinov3',
                f'dinov3_convnext_{model_size}',
                source='local',
                pretrained=False,
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

    def load_dinov3_pretrained(self):
        """Load DINOv3 pretrained weights into self.backbone (call after materialisation)."""
        sd = torch.load(_DINOV3_CKPT_PATHS[self.model_size], map_location="cpu", weights_only=True)
        # torch.hub-built convnext expects a flat state-dict
        msg = self.backbone.load_state_dict(sd, strict=False)
        return msg
            
            
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
    def __init__(
        self,
        out_channel     = 512,
    ):
        super().__init__()
        import sys
        sys.path.append('/weka/chenhaoz/home/mm_olmo')
        from olmo.nn.image_vit import VitConfig, VisionTransformer
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
            init_path = '/oetraining/mm-olmo/pretrained_image_encoders/siglip-so400m-14-384.pt',
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