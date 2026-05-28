import torch
import copy
import numpy as np
import cv2
from trajtok_segmenter.model.checkpoint_utils import interpolate_pos_embed, interpolate_pos_relative_bias_beit, load_temp_embed_with_mismatch
from trajtok_segmenter.text.tokenization_bert import BertTokenizer
from trajtok_segmenter.model.model_pretrain import Singularity, VideoViT, VideoTokCLIP, SegmentWrapper, SegmentCLIP

from trajtok_segmenter.train.scheduler import create_scheduler
from trajtok_segmenter.train.optimizer import create_optimizer

import logging
import os
logger = logging.getLogger(__name__)


def identify_model_cls(config):
    train_segmenter_flag = False
    train_vit_flag = True
    if config.vit_type == 'trajvit': model_cls = VideoTokCLIP
    elif config.vit_type == 'vit3d': model_cls = VideoViT
    elif 'trajvitv2' in config.vit_type: 
        model_cls = SegmentCLIP
        train_segmenter_flag = True
    elif 'segmenter' in config.vit_type: 
        model_cls = SegmentWrapper
        train_segmenter_flag = True
        train_vit_flag = False
    else: 
        raise NotImplementedError
    return model_cls, train_segmenter_flag, train_vit_flag

def load_model_ckpt(model_without_ddp, pretrained_path):
    logger.info(f"Loading checkpoint from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    # load temporal_embeddings, clip or expand when necessary
    state_dict["temporal_embeddings"] = load_temp_embed_with_mismatch(
        temp_embed_old=state_dict["temporal_embeddings"],
        temp_embed_new=model_without_ddp.temporal_embeddings.data
    )

    msg = model_without_ddp.load_state_dict(state_dict, strict=False)
    logger.info(msg)
    logger.info(f"Loaded checkpoint from {pretrained_path}")  


import re
def get_largest_ckpt(folder_path):
    if not os.path.exists(folder_path): return None
    # List all files in the folder
    files = os.listdir(folder_path)
    # Use a regex to match filenames of the format "ckpt_X.pth"
    ckpt_files = [f for f in files if re.match(r'ckpt_\d+\.pth', f)]
    if len(ckpt_files) == 0: return None
    # Extract the numbers from the filenames
    ckpt_numbers = [(int(re.search(r'\d+', f).group()), f) for f in ckpt_files]
    # Find the file with the largest number
    largest_ckpt = max(ckpt_numbers, key=lambda x: x[0])[1]
    return os.path.join(folder_path, largest_ckpt)


def get_sorted_ckpts(folder_path):
    if not os.path.exists(folder_path): return None
    # List all files in the folder
    files = os.listdir(folder_path)
    # Use a regex to match filenames of the format "ckpt_X.pth"
    ckpt_files = [f for f in files if re.match(r'ckpt_\d+\.pth', f)]
    if len(ckpt_files) == 0: return None
    # Extract the numbers from the filenames
    ckpt_number_pairs = sorted([(int(re.search(r'\d+', f).group()), f) for f in ckpt_files], key=lambda x: x[0])
    return [os.path.join(folder_path, k[1]) for k in ckpt_number_pairs]



def setup_model(config, model_cls, has_decoder=False, pretrain=False, find_unused_parameters=False):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    tokenizer = BertTokenizer.from_pretrained(config.text_encoder)
    model = model_cls(config=config, tokenizer=tokenizer)

    num_param = sum(p.numel() for p in model.parameters()) / 10**6
    logger.info(f"total parameters: {num_param:.3f} M")
    
        
    model = model.to(torch.device(config.device))
    model_without_ddp = model
    
    
    
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters  # `False` for image-only task
        )

    if not config.evaluate:
        optimizer = create_optimizer(config.optimizer, model)
        scheduler = create_scheduler(config.scheduler, optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    else:
        optimizer, scheduler, scaler = None, None, None

    start_epoch = 0
    global_step = 0
    
    # TODO: make pretrained_path necessary
    
    
    ckpt_path = config.pretrained_path

    latest_ckpt_path = os.path.join(config.output_dir, "latest.pth")
    resumed_from_latest = False
    if config.resume and not ckpt_path:
        if os.path.exists(latest_ckpt_path):
            ckpt_path = latest_ckpt_path
            resumed_from_latest = True
        else: ckpt_path = get_largest_ckpt(config.output_dir)

    if ckpt_path and os.path.exists(ckpt_path) and not os.path.isdir(ckpt_path):

        logger.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        global_step = checkpoint["global_step"]

        if config.evaluate:
            pass
        elif config.partial_resume:
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
        elif config.resume:
            logger.info("full resume mode")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]


        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning("No pretrained checkpoint provided, training from scratch")



    image_ckpt_path = config.image_pretrained_path
    # Skip warm-start when we resumed from latest.pth — otherwise the image
    # checkpoint would overwrite trained weights every time the job restarts.
    if resumed_from_latest:
        logger.info(
            f"Skipping image_pretrained_path warm-start because we resumed "
            f"from {latest_ckpt_path}"
        )
        image_ckpt_path = None
    if image_ckpt_path and os.path.exists(image_ckpt_path) and not os.path.isdir(image_ckpt_path):
        logger.info(f"Loading checkpoint from {image_ckpt_path}")
        checkpoint = torch.load(image_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        msg = model_without_ddp.load_image_model(state_dict, load_only_vision=config.load_vision_only)
        logger.info(msg)
        logger.info(f"Loaded image pretrained checkpoint from {image_ckpt_path}")
        
    return model, model_without_ddp, optimizer, scheduler, scaler, tokenizer, start_epoch, global_step






# for segmentation evaluation

from torchvision import transforms
from torchvision.transforms import InterpolationMode
def simple_create_transform(image_size):
    """ create image transform """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean, std)
    type_transform = transforms.Lambda(lambda x: x.float().div(255.))

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC),
        type_transform,
        normalize,
    ])
    
    
    
def simple_load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='model',
):
    if pretrained is None: 
        logger.info("pretrained model path is None, return")
        return encoder
    
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
    
    pretrained_dict = checkpoint[checkpoint_key]

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    m_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('vision_encoder.vision_encoder.'): k = k[len('vision_encoder.'):]
        elif k.startswith('text_encoder'): continue
        else: k = k.replace('vision_encoder.', '')
        m_pretrained_dict[k] = v
    pretrained_dict = m_pretrained_dict    
    

    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v

    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained encoder with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder





def simple_preprocess_mask_and_graph(masks, graphs, image_size=224, mask_res_down_factor=4):
    """ resize mask before doing inference.  mask: tensor of shape (T,H,W)"""
    def resize_masks(masks, size):
        T = masks.shape[0]
        resized_masks = np.empty((T, size[0], size[1]), dtype=masks.dtype)
        for t in range(T): resized_masks[t] = cv2.resize(masks[t], (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return resized_masks
    # TODO: dilate the mask
    
    if len(masks.shape) == 2: masks = masks[None, ...]
 
    masks = resize_masks(masks, size=(image_size//mask_res_down_factor, image_size//mask_res_down_factor))
    masks = torch.from_numpy(masks)
    
    if graphs is None: graphs = torch.arange(1,masks.max()+1).unsqueeze(1)  # (N,T) where T=1
    else: graphs = torch.from_numpy(graphs)
    
    masks[masks>graphs.max()] = 0
    graphs[graphs>masks.max()] = 0  # some seg idx is invalid due to vanished segmentation size
    graphs = graphs[~(graphs == 0).all(dim=1)] # filter out rows that are all 0
    return masks, graphs