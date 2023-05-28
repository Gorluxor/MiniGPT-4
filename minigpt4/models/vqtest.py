import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from diffusers.models import AutoencoderKL


from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.encode_mini_gpt4 import EncodeMiniGPT4

import torch.distributions as dist

from minigpt4.models.blip2_outputs import VAEOutput

from diffusers.models.autoencoder_kl import AutoencoderKLOutput

@registry.register_model("vqtest")
class VQTest(Blip2Base):
    """
    Alignment BLIP-2 VAE model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt5.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        vae_type="",  # SDVAE or VQGAN-VAE
        loss_type="",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.encoder = EncodeMiniGPT4(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )
        self.device_8bit = device_8bit
        self.tokenizer = self.encoder.tokenizer
        self.low_resource = self.encoder.low_resource

        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        print('Loaded Encoder part')

       
        
        self.vae = self.init_vae(vae_type, device_8bit)
        print('Loaded VAE')
        # disable training for the encoder
        for name, param in self.vae.named_parameters():
            param.requires_grad = False
        self.model = SimpleDecoder()
        print('Loaded SimpleDecoder')

    def loss_calc(self, x, ref, loss_type="mse"):
        _valid_loss_types = ["kl", "mse", "mse_tensor", "sample_mse"]
        assert loss_type in _valid_loss_types, f"Invalid loss type {loss_type}"

        # check if loss contains _tensor
        if "_tensor" in loss_type:
            assert isinstance(x, torch.Tensor), "x must be a tensor"
            assert isinstance(ref, torch.Tensor), "ref must be a tensor"
        else: # only check DiagonalGaussianDistribution
            assert isinstance(x, DiagonalGaussianDistribution), "x must be a DiagonalGaussianDistribution"
            assert isinstance(ref, DiagonalGaussianDistribution), "ref must be a DiagonalGaussianDistribution"

        if loss_type == "kl":
            loss = dist.kl_divergence(dist.Normal(x.mean, x.std), dist.Normal(ref.mean, ref.std))
            loss = loss.mean()
        elif loss_type == "mse_tensor":
            loss = torch.nn.functional.mse_loss(x, ref)
        elif loss_type == "mse":
            assert x.mean.shape == ref.mean.shape, f"Shape mismatch {x.mean.shape} != {ref.mean.shape}"
            loss = torch.nn.functional.mse_loss(x.mean, ref.mean) + torch.nn.functional.mse_loss(x.std, ref.std)
        elif loss_type == "sample_mse":
            sample_x = x.sample()
            sample_ref = ref.sample()
            loss = torch.nn.functional.mse_loss(sample_x, sample_ref)
        else:
            raise ValueError(f"Invalid loss type {loss_type}")
        return loss


    
    def custom_encode(self, model, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        if model.use_tiling and (x.shape[-1] > model.tile_sample_min_size or x.shape[-2] > model.tile_sample_min_size):
            return model.tiled_encode(x, return_dict=return_dict)

        h = model.encoder(x)
        moments = model.quant_conv(h)
        return moments
        # posterior = DiagonalGaussianDistribution(moments)

        # if not return_dict:
        #     return (posterior,)

        # return AutoencoderKLOutput(latent_dist=posterior)

    def forward(self, samples):
        # x input = [batch, 3, 224, 224]
        if isinstance(samples, dict):
            x = samples["image"]
        elif isinstance(samples, torch.Tensor):
            x = samples
        else:
            raise ValueError(f"Invalid input type {type(samples)}")
        device = x.device
        img = x.clone().detach().to(device)
        x = self.encoder.encode_img(img) # [batch, 32, 768]
        if isinstance(x, BaseModelOutputWithPoolingAndCrossAttentions):
            x = x.last_hidden_state # [batch, 32, 768]
        x = self.model(x) # [batch, 8, 28, 28]
        ref = self.custom_encode(self.vae,img) # [batch, 8, 28, 28]
        if isinstance(ref, AutoencoderKLOutput):
            ref = ref.latent_dist
        # sample_x = x.sample()
        # sample_ref = ref.sample()
        return VAEOutput(
            loss=self.loss_calc(x, ref, loss_type=self.loss_type),
            vae_embedding=ref,
            predicted_embedding=x
        )

        # return VAEOutput(
        #     loss=torch.nn.functional.mse_loss(sample_x, sample_ref),
        #     vae_embedding=ref,
        #     predicted_embedding=x
        # )

    @classmethod    
    def init_vae(cls, vae_type, device, ):
        if vae_type == "SDVAE":
            model_vae = "mse"
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{model_vae}", cache_dir='./models/huggingface').to(device)
        elif vae_type == "VQGAN-VAE":
            pass
        else:
            raise ValueError("Invalid VAE type")
        return vae
    
    @classmethod
    def from_config(cls, cfg):
        
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        vae_type = cfg.get("vae_type", "SDVAE")
        loss_type = cfg.get("loss_type", "mse")
        print(f"VAE type: {vae_type} and loss type: {loss_type}")
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            vae_type=vae_type,
            loss_type=loss_type
        )
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load MODEL-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model


dimensionality = {
    "SDVAE": {'channels': 4, 'spatial': 48 * 48},
    "VQGAN-VAE": {'channels': 256, 'spatial': 32 * 32},
    "BLIP-2": {'channels': 768, 'spatial': 32 },
    "Llamma": {'channels': 5120, 'spatial': 32 }
    # Be careful, that BLIP-2 actually is structured as B x 32 x 768
    # Where the 768 is token embedding size, and 32 is the number of tokens
}

# class VQTransform(nn.Module):
#     def __init__(self,
#                  input_image_dims=(3, 384, 384),
#                  vae_type: str = "SDVAE",
#                  network_type: str = "linear",
#                  use_llama: bool = False,
#                  ) -> None:
#         super(VQTransform, self).__init__()
#         self.network_type = network_type
#         self.latent_dim = dimensionality[vae_type]['channels'] 

#         input_type = "BLIP-2" if not use_llama else "Llamma"

#         if network_type == "linear":
#             self.model = nn.Sequential(
#                 # start from 768, finish at 256, while going from 32 to 48 x 48
#                 nn.Linear(dimensionality[input_type]['channels'], 256),
#                 nn.GELU(),
#                 nn.Linear(256, self.latent_dim),
#                 nn.GELU()
#             )
#         elif network_type == "conv":
#             self.model = nn.Sequential(
#                 nn.Conv2d(dimensionality[input_type]['channels'], 32, kernel_size=3, stride=1, padding=1),
#                 nn.GELU(),
#                 nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#                 nn.GELU(),
#                 nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#                 nn.GELU(),
#                 nn.Flatten(),
#                 nn.Linear(128 * 32 * 32, self.latent_dim),
#                 nn.GELU()
#             )
#         elif network_type == "mlp":
#             self.model = nn.Sequential(
#                 nn.Linear(dimensionality[input_type]['channels'] * dimensionality[input_type]['spatial'], 2048),
#                 nn.ReLU(),
#                 nn.Linear(2048, 1024),
#                 nn.ReLU(),
#                 nn.Linear(1024, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, self.latent_dim),
#                 nn.ReLU()
#             )
#         elif network_type == "transformer":
#             self.model = nn.Sequential(
#                 nn.Linear(dimensionality[input_type]['channels'] * dimensionality[input_type]['spatial'], 1024),
#                 nn.TransformerEncoderLayer(d_model=1024, nhead=8),
#                 nn.Linear(1024, self.latent_dim),
#             )

#     def forward(self, blip2_latent) -> torch.Tensor:
#         # input BLIP-2 latent is (batch_size, 32, 768)
#         # output SDVAE or VQGAN-VAE latent
        
#         #assert blip2_latent.shape[1] == 32 and blip2_latent.shape[2] == 768

#         return self.model(blip2_latent)
        

import torch
from torch import nn
from diffusers.models.vae import DiagonalGaussianDistribution

from minigpt4.models.blip2_outputs import BaseModelOutputWithPoolingAndCrossAttentions
class SimpleNet(nn.Module):
    def __init__(self, return_guassian:bool=True):
        super().__init__()
        self.return_guassian = return_guassian
        self.conv1 = nn.Conv2d(768, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 4 * 4, 8 * 48 * 48)
        self.activation = nn.GELU()
        print(f'{self.__class__.__name__} initialized with {return_guassian=}')


    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 32 and x.shape[2] == 768:
            batch_size, seq_len, hidden_size = x.shape
            x = x.permute(0, 2, 1).reshape(batch_size, hidden_size, 8, -1) # B x C x H x W => B x 768 x 8 x 4
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.reshape(-1, 8 * 4 * 4)
        x = self.fc(x).view(-1, 8, 48, 48)
        if self.return_guassian == True:
            return DiagonalGaussianDistribution(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, input_size:int=768, # channel
                output_size:int = 8, #channel
                output_size_spacial:int = 28, #spacial
                return_gaussian=False):
        super().__init__()
        self.return_gaussian = return_gaussian
        self.input_size = input_size
        self.output_size = output_size
        self.output_size_spacial = output_size_spacial

        self.conv1 = nn.Conv2d(input_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 4, kernel_size=3, padding=1)
        
        self.activation = nn.GELU()
        # Calculate the shape of the tensor after the convolutional layers
        self.conv_out_size = self._get_conv_out_size()
        self.fc = nn.Linear(self.conv_out_size, int(output_size * output_size_spacial * output_size_spacial))
        

    def forward(self, x):
        if isinstance(x, BaseModelOutputWithPoolingAndCrossAttentions):
            x = x.last_hidden_state
        if len(x.shape) == 3 and x.shape[2] == self.input_size:
            batch_size, seq_len, hidden_size = x.shape
            x = x.permute(0, 2, 1).reshape(batch_size, hidden_size, 8, -1) # B x C x H x W => B x input_size x 8 x 4
        
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        x = x.reshape(-1, self.conv_out_size)
        x = self.fc(x).view(-1, self.output_size, self.output_size_spacial, self.output_size_spacial)
        
        if self.return_gaussian:
            return DiagonalGaussianDistribution(x)
        return x

    def _get_conv_out_size(self):
        """
        Calculate the output size of the convolutional layers
        """
        x = torch.zeros(1, self.input_size, 8, 4)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return int(torch.prod(torch.tensor(x.size())))
    


import torch.nn as nn

# class LatentSpaceAligner(nn.Module):
#     def __init__(self, input_channels=768, hidden_channels = 256, output_channels=4, return_gaussian:bool = True):
#         super().__init__()
#         self.return_gaussian = return_gaussian
#         self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding='same')
#         self.act1 = nn.SELU()
#         self.conv2 = nn.Conv2d(hidden_channels, int(hidden_channels/2), kernel_size=3, stride=1, padding=0)
#         self.act2 = nn.SELU()
#         self.conv3 = nn.Conv2d(int(hidden_channels/2), output_channels*2, kernel_size=3, stride=1, padding=0)
#         self.act3 = nn.SELU()
        
#     def forward(self, x):
        
#         # [B, 768, 32] or [B, 32, 768]
#         # check if the input is [B, 32, 768]
#         if x.shape[1] == 32:
#             x = x.transpose(1, 2) # [B, 768, 32]
#         # repeat the 32 into 32 x 32
#         x = x.unsqueeze(2).repeat(1, 1, 32, 1) # [B, 768, 32 ,32]
#         # use conv1
#         x = self.act1(self.conv(x)) # [B, 256, 32, 32]
#         # use conv2
#         x = self.act2(self.conv2(x)) # [B, 128, 30, 30]
#         # use conv3
#         x = self.act3(self.conv3(x)) # [B, 4, 28, 28]
#         if self.return_gaussian:
#             return DiagonalGaussianDistribution(x)
#         return x


class LatentSpaceAligner(nn.Module):
    def __init__(self, input_channels=768, hidden_channels = 256, output_channels=4, return_gaussian:bool = True):
        super().__init__()
        self.return_gaussian = return_gaussian
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding='same')
        self.act1 = nn.SELU()
        self.conv2 = nn.Conv2d(hidden_channels, int(hidden_channels/2), kernel_size=3, stride=1, padding=0)
        self.act2 = nn.SELU()
        self.conv3 = nn.Conv2d(int(hidden_channels/2), output_channels*2, kernel_size=3, stride=1, padding=0)
        self.act3 = nn.SELU()
        
    def forward(self, x):
        
        # [B, 768, 32] or [B, 32, 768]
        # check if the input is [B, 32, 768]
        if x.shape[1] == 32:
            x = x.transpose(1, 2) # [B, 768, 32]
        # repeat the 32 into 32 x 32
        x = x.unsqueeze(2).repeat(1, 1, 32, 1) # [B, 768, 32 ,32]
        # use conv1
        x = self.act1(self.conv(x)) # [B, 256, 32, 32]
        # use conv2
        x = self.act2(self.conv2(x)) # [B, 128, 30, 30]
        # use conv3
        x = self.act3(self.conv3(x)) # [B, 8, 28, 28]
        if self.return_gaussian:
            t = DiagonalGaussianDistribution(x)
            t.mean = torch.clamp(x[:, :4, ...], -1, 1)
            t.std = torch.clamp(x[:, :-4, ...], 0, 1)
            x = t
        return x


from minigpt4.models.tf import PreNorm, Attention, FeedForward

from einops import rearrange
class AlignLatent(nn.Module):
    def __init__(self, queries_dim:int, dim:int) -> None:
        super().__init__()
        self.query_to_key = nn.Linear(dim, queries_dim)
        self.cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads=1, dim_head=dim), context_dim=dim)
        self.cross_ff = PreNorm(queries_dim, FeedForward(queries_dim, dim))

    def forward(self, input, query):
        # input: [B, L, C] -> K,V
        # query: [B, C1, H, W] -> Q

        query = rearrange(query, 'b c1 h w -> b (c1 h w)')
        query = self.query_to_key(query)

        x = self.cross_attn(input, context=query) + input
        x = self.cross_ff(x) + x

        return x

# class AlignLatent(nn.Module):
#     def __init__(self, queries_dim:int, dim:int) -> None:
#         super().__init__()
#         # 
#         self.query_to_key = nn.Embedding( h*w, dim )
#         self.cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
#         self.cross_ff = PreNorm(queries_dim, FeedForward(queries_dim, dim))

#     def forward(self, input):
#         # input: [B, 768, 32] -> K,V
#         # query: [B, (H * W), C] -> Q

#         x = self.cross_attn(query, context = input) + query
#         x = self.cross_ff(x) + x


# class LatentSpaceAligner(nn.Module):
#     def __init__(self, input_channels=768, output_channels=4, return_gaussian:bool = True):
#         super().__init__()
#         self.return_gaussian = return_gaussian
#         self.linear = nn.Linear(input_channels * 32, 2 * output_channels * 28 * 28)
#         self.act = nn.SELU()
#         # self.conv3 = nn.Conv2d(int(hidden_channels/2), output_channels*2, kernel_size=3, stride=1, padding=0)
#         # self.act3 = nn.SELU()
        
#     def forward(self, x):
#         # [B, 768, 32] or [B, 32, 768]
        
#         #x = x.view(x.shape[0], -1) # [B, 768*32]
#         # permute the input to [B, 768, 32]
#         if x.shape[1] == 32:
#             x = x.transpose(1, 2)
#         x = self.linear(x) # [B, 2*4*28*28]
#         x = self.act(x) # [B, 2*4*28*28]
#         x = x.view(x.shape[0], 2, 4, 28, 28) # [B, 2 * 4, 28, 28]
#         # instantiate DiagonalGaussianDistribution, which is overriden by the 
#         # predicted mean and std
#         if self.return_gaussian:
#             t = DiagonalGaussianDistribution(x)
#             t.mean = torch.clamp(x[:, 0, ...], -1, 1)
#             t.std = torch.clamp(x[:, 1, ...], 0, 1)
#             x = t
#         return x


