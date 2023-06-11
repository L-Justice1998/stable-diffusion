"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import rearrange
from PIL import Image
import math
from torch.nn import functional as F
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

#Adapted from DDIMSampler
class DDIMSampler_video(object):
    def __init__(self, model,RIFE_model,schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.RIFE_model = RIFE_model

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               frames,
               shape,
               alpha,
               sigma_1=0.01, 
               sigma_2=0.001, 
               sigma_3=0.001,
               first_guidance_scale=1, 
               second_guidance_scale=1,
               total_guidance_scale=1,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != frames:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {frames}")
            else:
                if conditioning.shape[0] != frames:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {frames}")
                    print(conditioning.shape)#(frames,77,768)

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        # 原来的shape只有三维 现在还是三维但是bs变成frame
        C, H, W = shape
        size = (frames, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    alpha = alpha,sigma_1=sigma_1, sigma_2=sigma_2,sigma_3=sigma_3,
                                                    first_guidance_scale=first_guidance_scale, 
                                                    second_guidance_scale=second_guidance_scale,
                                                    total_guidance_scale=total_guidance_scale,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,alpha,sigma_1, 
                      sigma_2, sigma_3, first_guidance_scale, 
                      second_guidance_scale,total_guidance_scale,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        # 原来的xT是bs*隐空间的三维
        # 现在改为frames*三维
        frames, C, H, W = shape
        if x_T is None:
            x_T = []
            # img = torch.randn(shape, device=device)
            # 用progressive的方法生成初始噪声
            first_frame = torch.randn((1,C,H,W),device = device)
            x_T.append(first_frame)
            for i in range(frames-1):
                added_noise = torch.randn([1,C,H,W],device = device) * (math.sqrt(1/(1+alpha*alpha)))
                current_frame = math.sqrt(alpha/(1+alpha*alpha)) * x_T[i] + added_noise
                x_T.append(current_frame)
            img = torch.cat(x_T,dim = 0)
        else:
            img = x_T
        # 此时img应该(frames,c,h,w)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        #因为要用到前两帧的信息 利用intermediates的信息
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            #这个循环是时间尺度上的
            index = total_steps - i - 1
            ts = torch.full((frames,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            e_t, pred_z0, guidance = self.cal_guidance(img, cond, ts, sigma_1,sigma_2,sigma_3, first_guidance_scale, second_guidance_scale,index,repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None)
            outs = self.p_sample_ddim(img, e_t, pred_z0, guidance,index=index, 
                                      total_guidance_scale = total_guidance_scale,
                                      use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout)
            #img的shape应该是(frames,c,w,h) 对同一个时刻的不同帧同时生成 原来是(bs,c,w,h)
            img, pred_z0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_z0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_z0)

        return img, intermediates

    @torch.no_grad()
    def cal_guidance(self, z, c, t,sigma1,sigma2,sigma3, first_guidance_scale, second_guidance_scale,index,repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        z是上一时刻的所有隐空间有噪声的帧 是(frames,c,h,w)
        返回值是e_t, pred_z0, guidance
        """
        frames, *_, device = *z.shape, z.device
        #unet的in_channel是4通道 第一个通道本来是bs的
        #用z_variable变量追踪梯度 

        z_variable = torch.tensor(z,requires_grad = True)
        # 无条件采样
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(z_variable, t, c)
        # 有条件采样 会给一个无条件的条件
        else:
            z_in = torch.cat([z_variable] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(z_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, z_variable, t, c, **corrector_kwargs)
        # e_t.requires_grad_(True) 

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # select parameters corresponding to the currently considered timestep
        # index是指示几步的 从(b, 1, 1, 1)变成(frames, 1, 1, 1)
        a_t = torch.full((frames, 1, 1, 1), alphas[index], device=device)
        sqrt_one_minus_at = torch.full((frames, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
        #三个高斯方差
        sigma1 = torch.full((2, 1, 1, 1), sigma1,device=device)
        sigma2 = torch.full((frames-2, 1, 1, 1), sigma2,device=device)
        sigma_loss1 = torch.cat((sigma1,sigma2),dim = 0)
        sigma3 = torch.full((frames, 1, 1, 1), sigma3,device=device)

        #所有的predict x0
        pred_z0 = (z_variable - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # pred_z0.requires_grad_(True) 
        #decode 到x0空间 还需检查中间转化是不是正确 follow txt2img.py line 313-322
        pred_x0 = self.model.decode_first_stage(pred_z0)
        # pred_x0.requires_grad_(True) 
        #将pred_x0转到[0,1]
        pred_x0 = torch.clamp((pred_x0 + 1.0) / 2.0, min=0.0, max=1.0)
        pred_x0 = pred_x0.permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        #预测中间帧 [0,1]
        # print(pred_x0.shape)torch.Size([8, 3, 512, 512])
        
        _, c, h, w = pred_x0.shape
        #如果是前两帧就取自己，相当于没有插值
        img0 = torch.cat([pred_x0[0:2],pred_x0[0:-2]],dim = 0)
        img1 = pred_x0
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        predict_mid = self.RIFE_model.inference(img0, img1)
        # predict_mid.requires_grad_(True) 
        # print(mid.shape)torch.Size([8, 3, 512, 512])

        #两项loss
        tmp_pred_x0 = torch.cat([pred_x0[0:1],pred_x0[0:1],pred_x0[0:-2]],dim=0)
        # print(pred_x0[0: ,].shape)torch.Size([8, 3, 512, 512])
        loss_1 = first_guidance_scale * (tmp_pred_x0 - pred_x0[0: ,]) / sigma_loss1
        loss_2 = second_guidance_scale * (pred_x0 - predict_mid) / sigma3
        # loss_1.requires_grad_(True) 
        # loss_2.requires_grad_(True) 
        # print(loss_1.shape)torch.Size([8, 3, 512, 512])
        # print(loss_2.shape)torch.Size([8, 3, 512, 512])
        # print(z_variable.shape)torch.Size([8, 4, 64, 64])
        loss = loss_1 + loss_2
        loss.requires_grad_(True) 
        gradient = torch.ones_like(loss)
        loss.backward(gradient)
        guidance = z_variable.grad
        # print(z_variable.requires_grad)#true
        # print(z_variable.is_leaf)#true
        print(guidance)#None

        return e_t, pred_z0, guidance
    
    @torch.no_grad()
    def p_sample_ddim(self, x, e_t,pred_z0, guidance,index , total_guidance_scale,repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0.):
        #sigma_1是第一个高斯的方差 sigma_2是第二个高斯的方差 
        #x从图片变成了图片列表 普通情况是三张，如果是初始状态是一张 没有额外引导 第二个状态 只有一个引导
        # x 每个元素都是(frames,c,h,w) 原来的x就是(bs,c,h,w)
        frames, *_, device = *x.shape, x.device

        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
        # select parameters corresponding to the currently considered timestep
        a_prev = torch.full((frames, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((frames, 1, 1, 1), sigmas[index], device=device)
        if quantize_denoised:
            pred_z0, _, *_ = self.model.first_stage_model.quantize(pred_z0)
        # direction pointing to x_t
        # print(e_t.shape)torch.Size([8, 4, 64, 64])

        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # 利用上一步算出的guidance进行引导
        x_prev = a_prev.sqrt() * pred_z0 + dir_xt + sigma_t**2 * total_guidance_scale * guidance + noise

        return x_prev, pred_z0
    #没改下面的 采样时没用到
    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec