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
from kornia.morphology import dilation
import torchvision
import torchvision.transforms as T
import os



#Adapted from DDIMSampler
class DDIMSampler_video(object):
    def __init__(self, model,RIFE_model,sod_model,schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.RIFE_model = RIFE_model
        self.sod_model = sod_model
        self.sod_model.eval()
        self.backwarp_tenGrid = {}

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    
    def warp(self,tenInput, tenFlow,device):
        k = (str(tenFlow.device), str(tenFlow.size()))
        if k not in self.backwarp_tenGrid:
            tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
                1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
                1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            self.backwarp_tenGrid[k] = torch.cat(
                [tenHorizontal, tenVertical], 1).to(device)

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

        g = (self.backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
        return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


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
               sigma_1, 
               sigma_2, 
               sigma_3,
               first_guidance_scale, 
               second_guidance_scale,
               total_guidance_scale,
               detector,
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
                                                    alpha = alpha,sigma_1=sigma_1, 
                                                    sigma_2=sigma_2,sigma_3=sigma_3,
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
                                                    detector=detector
                                                    )
        return samples, intermediates
    
    @torch.no_grad()
    def ddim_sampling(self, cond,shape,alpha,sigma_1, 
                      sigma_2, sigma_3, first_guidance_scale, 
                      second_guidance_scale,total_guidance_scale,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,use_original_steps=False,
                      detector=True
                      ):
        device = self.model.betas.device
        # 原来的xT是bs*隐空间的三维
        # 现在改为frames*三维
        frames, C, H, W = shape
        if x_T is None:
            x_T = []
            # 用progressive的方法生成初始噪声
            first_frame = torch.randn((1,C, H, W),device = device)
            x_T.append(first_frame)
            if timesteps is None:
                timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
            elif timesteps is not None and not ddim_use_original_steps:
                subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
                timesteps = self.ddim_timesteps[:subset_end]
            if detector:
                img = x_T[0]
                # 要对第一帧进行一次完全的去噪得到干净的mask
                intermediates_first_frame = {'x_inter': [x_T[0]], 'pred_z0': [x_T[0]]}
                time_range_first_frame = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
                total_steps_first_frame = timesteps if ddim_use_original_steps else timesteps.shape[0]
                print(f"Running DDIM Sampling for the first frame with {total_steps_first_frame} timesteps")
                iterator_first_frame = tqdm(time_range_first_frame, desc='DDIM Sampler', total=total_steps_first_frame)

                for i, step in enumerate(iterator_first_frame):
                    #这个循环是时间尺度上的
                    index = total_steps_first_frame - i - 1
                    ts = torch.full((1,), step, device=device, dtype=torch.long)

                    if mask is not None:
                        assert x0 is not None
                        img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                        img = img_orig * mask + (1. - mask) * img

                    outs = self.p_sample_ddim(img, cond[0:1,], ts,index=index, 
                                            use_original_steps=ddim_use_original_steps,
                                            quantize_denoised=quantize_denoised, temperature=temperature,
                                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning[0:1,])

                    img, pred_z0 = outs
                    if callback: callback(i)
                    if img_callback: img_callback(pred_z0, i)

                    if index % log_every_t == 0 or index == total_steps_first_frame - 1:
                        intermediates_first_frame['x_inter'].append(img)
                        intermediates_first_frame['pred_z0'].append(pred_z0)


                first_frame_x = self.model.decode_first_stage(img)
                first_frame_x = torch.clamp((first_frame_x + 1.0) / 2.0, min=0.0, max=1.0)

                first_frame_x = torch.squeeze(first_frame_x,dim=0)
                first_frame_x = first_frame_x.cpu().permute(1, 2, 0).numpy()
                first_frame_x = torch.from_numpy(first_frame_x).permute(2, 0, 1)
                first_frame_x = 255. * rearrange(first_frame_x.cpu().numpy(), 'c h w -> h w c')
                first_frame_x_img = Image.fromarray(first_frame_x.astype(np.uint8))

                # 将图像保存为PNG格式的文件
                # file_name =  os.path.join('./detector/results', 'img.jpg')
                # first_frame_x_img.save(file_name)

                #mask检测
                sod_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
                first_frame_x_img = sod_transform(first_frame_x_img)
                first_frame_x_img = first_frame_x_img[None]
                with torch.no_grad():
                    pred, loss = self.sod_model(first_frame_x_img.half().to(device))
                #pred是mask
                pred = pred[5].data
                pred.requires_grad_(False)

                target_size = (512, 512)
                foreground_mask = F.interpolate(pred[0].unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
                # pixel_kernel = torch.ones(7, 7, device=foreground_mask.device, dtype=foreground_mask.dtype)
                # torchvision.utils.save_image(mask, os.path.join('./detector/results', 'mask.jpg'))
                # 像素空间扩大
                # foreground_mask = dilation(foreground_mask, pixel_kernel)[0]
                # torchvision.utils.save_image(foreground_mask[None], os.path.join('./detector/results', 'foreground_mask.jpg'))
                # follow Text2Video-Zero/text_to_video_pipeline.py line 420~423
                kernel = torch.ones(5, 5, device=foreground_mask.device, dtype=foreground_mask.dtype)
                foreground_mask = T.Resize(size = (H, W),interpolation = T.InterpolationMode.NEAREST)(foreground_mask)
                # 隐空间扩大
                foreground_mask = dilation(foreground_mask, kernel)[0]
                #print(foreground_mask.shape)(1,64,64)
                #此时foreground mask是前景为1背景为0
                background_mask = 1 - foreground_mask

                #这里初始化噪声
                for i in range(frames-1):
                    added_noise = torch.randn((1,C,H,W),device = device) * (math.sqrt(1/(1+alpha*alpha))) * foreground_mask
                    current_frame = background_mask *  x_T[i] + foreground_mask * math.sqrt(alpha*alpha/(1+alpha*alpha)) * x_T[i] + added_noise
                    x_T.append(current_frame)
                img = torch.cat(x_T,dim = 0)
            else:
                for i in range(frames-1):
                    added_noise = torch.randn((1,C,H,W),device = device) * (math.sqrt(1/(1+alpha*alpha))) 
                    current_frame = math.sqrt(alpha*alpha/(1+alpha*alpha)) * x_T[i] + added_noise
                    x_T.append(current_frame)
                img = torch.cat(x_T,dim = 0)


        intermediates = {'x_inter': [img], 'pred_z0': [img]}
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

            with torch.enable_grad():
                e_t, guidance = self.cal_latent_guidance(img, cond, ts, 
                                                        sigma_1,sigma_2,sigma_3, first_guidance_scale, second_guidance_scale,foreground_mask,
                                                        index=index, use_original_steps=ddim_use_original_steps,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning)

            torch.cuda.empty_cache()
            outs = self.p_sample_ddim_video(img, e_t, guidance,index=index, 
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
                intermediates['pred_z0'].append(pred_z0)

        return img, intermediates
    
    def cal_guidance(self, z, c, t,
                     sigma1,sigma2,sigma3, first_guidance_scale, second_guidance_scale,
                     index, use_original_steps=False, 
                     score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        z是上一时刻的所有隐空间有噪声的帧 是(frames,c,h,w)
        返回值是e_t, pred_z0, guidance
        """
        frames, c, h, w, device = *z.shape, z.device
        #unet的in_channel是4通道 第一个通道本来是bs的
        #用z_variable变量追踪梯度 
        z_variable = torch.tensor(z, requires_grad = True)


        # 无条件采样
        with torch.no_grad():
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

        #梯度测试
        # e_t.backward(torch.ones_like(e_t))
        # print(z_variable.grad)
        # exit()
        #e-1~e-2

        #参数计算 select parameters corresponding to the currently considered timestep
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        
        a_t = torch.full((frames, 1, 1, 1), alphas[index], device=device)
        sqrt_one_minus_at = torch.full((frames, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
        
        #三个高斯方差
        sigma1 = torch.full((2, 1, 1, 1), sigma1,device=device)
        sigma2 = torch.full((frames-2, 1, 1, 1), sigma2,device=device)
        sigma_loss1 = torch.cat((sigma1,sigma2),dim = 0)
        sigma3 = torch.full((frames-2, 1, 1, 1), sigma3,device=device)

        #所有的predict z0
        pred_z0 = (z_variable - sqrt_one_minus_at * e_t) / a_t.sqrt()
 
        # pred_z0.backward(torch.ones_like(pred_z0))
        # print(z_variable.grad) #e0~e1
        # print(z_variable.grad.shape)
        # # print(alphas[index])0.0058
        # # print(sqrt_one_minus_alphas[index])0.9971
        
        #decode follow txt2img.py line 313-322
        pred_x0 = self.model.decode_first_stage(pred_z0)

        # pred_x0.backward(torch.ones_like(pred_x0))
        # print(z_variable.grad) e1~e2
        # print(z_variable.grad.shape)

        #将pred_x0转到[0,1]
        pred_x0 = torch.clamp((pred_x0 + 1.0) / 2.0, min=0.0, max=1.0)
        pred_x0 = pred_x0.permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        

        #预测中间帧 [0,1]
        # print(pred_x0.shape)torch.Size([8, 3, 512, 512])
        
        # pred_x0.backward(torch.ones_like(pred_x0))
        # print(z_variable.grad)
        # print(z_variable.grad.shape)
        # exit()
        #和上次相比梯度少了一半

        _, C, H, W = pred_x0.shape
        #如果是前两帧就取自己，相当于没有插值
        img0 = pred_x0[0:-2]
        img1 = pred_x0[2:]
        #这一部分是follow原来插值的处理
        ph = ((H - 1) // 32 + 1) * 32
        pw = ((W - 1) // 32 + 1) * 32
        padding = (0, pw - W, 0, ph - H)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        predict_mid = self.RIFE_model.inference(img0.detach(), img1)

        down_sampling_predict_mid = T.Resize(size = (h, w),interpolation = T.InterpolationMode.NEAREST)(down_sampling_predict_mid)

        # print(mid.shape)torch.Size([8, 3, 512, 512])

        # predict_mid.backward(torch.ones_like(predict_mid))
        # print(z_variable.grad)
        # print(z_variable.grad.shape)
        # print(predict_mid.shape)
        # exit()
        #梯度波动小

        #两项loss
        tmp_pred_x0 = torch.cat([pred_x0[0:1],pred_x0[0:1],pred_x0[0:-2]],dim=0)
        # print(pred_x0[0: ,].shape)torch.Size([8, 3, 512, 512])
        #只对i时刻的z求导
        loss_1 = first_guidance_scale * (- (tmp_pred_x0.detach() - pred_x0[0: ,])**2) / sigma_loss1**2
        # print(- (tmp_pred_x0 - pred_x0[0: ,])**2) 
        # e-1~e-2
        #因为要使shape保持一致 因此对于前两帧 插值的时候使得两张都是前两帧 插值出来的也应该是前两帧
        # pred_x0也是本身 因此为0
        loss_2 = second_guidance_scale * (- (pred_x0[1:-1].detach() - predict_mid)**2) / sigma3**2

        # e-1~e-3
        loss = loss_1.sum() + loss_2.sum()
        # loss.sum().requires_grad_(True) 
        loss.sum().backward()
        guidance = z_variable.grad
        # guidance = torch.autograd.grad(outputs = loss.sum(), inputs = z_variable)
        
        # print(z_variable.requires_grad)#true
        # print(z_variable.is_leaf)#true
        # print(guidance)
        # print(guidance.shape)
        #e1~e3
        # print(guidance)
        return e_t, guidance
    
    def cal_latent_guidance(self, z, c, t,
                     sigma1,sigma2,sigma3, first_guidance_scale, second_guidance_scale,foreground_mask,
                     index, use_original_steps=False, 
                     score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None):
        """
        z是上一时刻的所有隐空间有噪声的帧 是(frames,c,h,w)
        返回值是e_t, pred_z0, guidance
        """
        frames, _, h, w, device = *z.shape, z.device
        #unet的in_channel是4通道 第一个通道本来是bs的
        #用z_variable变量追踪梯度 
        # z_variable = torch.tensor(z, requires_grad = True)
        z_variable = z.clone().detach().requires_grad_(True)


        # 无条件采样
        with torch.no_grad():
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

        #梯度测试
        # e_t.backward(torch.ones_like(e_t))
        # print(z_variable.grad)
        # exit()
        #e-1~e-2

        #参数计算
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
        sigma3 = torch.full((frames-2, 1, 1, 1), sigma3,device=device)

        #所有的predict z0
        pred_z0 = (z_variable - sqrt_one_minus_at * e_t) / a_t.sqrt()
        
        #decode follow txt2img.py line 313-322 
        with torch.no_grad():
            pred_x0 = self.model.decode_first_stage(pred_z0.detach())

            # pred_x0.backward(torch.ones_like(pred_x0))
            # print(z_variable.grad) e1~e2
            # print(z_variable.grad.shape)

            #将pred_x0转到[0,1]
            pred_x0 = torch.clamp((pred_x0 + 1.0) / 2.0, min=0.0, max=1.0)
            pred_x0 = pred_x0.permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        

        _, C, H, W = pred_x0.shape
        latent_img0 = pred_z0[0:-2].detach()
        latent_img1 = pred_z0[2:]
        with torch.no_grad():
            img0 = pred_x0[0:-2]
            img1 = pred_x0[2:]
            ph = ((H - 1) // 32 + 1) * 32
            pw = ((W - 1) // 32 + 1) * 32
            padding = (0, pw - W, 0, ph - H)
            img0 = F.pad(img0, padding)
            img1 = F.pad(img1, padding)

            # 得到的光流resize符合z0
            inference_img = torch.cat((img0,img1),1)
            #0是返回元组的第一个元素 光流list -1是前向过程的光流的最后结果 2：4是前向光流 0：2是后向
            flow_list, mask_list, _ =  self.RIFE_model.flownet(inference_img)
            #最后的结果是merged[2]
        
            # predict_flow = flow_list[-1][:,2:4]
            # print(predict_flow.shape)(6,2,512,512)
            # predict_flow = T.Resize(size = (h, w),interpolation = T.InterpolationMode.NEAREST)(predict_flow)
            #follow ./RIFEModel/IFNet_HDv3.py forward
            for i,flow in enumerate(flow_list):
                flow_list[i] = T.Resize(size = (h, w),interpolation = T.InterpolationMode.NEAREST)(flow)
            
            for i,mask in enumerate(mask_list):
                mask_list[i] = T.Resize(size = (h, w),interpolation = T.InterpolationMode.NEAREST)(mask)
        latent_merged = []
        for i in range(3):
            warped_latent_img0 = self.warp(latent_img0,flow_list[i][:,:2],device)
            warped_latent_img1 = self.warp(latent_img1,flow_list[i][:,2:4],device)
            latent_merged.append(warped_latent_img0 * mask_list[i] + warped_latent_img1 * (1 - mask_list[i]))

        #将predict_mid做resize
        # predict_mid = self.RIFE_model.inference(img0.detach(), img1)
        # predict_mid = self.model.get_first_stage_encoding(self.model.encode_first_stage(predict_mid))

        #原始的简单版本
        # predict_mid = self.RIFE_model.inference(img0, img1)
        # down_sampling_predict_mid = T.Resize(size = (h, w),interpolation = T.InterpolationMode.NEAREST)(down_sampling_predict_mid)
        # img0 = pred_z0[0:-2].detach()
        # img1 = pred_z0[2:]
        # predict_mid = (img0 + img1) * 0.5

        tmp_pred_z0 = torch.cat([pred_z0[0:1],pred_z0[0:1],pred_z0[0:-2]],dim=0)
        #光流resize版本
        # predict_mid = (1 - foreground_mask) * latent_img0 + foreground_mask * predict_flow * latent_img0

        #predict_mid resize版本

        #只对i时刻的z求导
        loss_1 = first_guidance_scale * (- (tmp_pred_z0.detach() * foreground_mask - pred_z0[0: ,] * foreground_mask)**2) / sigma_loss1**2
        # print(- (tmp_pred_x0 - pred_x0[0: ,])**2) 
        # e-1~e-2

        loss_2 = second_guidance_scale * (- (pred_z0[1:-1].detach() * foreground_mask - latent_merged[2] * foreground_mask)**2) / sigma3**2

        # e-1~e-3
        loss = loss_1.sum() + loss_2.sum()
        # loss.sum().requires_grad_(True) 
        loss.sum().backward()
        guidance = z_variable.grad
        # guidance = torch.autograd.grad(outputs = loss.sum(), inputs = z_variable,allow_unused=True)
        
        # print(z_variable.requires_grad)#true
        # print(z_variable.is_leaf)#true
        # print(guidance)
        # print(guidance.shape)
        #e1~e3
        return e_t, guidance
    
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    @torch.no_grad()
    def p_sample_ddim_video(self, z, e_t, guidance,index , total_guidance_scale,repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0.):
        #sigma_1是第一个高斯的方差 sigma_2是第二个高斯的方差 
        #x从图片变成了图片列表 普通情况是三张，如果是初始状态是一张 没有额外引导 第二个状态 只有一个引导
        # x 每个元素都是(frames,c,h,w) 原来的x就是(bs,c,h,w)
        frames, *_, device = *z.shape, z.device
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas

        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
        # select parameters corresponding to the currently considered timestep
        a_prev = torch.full((frames, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((frames, 1, 1, 1), sigmas[index], device=device)
        a_t = torch.full((frames, 1, 1, 1), alphas[index], device=device)
        sqrt_one_minus_at = torch.full((frames, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        e_t_new = e_t - sqrt_one_minus_at * total_guidance_scale * guidance
        print(sqrt_one_minus_at * total_guidance_scale * guidance)
        # pred_z0_old = (z - sqrt_one_minus_at * e_t) / a_t.sqrt()
        pred_z0 = (z - sqrt_one_minus_at * e_t_new) / a_t.sqrt()
 
        if quantize_denoised:
            pred_z0, _, *_ = self.model.first_stage_model.quantize(pred_z0)
        # direction pointing to z_t
        # print(e_t.shape)torch.Size([8, 4, 64, 64])
        # dir_zt_old = (1. - a_prev - sigma_t**2).sqrt() * e_t
        dir_zt = (1. - a_prev - sigma_t**2).sqrt() * e_t_new
        noise = sigma_t * noise_like(z.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # 利用上一步算出的guidance进行引导
        # x_prev_old = a_prev.sqrt() * pred_z0_old + dir_zt_old  + noise
        x_prev = a_prev.sqrt() * pred_z0 + dir_zt  + noise
        # print(index,'new')
        # print(x_prev)
        # print(index,'old')
        # print(x_prev_old)
        #正常e-2~e0 不超过3
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