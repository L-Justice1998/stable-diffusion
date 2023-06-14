#Adapted from txt2img.py
import argparse, os, sys, glob
import imageio
import cv2
import math
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddim_video import DDIMSampler_video
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2video-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    # parser.add_argument(
    #     "--n_samples",
    #     type=int,
    #     default=3,
    #     help="how many samples to produce for each given prompt. A.k.a. batch size",
    # )
    parser.add_argument(
        "--frames",
        type=int,
        default=9,
        help="how many frames to produce for each given prompt.",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--sigma1",
        type=float,
        help="first guidance variance",
        default=1.
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        help="first guidance variance",
        default=1.
    )
    parser.add_argument(
        "--sigma3",
        type=float,
        help="first guidance variance",
        default=1.
    )
    parser.add_argument(
        "--first_guidance_scale",
        type=float,
        help="first guidance scale",
        default=1.0
    )
    parser.add_argument(
        "--second_guidance_scale",
        type=float,
        help="second guidance scale",
        default=1.0
    )
    parser.add_argument(
        "--total_guidance_scale",
        type=float,
        help="total guidance scale",
        default=1.
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="progressive video prior hyperparameter",
        default=10
    )
    parser.add_argument(
        '--model', 
        dest='modelDir', 
        type=str, 
        default='RIFEModel', 
        help='directory with trained model files'
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # load RIFE 

    try:
        try:
            try:
                from RIFEModel.RIFE_HDv2 import Model
                RIFE_model = Model()
                RIFE_model.load_model(opt.modelDir, -1)
                print("Loaded v2.x HD model.")
            except:
                from RIFEModel.RIFE_HDv3 import Model
                RIFE_model = Model()
                RIFE_model.load_model(opt.modelDir, -1)
                print("Loaded v3.x HD model.")
        except:
            from RIFEModel.RIFE_HD import Model
            RIFE_model = Model()
            RIFE_model.load_model(opt.modelDir, -1)
            print("Loaded v1.x HD model")
    except:
        from RIFEModel.RIFE import Model
        RIFE_model = Model()
        RIFE_model.load_model(opt.modelDir, -1)
        print("Loaded ArXiv-RIFE model")
    
    # if opt.dpm_solver:
    #     sampler = DPMSolverSampler(model)
    # elif opt.plms:
    #     sampler = PLMSSampler(model)
    # else:
    sampler = DDIMSampler_video(model,RIFE_model)
    sampler1 = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # batch_size = opt.n_samples
    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    # if not opt.from_file:
    #     prompt = opt.prompt
    #     assert prompt is not None
    #     data = [batch_size * [prompt]]
    #只允许有一个prompt
    frames = opt.frames
    n_rows = opt.n_rows if opt.n_rows > 0 else frames
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [frames * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, frames))
            assert len(data) == 1

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        # start_code = torch.randn([frames, opt.C, opt.H // opt.f, opt.W // opt.f], device=device) 
        x_T = []
        first_frame = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f],device = device)
        x_T.append(first_frame)
        for i in range(opt.frames - 1):
            added_noise = torch.randn([1,opt.C, opt.H // opt.f, opt.W // opt.f],device = device) * (math.sqrt(1/(1+opt.alpha*opt.alpha)))
            current_frame = (opt.alpha/math.sqrt((1+opt.alpha*opt.alpha))) * x_T[i] + added_noise
            x_T.append(current_frame)
        start_code = torch.cat(x_T,dim = 0)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(frames * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        frames=frames,
                                                        shape=shape,
                                                        alpha = opt.alpha,
                                                        sigma_1 = opt.sigma1,
                                                        sigma_2 = opt.sigma2,
                                                        sigma_3 = opt.sigma3,
                                                        first_guidance_scale=opt.first_guidance_scale, 
                                                        second_guidance_scale=opt.second_guidance_scale,
                                                        total_guidance_scale=opt.total_guidance_scale,
                                                        conditioning=c,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)
                    # samples_ddim, _ = sampler1.sample(S=opt.ddim_steps,
                    #                                      conditioning=c,
                    #                                      batch_size=frames,
                    #                                      shape=shape,
                    #                                      verbose=False,
                    #                                      unconditional_guidance_scale=opt.scale,
                    #                                      unconditional_conditioning=uc,
                    #                                      eta=opt.ddim_eta,
                    #                                      x_T=start_code)
                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    if not opt.skip_save:
                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            # img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

                    if not opt.skip_grid:
                        all_samples.append(x_checked_image_torch)
            outputs = []
            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                img = Image.fromarray(grid.astype(np.uint8))
                # img = put_watermark(img, wm_encoder)
                outputs.append(img)
                img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

            imageio.mimsave(outpath + '/test.gif', outputs, fps=2)
            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
