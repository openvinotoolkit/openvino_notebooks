import inspect
from typing import List, Optional, Union, Dict

import PIL
import cv2

import numpy as np
import torch

from transformers import CLIPTokenizer
from diffusers import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
import openvino as ov


def prepare_mask_and_masked_image(image:PIL.Image.Image, mask:PIL.Image.Image):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``np.array`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``np.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``np.float32`` too.

    Args:
        image (Union[np.array, PIL.Image]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array``
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array``.

    Returns:
        tuple[np.array]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, (PIL.Image.Image, np.ndarray)):
        image = [image]

    if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
    elif isinstance(image, list) and isinstance(image[0], np.ndarray):
        image = np.concatenate([i[None, :] for i in image], axis=0)

    image = image.transpose(0, 3, 1, 2)
    image = image.astype(np.float32) / 127.5 - 1.0

    # preprocess mask
    if isinstance(mask, (PIL.Image.Image, np.ndarray)):
        mask = [mask]

    if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
        mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
        mask = mask.astype(np.float32) / 255.0
    elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
        mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    masked_image = image * (mask < 0.5)

    return mask, masked_image


class OVStableDiffusionInpaintingPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae_decoder: ov.Model,
        text_encoder:ov. Model,
        tokenizer: CLIPTokenizer,
        unet: ov.Model,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        vae_encoder: ov.Model = None,
    ):
        """
        Pipeline for text-to-image generation using Stable Diffusion.
        Parameters:
            vae_decoder (Model):
                Variational Auto-Encoder (VAE) Model to decode images to and from latent representations.
            text_encoder (Model):
                Frozen text-encoder. Stable Diffusion uses the text portion of
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
                the clip-vit-large-patch14(https://huggingface.co/openai/clip-vit-large-patch14) variant.
            tokenizer (CLIPTokenizer):
                Tokenizer of class CLIPTokenizer(https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            unet (Model): Conditional U-Net architecture to denoise the encoded image latents.
            vae_encoder (Model):
                Variational Auto-Encoder (VAE) Model to encode images to latent representation.
            scheduler (SchedulerMixin):
                A scheduler to be used in combination with unet to denoise the encoded image latents. Can be one of
                DDIMScheduler, LMSDiscreteScheduler, or PNDMScheduler.
        """
        super().__init__()
        self.scheduler = scheduler
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.text_encoder = text_encoder
        self.unet = unet
        self._text_encoder_output = text_encoder.output(0)
        self._unet_output = unet.output(0)
        self._vae_d_output = vae_decoder.output(0)
        self._vae_e_output = vae_encoder.output(0) if vae_encoder is not None else None
        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8
        self.tokenizer = tokenizer

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        height=512,
        width=512,
        do_classifier_free_guidance=True,
    ):
        """
        Prepare mask as Unet nput and encode input masked image to latent space using vae encoder

        Parameters:
          mask (np.array): input mask array
          masked_image (np.array): masked input image tensor
          heigh (int, *optional*, 512): generated image height
          width (int, *optional*, 512): generated image width
          do_classifier_free_guidance (bool, *optional*, True): whether to use classifier free guidance or not
        Returns:
          mask (np.array): resized mask tensor
          masked_image_latents (np.array): masked image encoded into latent space using VAE
        """
        mask = torch.nn.functional.interpolate(torch.from_numpy(mask), size=(height // 8, width // 8))
        mask = mask.numpy()

        # encode the mask image into latents space so we can concatenate it to the latents
        logits = self.vae_encoder(masked_image)[self._vae_e_output]
        masked_image_latents = logits * 0.18215

        mask = np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            np.concatenate([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )
        return mask, masked_image_latents

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PIL.Image.Image,
        mask_image: PIL.Image.Image,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0,
        output_type: Optional[str] = "pil",
        seed: Optional[int] = None,
    ):
        """
        Function invoked when calling the pipeline for generation.
        Parameters:
            prompt (str or List[str]):
                The prompt or prompts to guide the image generation.
            image (PIL.Image.Image):
                 Source image for inpainting.
            mask_image (PIL.Image.Image):
                 Mask area for inpainting
            negative_prompt (str or List[str]):
                The negative prompt or prompts to guide the image generation.
            num_inference_steps (int, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (float, *optional*, defaults to 7.5):
                Guidance scale as defined in Classifier-Free Diffusion Guidance(https://arxiv.org/abs/2207.12598).
                guidance_scale is defined as `w` of equation 2.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt,
                usually at the expense of lower image quality.
            eta (float, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [DDIMScheduler], will be ignored for others.
            output_type (`str`, *optional*, defaults to "pil"):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): PIL.Image.Image or np.array.
            seed (int, *optional*, None):
                Seed for random generator state initialization.
        Returns:
            Dictionary with keys:
                sample - the last generated image PIL.Image.Image or np.array
        """
        if seed is not None:
            np.random.seed(seed)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get prompt text embeddings
        text_embeddings = self._encode_prompt(
            prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        # prepare mask
        mask, masked_image = prepare_mask_and_masked_image(image, mask_image)
        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, 1)
        latent_timestep = timesteps[:1]

        # get the initial random noise unless the user supplied it
        latents, meta = self.prepare_latents(None, latent_timestep)
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for t in self.progress_bar(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                np.concatenate([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = np.concatenate(
                [latent_model_input, mask, masked_image_latents], axis=1
            )
            # predict the noise residual
            noise_pred = self.unet(
                [latent_model_input, np.array(t, dtype=np.float32), text_embeddings]
            )[self._unet_output]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                torch.from_numpy(noise_pred),
                t,
                torch.from_numpy(latents),
                **extra_step_kwargs,
            )["prev_sample"].numpy()
        # scale and decode the image latents with vae
        image = self.vae_decoder(latents * (1 / 0.18215))[self._vae_d_output]

        image = self.postprocess_image(image, meta, output_type)
        return {"sample": image}

    def _encode_prompt(self, prompt:Union[str, List[str]], num_images_per_prompt:int = 1, do_classifier_free_guidance:bool = True, negative_prompt:Union[str, List[str]] = None):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(
            text_input_ids)[self._text_encoder_output]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(
                text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(
                text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(self, image:PIL.Image.Image = None, latent_timestep:torch.Tensor = None):
        """
        Function for getting initial latents for starting generation
        
        Parameters:
            image (PIL.Image.Image, *optional*, None):
                Input image for generation, if not provided randon noise will be used as starting point
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)
        noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None:
            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                noise = noise * self.scheduler.sigmas[0].numpy()
            return noise, {}
        input_image, meta = preprocess(image)
        latents = self.vae_encoder(input_image)[self._vae_e_output]
        latents = latents * 0.18215
        latents = self.scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        return latents, meta

    def postprocess_image(self, image:np.ndarray, meta:Dict, output_type:str = "pil"):
        """
        Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initila image size (if required), 
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format
        
        Parameters:
            image (np.ndarray):
                Generated image
            meta (Dict):
                Metadata obtained on latents preparing step, can be empty
            output_type (str, *optional*, pil):
                Output format for result, can be pil or numpy
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images
        """
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [img.resize((orig_width, orig_height),
                                    PIL.Image.Resampling.LANCZOS) for img in image]
        else:
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [cv2.resize(img, (orig_width, orig_width))
                         for img in image]
        return image

    def get_timesteps(self, num_inference_steps:int, strength:float):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength
        
        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. 
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start 


def generate_video(
    pipe:OVStableDiffusionInpaintingPipeline,
    prompt:Union[str, List[str]],
    negative_prompt:Union[str, List[str]],
    guidance_scale:float = 7.5,
    num_inference_steps:int = 20,
    num_frames:int = 20,
    mask_width:int = 128,
    seed:int = 9999,
    zoom_in:bool = False,
):
    """
    Zoom video generation function
    
    Parameters:
      pipe (OVStableDiffusionInpaintingPipeline): inpainting pipeline.
      prompt (str or List[str]): The prompt or prompts to guide the image generation.
      negative_prompt (str or List[str]): The negative prompt or prompts to guide the image generation.
      guidance_scale (float, *optional*, defaults to 7.5):
                Guidance scale as defined in Classifier-Free Diffusion Guidance(https://arxiv.org/abs/2207.12598).
                guidance_scale is defined as `w` of equation 2.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt,
                usually at the expense of lower image quality.
      num_inference_steps (int, *optional*, defaults to 50): The number of denoising steps for each frame. More denoising steps usually lead to a higher quality image at the expense of slower inference.
      num_frames (int, *optional*, 20): number frames for video.
      mask_width (int, *optional*, 128): size of border mask for inpainting on each step.
      seed (int, *optional*, None): Seed for random generator state initialization.
      zoom_in (bool, *optional*, False): zoom mode Zoom In or Zoom Out.
    Returns:
      output_path (str): Path where generated video loacated.
    """

    height = 512
    width = height

    current_image = PIL.Image.new(mode="RGBA", size=(height, width))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = PIL.Image.fromarray(255 - mask_image).convert("RGB")
    current_image = current_image.convert("RGB")

    init_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=current_image,
        guidance_scale=guidance_scale,
        mask_image=mask_image,
        seed=seed,
        num_inference_steps=num_inference_steps,
    )["sample"]

    image_grid(init_images, rows=1, cols=1)

    num_outpainting_steps = num_frames
    num_interpol_frames = 30

    current_image = init_images[0]
    all_frames = []
    all_frames.append(current_image)

    for i in range(num_outpainting_steps):
        print(f"Generating image: {i + 1} / {num_outpainting_steps}")

        prev_image_fix = current_image

        prev_image = shrink_and_paste_on_blank(current_image, mask_width)

        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = PIL.Image.fromarray(255 - mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=current_image,
            guidance_scale=guidance_scale,
            mask_image=mask_image,
            seed=seed,
            num_inference_steps=num_inference_steps,
        )["sample"]
        current_image = images[0]
        current_image.paste(prev_image, mask=prev_image)

        # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            interpol_image = current_image
            interpol_width = round((1 - (1 - 2 * mask_width / height) ** (1 - (j + 1) / num_interpol_frames)) * height / 2)
            interpol_image = interpol_image.crop(
                (
                    interpol_width,
                    interpol_width,
                    width - interpol_width,
                    height - interpol_width,
                )
            )

            interpol_image = interpol_image.resize((height, width))

            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = round((1 - (height - 2 * mask_width) / (height - 2 * interpol_width)) / 2 * height)
            prev_image_fix_crop = shrink_and_paste_on_blank(prev_image_fix, interpol_width2)
            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)
            all_frames.append(interpol_image)
        all_frames.append(current_image)

    video_file_name = f"infinite_zoom_{'in' if zoom_in else 'out'}"
    fps = 30
    save_path = video_file_name + ".mp4"
    write_video(save_path, all_frames, fps, reversed_order=zoom_in)
    return save_path


def shrink_and_paste_on_blank(current_image:PIL.Image.Image, mask_width:int):
    """
    Decreases size of current_image by mask_width pixels from each side,
    then adds a mask_width width transparent frame,
    so that the image the function returns is the same size as the input.
    
    Parameters:
        current_image (PIL.Image): input image to transform
        mask_width (int): width in pixels to shrink from each side
    Returns:
       prev_image (PIL.Image): resized image with extended borders
    """

    height = current_image.height
    width = current_image.width

    # shrink down by mask_width
    prev_image = current_image.resize((height - 2 * mask_width, width - 2 * mask_width))
    prev_image = prev_image.convert("RGBA")
    prev_image = np.array(prev_image)

    # create blank non-transparent image
    blank_image = np.array(current_image.convert("RGBA")) * 0
    blank_image[:, :, 3] = 1

    # paste shrinked onto blank
    blank_image[
        mask_width : height - mask_width, mask_width : width - mask_width, :
    ] = prev_image
    prev_image = PIL.Image.fromarray(blank_image)

    return prev_image


def image_grid(imgs:List[PIL.Image.Image], rows:int, cols:int):
    """
    Insert images to grid
    
    Parameters:
        imgs (List[PIL.Image.Image]): list of images for making grid
        rows (int): number of rows in grid
        cols (int): number of columns in grid
    Returns:
        grid (PIL.Image): image with input images collage
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PIL.Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def write_video(file_path:str, frames:List[PIL.Image.Image], fps:float, reversed_order:bool = True, gif:bool = True):
    """
    Writes frames to an mp4 video file and optionaly to gif
    
    Parameters:
        file_path (str): Path to output video, must end with .mp4
        frames (List of PIL.Image): list of frames
        fps (float): Desired frame rate
        reversed_order (bool): if order of images to be reversed (default = True)
        gif (bool): save frames to gif format (default = True)
    Returns:
        None
    """
    if reversed_order:
        frames.reverse()

    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        np_frame = np.array(frame.convert("RGB"))
        cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)

    writer.release()
    if gif:
        frames[0].save(
            file_path.replace(".mp4", ".gif"),
            save_all=True,
            append_images=frames[1:],
            duratiobn=len(frames) / fps,
            loop=0,
        )
