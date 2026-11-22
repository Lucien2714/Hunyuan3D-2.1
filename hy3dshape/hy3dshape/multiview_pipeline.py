"""Training-free multi-view geometry pipeline for the official Hunyuan3D 2.1 shape weights.

The input is an unposed image set: 1 to ``MAX_IMAGES`` images of the same object, without view names or
orientation labels, and their order carries no meaning (see ``docs/decisions/0001-unposed-image-set-input.md``).
The whole set produces one latent and one mesh. Two fusion strategies combine the images with the unchanged
single-image weights (see ``docs/design.md`` and ``docs/decisions/0002-training-free-multiview-baseline.md``):

- ``velocity_mean``: every image conditions its own DiT forward pass on the same noisy latent, and the conditional
  velocities are averaged before classifier-free guidance.
- ``token_concat``: the DINOv2 tokens of all images are concatenated along the sequence axis and the DiT runs once
  per step with the longer context.

A single image is handed to ``Hunyuan3DDiTFlowMatchingPipeline.__call__`` unchanged, so its result is the same as
the upstream single-view pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

from .pipelines import Hunyuan3DDiTFlowMatchingPipeline, retrieve_timesteps
from .utils import synchronize_timer

ImageInput = Union[str, Path, Image.Image]
FusionMode = Literal["velocity_mean", "token_concat"]

MAX_IMAGES = 4
FUSION_MODES = ("velocity_mean", "token_concat")
OUTPUT_TYPES = ("trimesh", "latent")


def _normalize_images(images: ImageInput | Sequence[ImageInput]) -> List[Union[str, Image.Image]]:
    """Turn the public ``images`` argument into a flat list of paths (``str``) and PIL images."""
    if isinstance(images, (str, Path, Image.Image)):
        items = [images]
    elif isinstance(images, (list, tuple)):
        items = list(images)
    else:
        raise ValueError(
            f"`images` must be an image path, a PIL image, or a flat list/tuple of them showing one object, "
            f"got {type(images).__name__}."
        )

    if len(items) == 0:
        raise ValueError(f"Expected 1 to {MAX_IMAGES} images of one object, got 0. Pass at least one image.")
    if len(items) > MAX_IMAGES:
        raise ValueError(
            f"Expected 1 to {MAX_IMAGES} images of one object, got {len(items)}. Pass at most {MAX_IMAGES} images."
        )

    normalized = []
    for index, item in enumerate(items):
        if isinstance(item, Path):
            item = str(item)
        if not isinstance(item, (str, Image.Image)):
            raise ValueError(
                f"Image #{index + 1} has type {type(item).__name__}; each image must be a path or a PIL image. "
                f"View-name dicts, tensors and nested lists are not supported: one call takes one object."
            )
        normalized.append(item)
    return normalized


def _describe(item: Union[str, Image.Image], index: int) -> str:
    return f"image #{index + 1} ({item})" if isinstance(item, str) else f"image #{index + 1} (PIL image)"


class Hunyuan3DUnposedMultiViewPipeline(Hunyuan3DDiTFlowMatchingPipeline):
    """Geometry pipeline that turns an unposed set of 1-4 images of one object into one mesh.

    Loading is inherited: ``Hunyuan3DUnposedMultiViewPipeline.from_pretrained("tencent/Hunyuan3D-2.1")`` reads the
    official ``hunyuan3d-dit-v2-1`` config and weights without any change to their format.
    """

    @torch.inference_mode()
    def __call__(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        fusion: FusionMode = "velocity_mean",
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
        octree_resolution: int = 384,
        num_chunks: int = 8000,
        output_type: Literal["trimesh", "latent"] = "trimesh",
        enable_pbar: bool = True,
    ) -> Union[List[Optional[trimesh.Trimesh]], torch.Tensor]:
        """Generate one mesh from an unposed image set of one object.

        Args:
            images: An image path or PIL image, or a flat list/tuple of 1 to ``MAX_IMAGES`` of them. All images show
                the same object; their order has no meaning.
            fusion: ``"velocity_mean"`` or ``"token_concat"``. Ignored when only one image is given.
            num_inference_steps, guidance_scale, generator, octree_resolution, num_chunks, enable_pbar: Same meaning
                as in ``Hunyuan3DDiTFlowMatchingPipeline.__call__``.
            output_type: ``"trimesh"`` returns a list with one mesh (``None`` if surface extraction found nothing);
                ``"latent"`` returns the final latent of shape ``(1, L, C)``.
        """
        items = _normalize_images(images)
        if fusion not in FUSION_MODES:
            raise ValueError(f"Unknown fusion {fusion!r}; choose one of {', '.join(FUSION_MODES)}.")
        if output_type not in OUTPUT_TYPES:
            raise ValueError(f"Unknown output_type {output_type!r}; choose one of {', '.join(OUTPUT_TYPES)}.")

        if len(items) == 1:
            # Hand over before touching any state (RNG, scheduler, latents) so that one image gives exactly the
            # upstream single-view result.
            return super().__call__(
                image=items[0],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                output_type=output_type,
                enable_pbar=enable_pbar,
            )

        if fusion == "token_concat" and getattr(self.model, "use_attention_pooling", False):
            raise RuntimeError(
                "fusion='token_concat' needs a denoiser without fixed-length attention pooling, but this model has "
                "`use_attention_pooling: true`. Load the official tencent/Hunyuan3D-2.1 weights with subfolder "
                "hunyuan3d-dit-v2-1, or use fusion='velocity_mean'."
            )

        device = self.device
        dtype = self.dtype
        # Same switches as Hunyuan3DDiTFlowMatchingPipeline.__call__ (hy3dshape/hy3dshape/pipelines.py): CFG runs
        # whenever guidance_scale >= 0, unless the denoiser embeds the guidance scale itself.
        guidance_embed = getattr(self.model, "guidance_embed", None) is True
        do_classifier_free_guidance = guidance_scale >= 0 and not guidance_embed

        conds = self._encode_image_set(items)
        uncond = self.conditioner.unconditional_embedding(1)["main"].to(device=device, dtype=dtype)

        sigmas = np.linspace(0, 1, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)
        latents = self.prepare_latents(1, dtype, device, generator)

        guidance = None
        if guidance_embed:
            guidance = torch.tensor([guidance_scale], device=device, dtype=dtype)

        if fusion == "token_concat":
            # (1, N * L, C); the unconditional branch repeats the single-image zero tokens N times so that both CFG
            # rows have the same length.
            contexts = torch.cat(conds, dim=1)
            if do_classifier_free_guidance:
                contexts = torch.cat([contexts, uncond.repeat(1, len(conds), 1)], dim=0)
            contexts = {"main": contexts}

        with synchronize_timer("Diffusion Sampling"):
            for t in tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:"):
                if fusion == "velocity_mean":
                    velocity = self._mean_velocity(
                        latents, t, conds, uncond, guidance_scale, do_classifier_free_guidance, guidance
                    )
                else:
                    velocity = self._concat_velocity(
                        latents, t, contexts, guidance_scale, do_classifier_free_guidance, guidance
                    )
                # One scheduler step per timestep: FlowMatchEulerDiscreteScheduler.step advances its step index.
                latents = self.scheduler.step(velocity, t, latents).prev_sample

        # Export arguments are the defaults of Hunyuan3DDiTFlowMatchingPipeline.__call__, not of _export().
        return self._export(
            latents,
            output_type,
            box_v=1.01,
            mc_level=0.0,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=None,
            enable_pbar=enable_pbar,
        )

    def _encode_image_set(self, items: List[Union[str, Image.Image]]) -> List[torch.Tensor]:
        """Preprocess and encode each image on its own; returns N tensors of shape ``(1, L, C)``.

        For the official config ``L = 37 * 37 + 1 = 1370`` DINOv2 tokens (CLS included) and ``C = 1024``.
        """
        prepared = []
        for index, item in enumerate(items):
            # Never call prepare_image() on the whole list: it stacks the images as a batch of separate objects.
            try:
                cond_inputs = self.prepare_image(item)
            except ValueError as exc:
                raise ValueError(
                    f"Could not preprocess {_describe(item, index)}: {exc}. Each image needs a visible foreground "
                    f"(an alpha channel or a background that can be removed)."
                ) from exc
            image = cond_inputs.pop("image")  # (1, 3, 512, 512), CPU float in [-1, 1]
            prepared.append((image.numpy().tobytes(), image, cond_inputs))

        # Fix the processing order by preprocessed pixel content, so that reordering the same set reproduces the
        # same floating-point result. The order is not passed to the model.
        prepared.sort(key=lambda entry: entry[0])

        conds = []
        for _, image, cond_inputs in prepared:
            cond = self.encode_cond(
                image=image,
                additional_cond_inputs=cond_inputs,
                do_classifier_free_guidance=False,
                dual_guidance=False,
            )
            if set(cond.keys()) != {"main"}:
                raise RuntimeError(
                    f"The multi-view pipeline expects a conditioner that returns only 'main' tokens "
                    f"(SingleImageEncoder in the official hunyuan3d-dit-v2-1 config), got {sorted(cond.keys())}. "
                    f"Load tencent/Hunyuan3D-2.1 with subfolder hunyuan3d-dit-v2-1."
                )
            conds.append(cond["main"].to(dtype=self.dtype))
        return conds

    def _model_timestep(self, t: torch.Tensor, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        # Copied from Hunyuan3DDiTFlowMatchingPipeline.__call__: the DiT takes timesteps in [0, 1].
        timestep = t.expand(batch_size).to(dtype)
        return timestep / self.scheduler.config.num_train_timesteps

    def _mean_velocity(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        conds: List[torch.Tensor],
        uncond: torch.Tensor,
        guidance_scale: float,
        do_classifier_free_guidance: bool,
        guidance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Average the conditional velocities of all images at the same ``(x_t, t)``, then apply CFG.

        ``latents`` is ``(1, L, C)``. Every forward pass uses batch 1; the unconditional prediction is shared by all
        images. Accumulation is in float32 and only the running sum is kept.
        """
        timestep = self._model_timestep(t, 1, latents.dtype)
        uncond_velocity = None
        if do_classifier_free_guidance:
            uncond_velocity = self.model(latents, timestep, {"main": uncond}, guidance=guidance).float()

        velocity_sum = None
        for cond in conds:
            velocity = self.model(latents, timestep, {"main": cond}, guidance=guidance).float()
            velocity_sum = velocity if velocity_sum is None else velocity_sum.add_(velocity)
        velocity = velocity_sum / len(conds)

        if do_classifier_free_guidance:
            # Equal to the mean of the per-image guided velocities.
            velocity = uncond_velocity + guidance_scale * (velocity - uncond_velocity)
        return velocity.to(latents.dtype)

    def _concat_velocity(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        contexts: dict,
        guidance_scale: float,
        do_classifier_free_guidance: bool,
        guidance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """One DiT pass with the concatenated tokens; same CFG as the upstream single-image loop.

        ``contexts["main"]`` is ``(2, N * L, C)`` with CFG (conditional row first) and ``(1, N * L, C)`` without.
        """
        # Copied from Hunyuan3DDiTFlowMatchingPipeline.__call__ (hy3dshape/hy3dshape/pipelines.py).
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents
        timestep = self._model_timestep(t, latent_model_input.shape[0], latents.dtype)
        noise_pred = self.model(latent_model_input, timestep, contexts, guidance=guidance)
        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred
