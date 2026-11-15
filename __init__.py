"""Open Cinema Studio: Film Grade node. The film finish from Open Cinema Studio (lens, grade, grain, letterbox)
on images, local and free."""
import os
import sys

import numpy
import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

sys.path.insert(0, os.path.dirname(__file__))
import grade as _grade  # noqa: E402  (a node input is named grade)


class CS_GradeImage(io.ComfyNode):
    """The film's finish on images (lens effects, grade, grain, letterbox), local and free."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CS_GradeImage",
            display_name="Film Grade (Cinema Studio)",
            category="cinema_studio",
            inputs=[io.Image.Input("images"), io.Combo.Input("grade", options=["none", *_grade.GRADES], default="oldmovie"),
                    io.Float.Input("strength", default=0.5, min=0.0, max=1.0, step=0.05),
                    io.Combo.Input("letterbox", options=["none", *_grade.LETTERBOXES], default="none", optional=True),
                    io.Combo.Input("grain", options=["none", *_grade.GRAINS], default="none", optional=True),
                    io.Combo.Input("vignette", options=["none", *_grade.VIGNETTES], default="none", optional=True),
                    io.Combo.Input("diffusion", options=["none", *_grade.DIFFUSIONS], default="none", optional=True),
                    io.Combo.Input("aberration", options=["none", *_grade.ABERRATIONS], default="none", optional=True)],
            outputs=[io.Image.Output("graded")],
        )

    @classmethod
    def execute(cls, images, grade, strength, letterbox="none", grain="none", vignette="none", diffusion="none", aberration="none"):
        frames = (images.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype("uint8")
        out = [_grade.finish_frame(f, grade, strength, letterbox, grain, seed=i, vignette_amount=vignette, diffusion_amount=diffusion,
                                   aberration_amount=aberration) for i, f in enumerate(frames)]
        return io.NodeOutput(torch.from_numpy(numpy.stack(out)).float() / 255)


class CinemaStudioGradeExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [CS_GradeImage]


async def comfy_entrypoint():
    return CinemaStudioGradeExtension()
