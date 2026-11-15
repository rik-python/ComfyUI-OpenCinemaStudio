# Open Cinema Studio: Film Grade (Beta v0.1)

One ComfyUI node, **Film Grade (Cinema Studio)** (`CS_GradeImage`): a film finish for images, done locally on the CPU. No API keys, no account, no models to download.

It is the colorist's finish from Open Cinema Studio, my entry for the Comfy Org "Open Source Higgsfield" contest. Lens effects are done in linear light, then the film look, then grain and letterbox, in the same order a comp would stack them.

## Install

**ComfyUI Manager:** search for "Open Cinema Studio" in Custom Nodes Manager, install, restart ComfyUI.

**Comfy CLI:**

```bash
comfy node install comfyui-open-cinema-studio
```

**git clone:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rik-python/ComfyUI-OpenCinemaStudio
pip install -r ComfyUI-OpenCinemaStudio/requirements.txt
```

Restart ComfyUI. The node is under `cinema_studio` in the node menu, or search "Film Grade".

## Grade node

Input: `images` (any batch). Output: `graded`.

| Input | Options | What it does |
|---|---|---|
| `grade` | `oldmovie` (default), `warm`, `cool`, `bleach`, `clean`, `none` | The film look |
| `strength` | 0.0 to 1.0 (default 0.5) | How much of the look. 0 leaves the image alone |
| `letterbox` | `none`, `1.85`, `2.39` | Black bars to that aspect |
| `grain` | `none`, `fine` (35mm), `heavy` (16mm) | Film grain on top of the grade's own fine grain |
| `vignette` | `none`, `subtle` (85mm), `strong` (35mm wide open) | Natural cos4 lens fall off |
| `diffusion` | `none`, `light`, `medium`, `heavy` | Pro-Mist style glow (about 1/8, 1/4, 1/2 filter) |
| `aberration` | `none`, `subtle`, `strong` | Lateral chromatic aberration, red out and blue in |

The grades:

- **oldmovie**: faded warm colour, milky blacks, halation, diffusion, vignette, fine grain. The house look.
- **warm**: warm print, more halation.
- **cool**: cool night, cooler and more desaturated.
- **bleach**: bleach bypass, low saturation, more contrast.
- **clean**: a light touch of the same steps.

Good start for AI stills: `oldmovie` at 0.5.

## Known limits (v0.1)

- 8 bit in and out: the node rounds to 8 bit, so it is a finish for display images, not for EXR or HDR plates.
- CPU only (numpy and OpenCV). A 4K frame takes a moment; long video batches are slow.
- Grain is seeded by frame index, so the same batch always gets the same grain.
- Only the grade node ships in this pack. The full Open Cinema Studio app is not public yet.

## License

MIT. See [LICENSE](LICENSE).
