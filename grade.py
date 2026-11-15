"""The colorist's grades, local and free. "oldmovie" is the house look, a 90s film-to-video print;
the others reuse its steps with other amounts.
Order (like a Nuke comp): lift blacks + soft contrast -> colour (desaturate, warm or cool) -> halation (red-orange bloom
from highlights) -> diffusion (Pro-Mist glow) -> slight softness -> vignette -> fine grain.
No ComfyUI imports, so tests run without a server."""
import subprocess

import cv2
import numpy as np

# Each grade: every step's amount at full strength (k = 1). gains: added to R, G, B. contrast: around mid grey.
GRADES = {
    "oldmovie": dict(lift=0.06, flatten=0.12, curve=0.08, contrast=0.0, desat=0.18, gains=(0.04, 0.01, -0.06),
                     halation=0.35, glow=0.18, soften=0.6, vignette=0.22, grain=0.035),
    "warm": dict(lift=0.04, flatten=0.08, curve=0.05, contrast=0.0, desat=0.08, gains=(0.07, 0.02, -0.10),
                 halation=0.45, glow=0.12, soften=0.4, vignette=0.18, grain=0.025),
    "cool": dict(lift=0.05, flatten=0.10, curve=0.06, contrast=0.0, desat=0.25, gains=(-0.05, 0.0, 0.07),
                 halation=0.20, glow=0.15, soften=0.5, vignette=0.28, grain=0.040),
    "bleach": dict(lift=0.02, flatten=0.04, curve=0.0, contrast=0.25, desat=0.45, gains=(0.0, 0.0, 0.0),
                   halation=0.10, glow=0.05, soften=0.3, vignette=0.25, grain=0.045),
    "clean": dict(lift=0.02, flatten=0.04, curve=0.03, contrast=0.0, desat=0.05, gains=(0.01, 0.0, -0.01),
                  halation=0.10, glow=0.06, soften=0.3, vignette=0.10, grain=0.015),
}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _wide_blur(x, sigma):
    """A wide Gaussian blur done at a quarter size: the same look for a glow, about 16 times faster on 1080p."""
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), max(sigma / 4, 0.5)), (w, h), interpolation=cv2.INTER_LINEAR)


def grade(rgb, name="oldmovie", k=0.5, seed=0):
    """One frame (uint8 RGB [H,W,3]) through a grade at strength k (0 = untouched, 1 = full)."""
    p = GRADES[name]
    x = rgb.astype(np.float32) / 255
    x = p["lift"] * k + x * (1 - p["flatten"] * k)  # low contrast, milky blacks, rolled highlights
    x = x - p["curve"] * k * (x - x ** 2)  # gentle S-curve flattening
    x = 0.5 + (x - 0.5) * (1 + p["contrast"] * k)
    lum = x @ LUMA
    x = lum[..., None] + (x - lum[..., None]) * (1 - p["desat"] * k)
    x = x * (1 + k * np.array(p["gains"], np.float32))
    h, w = x.shape[:2]
    size = max(h, w)
    if p["halation"] * k:  # blurred highlights, tinted red-orange, added back
        hi = np.clip((lum - 0.65) / 0.35, 0, 1)[..., None] * x
        x = x + p["halation"] * k * _wide_blur(hi, size * 0.012) * np.array([1.0, 0.45, 0.2], np.float32)
    if p["glow"] * k:  # diffusion glow: a wide soft copy screened over the image
        x = 1 - (1 - x) * (1 - p["glow"] * k * _wide_blur(x, size * 0.02))
    if p["soften"] * k:  # slight video-transfer softness
        x = cv2.GaussianBlur(x, (0, 0), p["soften"] * k * size / 1920)
    if p["vignette"] * k:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        x = x * (1 - p["vignette"] * k * np.clip(r - 0.55, 0, 1) ** 1.5)[..., None]
    if p["grain"] * k:  # fine grain, stronger in the mids like film
        g = cv2.GaussianBlur(np.random.default_rng(seed).standard_normal((h, w)).astype(np.float32), (0, 0), 0.8)
        x = x + (p["grain"] * k * g * (0.4 + 0.6 * (1 - np.abs(lum * 2 - 1))))[..., None]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


# Simple VFX on top of the grade. Grain: amount and clump size (sigma, 1080p pixels); heavy reads as 16mm.
GRAINS = {"fine": (0.03, 0.8), "heavy": (0.07, 1.4)}
# Letterbox: the picture's aspect ratio inside the 16:9 frame, black bars top and bottom.
LETTERBOXES = {"1.85": 1.85, "2.39": 2.39}


def vfx(rgb, letterbox="none", grain="none", seed=0):
    """Film grain (stronger in the mids, like film) and letterbox bars on one uint8 RGB frame."""
    x = rgb
    if grain in GRAINS:
        amount, sigma = GRAINS[grain]
        h, w = x.shape[:2]
        f = x.astype(np.float32) / 255
        lum = f @ LUMA
        g = cv2.GaussianBlur(np.random.default_rng(seed + 7919).standard_normal((h, w)).astype(np.float32), (0, 0), sigma * max(h, w) / 1920)
        f = f + (amount * g * (0.4 + 0.6 * (1 - np.abs(lum * 2 - 1))))[..., None]
        x = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
    if letterbox in LETTERBOXES:
        h, w = x.shape[:2]
        bar = max(0, round((h - w / LETTERBOXES[letterbox]) / 2))
        if bar:
            x = x.copy()
            x[:bar] = 0
            x[h - bar:] = 0
    return x


# ---- Lens: optics, done to light (linear values), before the film look, like a real camera ----
# sRGB transfer (IEC 61966-2-1) as tables: decode by index, encode through a 4096-step table (finer than 8-bit output).
_V = np.arange(256, dtype=np.float64) / 255
DECODE = np.where(_V <= 0.04045, _V / 12.92, ((_V + 0.055) / 1.055) ** 2.4).astype(np.float32)
_L = np.linspace(0, 1, 4096)
ENCODE = (np.where(_L <= 0.0031308, _L * 12.92, 1.055 * _L ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


def to_linear(rgb):
    return DECODE[rgb]


def to_srgb(lin):
    return ENCODE[np.clip(lin * 4095 + 0.5, 0, 4095).astype(np.int32)]


# Diffusion filter (Pro-Mist style): a share of the light is scattered into a halo; the rest passes. Energy
# conserving: out = (1 - a) * image + a * (image through the filter's point spread). The spread is a long tail, like
# real mist filters: a tight core plus ever wider, fainter rings (sigma as a share of the frame diagonal, weight).
DIFFUSIONS = {"light": 0.08, "medium": 0.16, "heavy": 0.28}  # about a 1/8, 1/4 and 1/2 filter
MIST_PSF = ((0.004, 0.40), (0.012, 0.30), (0.035, 0.20), (0.090, 0.10))
# Lateral chromatic aberration: red is magnified a little more than green and blue a little less, growing with the
# square of the distance from the optical centre; zero at the centre. Value: relative size change at the corner
# (0.0008 is under 1 px at a 1080p corner, 0.002 about 2 px: what real lenses show wide open).
ABERRATIONS = {"subtle": 0.0008, "strong": 0.002}
# Natural vignetting: the cos^4 law of illumination. Value: tan of the half field angle at the frame corner, from a
# full-frame lens (21.6 mm half diagonal): an 85 mm gives about 11% fall-off at the corners, a 35 mm about 48%.
VIGNETTES = {"subtle": 21.6 / 85, "strong": 21.6 / 35}


def _radius(h, w):
    """Distance from the centre as a share of the half diagonal (1 at the corners)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.sqrt((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / (np.hypot(w, h) / 2)


_cache = {}


def _cached(key, make):
    if key not in _cache:
        if len(_cache) > 16:  # a film has one frame size: a handful of maps at most
            _cache.clear()
        _cache[key] = make()
    return _cache[key]


def _blur(x, sigma):
    """Gaussian blur; wide ones at a quarter size (same look, much faster)."""
    return _wide_blur(x, sigma) if sigma >= 8 else cv2.GaussianBlur(x, (0, 0), sigma)


def diffusion(lin, amount):
    h, w = lin.shape[:2]
    diag = np.hypot(w, h)
    halo = sum(wt * _blur(lin, s * diag) for s, wt in MIST_PSF)
    return (1 - amount) * lin + amount * halo


def aberration(lin, k):
    h, w = lin.shape[:2]

    def maps(scale):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        r2 = _radius(h, w) ** 2
        f = 1 / (1 + scale * r2)  # sample nearer the centre = that colour is magnified
        # cv2.remap needs float32 maps; numpy 2 promotes to float64 via the np.hypot scalar in _radius
        return (cx + (xx - cx) * f).astype(np.float32), (cy + (yy - cy) * f).astype(np.float32)

    out = lin.copy()
    for ch, scale in ((0, k), (2, -k)):  # red out, blue in, green stays as the reference
        mx, my = _cached(("ca", h, w, scale), lambda: maps(scale))
        out[..., ch] = cv2.remap(lin[..., ch], mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return out


def vignette(lin, t):
    h, w = lin.shape[:2]
    fall = _cached(("vig", h, w, t), lambda: (np.cos(np.arctan(_radius(h, w) * t)) ** 4).astype(np.float32))
    return lin * fall[..., None]


def lens(rgb, vignette_amount="none", diffusion_amount="none", aberration_amount="none"):
    """The lens on one uint8 frame, in the order light meets it: the mist filter in front, then the glass (colour
    fringes), then the fall-off to the corners."""
    if not (vignette_amount in VIGNETTES or diffusion_amount in DIFFUSIONS or aberration_amount in ABERRATIONS):
        return rgb
    x = to_linear(rgb)
    if diffusion_amount in DIFFUSIONS:
        x = diffusion(x, DIFFUSIONS[diffusion_amount])
    if aberration_amount in ABERRATIONS:
        x = aberration(x, ABERRATIONS[aberration_amount])
    if vignette_amount in VIGNETTES:
        x = vignette(x, VIGNETTES[vignette_amount])
    return to_srgb(x)


def finish_frame(rgb, name="none", k=0.5, letterbox="none", grain="none", seed=0, vignette_amount="none",
                 diffusion_amount="none", aberration_amount="none"):
    """The whole finish on one frame, in camera order: the lens, then the film look (grade), then grain, then the
    letterbox mask."""
    x = lens(rgb, vignette_amount, diffusion_amount, aberration_amount)
    x = grade(x, name, k, seed) if name in GRADES and k > 0 else x
    return vfx(x, letterbox, grain, seed)


def needs_finish(name, k, letterbox, grain, vignette_amount="none", diffusion_amount="none", aberration_amount="none"):
    return ((name in GRADES and k > 0) or letterbox in LETTERBOXES or grain in GRAINS or vignette_amount in VIGNETTES
            or diffusion_amount in DIFFUSIONS or aberration_amount in ABERRATIONS)


def grade_video(src, dst, name, k, ffmpeg, letterbox="none", grain="none", lens_fx=None):
    """Grade (and grain, letterbox) every frame of src into dst, sound copied untouched. The grain seed changes per
    frame, like real film."""
    cap = cv2.VideoCapture(src)  # only for size and frame rate: ffprobe is not always installed next to ffmpeg
    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not (w and h and fps):
        raise RuntimeError(f"Could not read the film to grade: {src}")
    fps = f"{fps:.3f}"
    dec = subprocess.Popen([ffmpeg, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps, "-i", "-",
                            "-i", src, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", dst], stdin=subprocess.PIPE)
    n, frame = 0, w * h * 3
    try:
        while True:
            buf = dec.stdout.read(frame)
            if len(buf) < frame:
                break
            enc.stdin.write(finish_frame(np.frombuffer(buf, np.uint8).reshape(h, w, 3), name, k, letterbox, grain, seed=n,
                                         **(lens_fx or {})).tobytes())
            n += 1
    finally:
        enc.stdin.close()
        enc_ok, dec_ok = enc.wait() == 0, dec.wait() == 0
    if not (enc_ok and dec_ok) or not n:
        raise RuntimeError(f"Grading {src} failed after {n} frames.")
    return n
