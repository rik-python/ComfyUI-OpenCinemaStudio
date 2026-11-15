import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import grade as gr  # noqa: E402


class Grade(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.img = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)

    def test_strength_zero_leaves_the_frame_alone(self):
        for name in gr.GRADES:
            out = gr.grade(self.img, name, 0.0)
            self.assertLessEqual(int(np.abs(out.astype(int) - self.img.astype(int)).max()), 1, name)

    def test_every_grade_changes_the_frame_and_keeps_its_shape(self):
        for name in gr.GRADES:
            out = gr.grade(self.img, name, 1.0, seed=3)
            self.assertEqual(out.shape, self.img.shape)
            self.assertEqual(out.dtype, np.uint8)
            self.assertGreater(float(np.abs(out.astype(int) - self.img.astype(int)).mean()), 1.0, name)

    def test_old_movie_lifts_the_blacks_and_bleach_drains_the_colour(self):
        black = np.zeros((90, 160, 3), np.uint8)
        self.assertGreater(gr.grade(black, "oldmovie", 1.0)[45, 80].mean(), 8)
        red = np.full((90, 160, 3), (200, 40, 40), np.uint8)
        sat = lambda im: float(im.max(axis=2).astype(int).mean() - im.min(axis=2).astype(int).mean())  # noqa: E731
        self.assertLess(sat(gr.grade(red, "bleach", 1.0)), sat(gr.grade(red, "warm", 1.0)))


class Lens(unittest.TestCase):
    def test_srgb_tables_round_trip(self):
        v = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
        self.assertLessEqual(int(np.abs(gr.to_srgb(gr.to_linear(v)).astype(int) - v).max()), 1)

    def test_vignette_follows_cos4_and_leaves_the_centre(self):
        flat = np.ones((201, 301, 3), np.float32)
        out = gr.vignette(flat, gr.VIGNETTES["strong"])
        self.assertAlmostEqual(float(out[100, 150, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(out[0, 0, 0]), float(np.cos(np.arctan(gr.VIGNETTES["strong"])) ** 4), places=2)

    def test_diffusion_moves_light_into_halos_without_adding_any(self):
        lin = np.zeros((180, 320, 3), np.float32)
        lin[85:95, 155:165] = 1.0  # a small bright source on black
        out = gr.diffusion(lin, gr.DIFFUSIONS["heavy"])
        self.assertAlmostEqual(float(out.sum()), float(lin.sum()), delta=float(lin.sum()) * 0.02)
        self.assertGreater(float(out[60, 160, 0]), 0.0, "light spreads round the source")
        self.assertLess(float(out[90, 160, 0]), 1.0)

    def test_colour_fringes_grow_from_the_centre(self):
        h, w = 181, 321
        lin = np.zeros((h, w, 3), np.float32)
        lin[:, ::8] = 1.0  # thin white vertical lines
        out = gr.aberration(lin, 0.02)
        centre, edge = out[90, 156:166], out[90, w - 12:w]
        self.assertLess(float(np.abs(centre[:, 0] - centre[:, 2]).max()), 0.05, "no fringe at the optical centre")
        self.assertGreater(float(np.abs(edge[:, 0] - edge[:, 2]).max()), 0.2, "red and blue split near the edge")
        self.assertTrue(np.allclose(out[..., 1], lin[..., 1]), "green is the reference")

    def test_finish_frame_runs_every_effect_and_keeps_the_frame(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
        out = gr.finish_frame(img, "oldmovie", 0.5, "2.39", "heavy", 1, "strong", "medium", "subtle")
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(int(out[0].max()), 0, "letterbox bar on top")
        self.assertTrue(gr.needs_finish("none", 0.5, "none", "none", "none", "light", "none"))
        self.assertFalse(gr.needs_finish("none", 0.5, "none", "none"))
        self.assertTrue(np.array_equal(gr.lens(img), img), "no lens effect: untouched")


if __name__ == "__main__":
    unittest.main()
