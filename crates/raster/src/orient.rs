//! Exact right-angle turns and flips of a whole surface (Image › Image Rotation), tile by tile.
//!
//! Every target tile is built on its own (in parallel) by copying encoded pixels straight out of
//! the source tiles: no per-pixel decode/encode and no tile lookup per pixel, so flipping a large
//! document is about as fast as copying it once.

use std::collections::BTreeSet;
use std::sync::Arc;

use photocraft_geom::{Rect, TILE_SIZE, TileCoord};
use rayon::prelude::*;

use crate::{Surface, Tile};

/// A flip or right-angle turn of a `w × h` canvas, as pixel-index maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orient {
    FlipHorizontal,
    FlipVertical,
    Rotate180,
    Cw90,
    Ccw90,
}

impl Orient {
    /// Where pixel `(x, y)` of a `w × h` canvas goes.
    pub fn map(self, w: i32, h: i32, x: i32, y: i32) -> (i32, i32) {
        match self {
            Orient::FlipHorizontal => (w - 1 - x, y),
            Orient::FlipVertical => (x, h - 1 - y),
            Orient::Rotate180 => (w - 1 - x, h - 1 - y),
            Orient::Cw90 => (h - 1 - y, x),
            Orient::Ccw90 => (y, w - 1 - x),
        }
    }

    /// The source pixel that [`Self::map`] sends to `(x, y)`.
    pub fn inverse(self, w: i32, h: i32, x: i32, y: i32) -> (i32, i32) {
        match self {
            Orient::Cw90 => (y, h - 1 - x),
            Orient::Ccw90 => (w - 1 - y, x),
            // Flips and the half turn are their own inverse.
            o => o.map(w, h, x, y),
        }
    }

    /// The rectangle a (non-empty) rectangle of pixels maps onto.
    fn map_rect(self, w: i32, h: i32, r: Rect) -> Rect {
        let (ax, ay) = self.map(w, h, r.x0, r.y0);
        let (bx, by) = self.map(w, h, r.x1 - 1, r.y1 - 1);
        Rect::new(ax.min(bx), ay.min(by), ax.max(bx) + 1, ay.max(by) + 1)
    }
}

impl Surface {
    /// This surface turned or flipped by `o` about a `w × h` canvas, pixel for pixel (pixels
    /// outside the canvas move with it). Target tiles left holding only the default pixel are
    /// not stored, so the result stays as sparse as the source.
    pub fn oriented(&self, o: Orient, w: i32, h: i32) -> Surface {
        let bpp = self.format.bytes_per_pixel();
        let ts = TILE_SIZE as usize;
        let dp: &[u8] = &self.default_pixel;
        // Only source tiles with content make target tiles: a large area written with the default
        // pixel (a cleared layer) can leave many allocated blank tiles, which would each cost a
        // target tile's allocation and scan for nothing. Gaps read as the default pixel anyway.
        let sources: Vec<TileCoord> = self.tiles.par_iter().filter(|(_, t)| t.data.chunks_exact(bpp).any(|px| px != dp)).map(|(c, _)| *c).collect();
        let targets: Vec<TileCoord> = sources.iter().flat_map(|c| o.map_rect(w, h, c.rect()).tiles()).collect::<BTreeSet<_>>().into_iter().collect();
        let tiles: Vec<(TileCoord, Arc<Tile>)> = targets
            .par_iter()
            .filter_map(|&dc| {
                let origin = dc.rect();
                let mut data = vec![0u8; ts * ts * bpp];
                // Consecutive target pixels nearly always come from the same source tile.
                let mut cached: Option<(TileCoord, Option<&Arc<Tile>>)> = None;
                let mut content = false;
                for (ly, row) in data.chunks_exact_mut(ts * bpp).enumerate() {
                    let y = origin.y0 + ly as i32;
                    for (lx, out) in row.chunks_exact_mut(bpp).enumerate() {
                        let (sx, sy) = o.inverse(w, h, origin.x0 + lx as i32, y);
                        let sc = TileCoord::containing(sx, sy);
                        let src = match cached {
                            Some((c, t)) if c == sc => t,
                            _ => {
                                let t = self.tiles.get(&sc);
                                cached = Some((sc, t));
                                t
                            }
                        };
                        match src {
                            Some(t) => {
                                let i = (sy.rem_euclid(TILE_SIZE) as usize * ts + sx.rem_euclid(TILE_SIZE) as usize) * bpp;
                                let px = &t.data[i..i + bpp];
                                out.copy_from_slice(px);
                                content |= px != dp;
                            }
                            None => out.copy_from_slice(dp),
                        }
                    }
                }
                content.then(|| (dc, Arc::new(Tile { data: data.into_boxed_slice() })))
            })
            .collect();
        Surface { format: self.format, default_pixel: self.default_pixel.clone(), tiles: tiles.into_iter().collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, PixelFormat, SampleType};

    /// The plain per-pixel remap this replaces (`photocraft_engine::pixels::remap_surface`).
    fn naive(s: &Surface, o: Orient, w: i32, h: i32) -> Surface {
        let r = s.content_bounds();
        let mut out = Surface::with_default(s.format(), &s.default_pixel());
        let mut px = vec![0.0; s.channels()];
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                s.read_pixel(x, y, &mut px);
                let (nx, ny) = o.map(w, h, x, y);
                out.write_pixel(nx, ny, &px);
            }
        }
        out
    }

    fn assert_same(a: &Surface, b: &Surface, area: Rect) {
        let (mut pa, mut pb) = (vec![0.0; a.channels()], vec![0.0; b.channels()]);
        for y in area.y0..area.y1 {
            for x in area.x0..area.x1 {
                a.read_pixel(x, y, &mut pa);
                b.read_pixel(x, y, &mut pb);
                assert_eq!(pa, pb, "pixel ({x}, {y})");
            }
        }
    }

    fn sample(format: PixelFormat, default: &[f32]) -> Surface {
        let mut s = Surface::with_default(format, default);
        let n = s.channels();
        // Content spanning tile seams, off the canvas on the left/top and past it on the right.
        for (x, y) in [(-3, -2), (0, 0), (255, 255), (256, 1), (300, 7), (511, 260), (612, 299)] {
            let px: Vec<f32> = (0..n).map(|c| ((x * 7 + y * 13 + c as i32 * 29).rem_euclid(255)) as f32 / 255.0).collect();
            s.write_pixel(x, y, &px);
        }
        let block: Vec<f32> = (0..40 * 30 * n).map(|i| (i % 251) as f32 / 255.0).collect();
        s.write_region(Rect::new(240, 240, 280, 270), &block);
        s
    }

    #[test]
    fn matches_the_per_pixel_remap_for_every_turn() {
        let formats =
            [(PixelFormat::RGBA8, vec![0.0; 4]), (PixelFormat::new(ColorMode::Rgb, SampleType::U16, true), vec![0.0; 4]), (PixelFormat::GRAY8, vec![1.0])];
        // Odd sizes that aren't tile multiples, so target tiles straddle source seams.
        let (w, h) = (601, 300);
        for (format, default) in formats {
            let s = sample(format, &default);
            for o in [Orient::FlipHorizontal, Orient::FlipVertical, Orient::Rotate180, Orient::Cw90, Orient::Ccw90] {
                let fast = s.oriented(o, w, h);
                let slow = naive(&s, o, w, h);
                assert_same(&fast, &slow, Rect::new(-300, -300, 1000, 1000));
                assert_eq!(fast.content_bounds(), slow.content_bounds(), "{o:?}");
            }
        }
    }

    #[test]
    fn inverse_undoes_map() {
        let (w, h) = (37, 11);
        for o in [Orient::FlipHorizontal, Orient::FlipVertical, Orient::Rotate180, Orient::Cw90, Orient::Ccw90] {
            for (x, y) in [(0, 0), (36, 10), (5, 3), (-4, 20)] {
                let (mx, my) = o.map(w, h, x, y);
                assert_eq!(o.inverse(w, h, mx, my), (x, y), "{o:?}");
            }
        }
    }

    #[test]
    fn empty_and_default_tiles_stay_sparse() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        assert_eq!(s.oriented(Orient::FlipHorizontal, 100, 100).tile_count(), 0);
        // A tile allocated but holding only the default pixel is not carried over.
        let _ = s.tile_mut(TileCoord::new(0, 0));
        assert_eq!(s.oriented(Orient::Cw90, 100, 100).tile_count(), 0);
        // Among many blank tiles, only the one with content makes a target tile.
        for tx in 1..40 {
            let _ = s.tile_mut(TileCoord::new(tx, 3));
        }
        s.write_pixel(5, 5, &[1.0, 0.0, 0.0, 1.0]);
        let flipped = s.oriented(Orient::FlipHorizontal, 100, 100);
        assert_eq!(flipped.tile_count(), 1);
        assert_eq!(flipped.pixel(94, 5), vec![1.0, 0.0, 0.0, 1.0]);
    }
}
