use std::collections::HashMap;
use std::f64::consts::PI;
use ndarray::{Array2};
use numpy::{PyArray, ToPyArray};
use pyo3::prelude::*;
use pyo3::exceptions;
use rand;
use rand::seq::SliceRandom;
use rasterize::{BBox, Color, FillRule, Image, Layer, LinColor, LineCap, LineJoin, Path, PathBuilder, Point, SignedDifferenceRasterizer, StrokeStyle, Transform};
use freetype::{Library, face::LoadFlag, outline::Curve, Vector, GlyphSlot};
use rand::Rng;

fn to_point(p: &Vector) -> Point {
    Point::new(p.x as f64, p.y as f64)
}

#[derive(Debug)]
enum Segment {
    Line { p0: Point },
    Bezier2 { p0: Point, p1: Point },
    Bezier3 { p0: Point, p1: Point, p2: Point },
}

#[derive(Debug)]
struct Contour {
    start: Point,
    segments: Vec<Segment>,
}

/// A vector graphics representation of a glyph
#[derive(Debug)]
struct Glyph {
    contours: Vec<Contour>,
    width: f64,
    height: f64,
    /// Horizontal Bearing X
    hbx: f64,
    /// Horizontal Bearing Y
    hby: f64,
}

#[derive(Debug)]
struct FontFace {
    path: String,
    em_size: f64,
    glyphs: HashMap<char, Glyph>,
}

impl From<Curve> for Segment {
    fn from(curve: Curve) -> Self {
        match curve {
            Curve::Line(p0) => {
                Segment::Line{ p0: to_point(&p0) }
            },
            Curve::Bezier2(p0, p1) => {
                Segment::Bezier2{ p0: to_point(&p0), p1: to_point(&p1) }
            },
            Curve::Bezier3(p0, p1, p2) => {
                Segment::Bezier3{ p0: to_point(&p0), p1: to_point(&p1), p2: to_point(&p2) }
            },
        }
    }
}

impl From<&GlyphSlot> for Glyph {
    fn from(gs: &GlyphSlot) -> Self {
        let metrics = gs.metrics();
        let contours = gs.outline().unwrap().contours_iter().map(|contour| {
            Contour {
                start: to_point(contour.start()),
                segments: contour.map(|curve| {Segment::from(curve)}).collect()
            }
        }).collect();
        Glyph {
            contours,
            width: metrics.width as f64,
            height: metrics.height as f64,
            hbx: metrics.horiBearingX as f64,
            hby: metrics.horiBearingY as f64,
        }
    }
}

impl Glyph {
    fn to_path(&self) -> Path {
        let mut pb = PathBuilder::new();
        for contour in self.contours.iter() {
            pb.move_to(contour.start);
            for seg in contour.segments.iter() {
                match seg {
                    Segment::Line{p0} => { pb.line_to(p0); },
                    Segment::Bezier2{p0, p1} => { pb.quad_to(p0, p1); },
                    Segment::Bezier3{p0, p1, p2} => { pb.cubic_to(p0, p1, p2); },
                }
            }
        }
        pb.build()
    }
}

impl FontFace {
    /// Returns a reference to the glyph corresponding to `c`.
    /// If the glyph wasn't already cached it is added to the cache.
    fn get_glyph(&mut self, c: &char) -> &Glyph {
        self.glyphs.entry(*c).or_insert( {
            let lib = Library::init().unwrap();
            let face = lib.new_face(&self.path, 0).unwrap();
            face.load_char(*c as usize, LoadFlag::NO_SCALE).unwrap();
            Glyph::from(face.glyph())
        })
    }

    fn get_cached_glyph(&self, c: &char) -> &Glyph {
        &self.glyphs[c]
    }

    /// Populates the cache with the glyphs corresponding to `chars`.
    fn ensure_in_cache(&mut self, chars: &[char]) {
        let lib = Library::init().unwrap();
        let face = lib.new_face(&self.path, 0).unwrap();
        for c in chars {
            if !self.glyphs.contains_key(c)
            {
                face.load_char(c.clone() as usize, LoadFlag::NO_SCALE).unwrap();
                self.glyphs.insert(c.clone(), Glyph::from(face.glyph()));
            }
        }
    }
}

struct TrainingSample {
    img: Array2<f32>,
    kanji_index: i32,
    font_size: i32,
    angle_category: i32,
}

/// A class for efficiently generating training data.
#[pyclass]
struct RustRenderer {
    fonts: Vec<FontFace>,
    /// Images are always square
    img_size: u32,
    /// The current set of kanji used to generate samples
    kanji: Vec<char>,
    /// The master table mapping kanji to categories
    category_map: HashMap<char, i32>,
    /// Number of categories to use for the angle
    angle_categories: i32,
}

impl RustRenderer {
    fn render_raw(&self) -> TrainingSample {
        let mut rng = rand::thread_rng();
        let char = self.kanji.choose(&mut rng).unwrap();
        let kanji_index = self.category_map[char];
        let font_size = rng.gen_range(16.0..24.0);
        let angle_category = 0;
        let n_angles = self.angle_categories as f64;
        let angle_mid = 2.0 * PI * (angle_category as f64) / n_angles;
        let angle_min = angle_mid - (PI / n_angles);
        let angle_max = angle_mid + (PI / n_angles);
        let angle = rng.gen_range(angle_min..angle_max);

        let rasterizer = SignedDifferenceRasterizer::default();
        let fill_rule = FillRule::default();
        let black = LinColor::new(0.0, 0.0, 0.0, 1.0);
        let white = LinColor::new(1.0, 1.0, 1.0, 1.0);
        let size64 = self.img_size as f64;

        let mut img = Layer::new(
            BBox::new((0.0, 0.0), (size64, size64)),
            Some(white),
        );

        //println!("image size: {}, {}", img.width(), img.height());
        //println!("img.data().len(): {}", img.data().len());
        //println!("row_offset: {}", img.shape().offset(img.height()-1, 0));

        let transform = Transform::new_scale(size64, size64);

        // draw a whole bunch of random shapes to add noise to the image.
        for _ in 0..rng.gen_range(1..20) {
            let p1 = Point::new(rng.gen(), rng.gen());
            let color = black.with_alpha(rng.gen());
            let mut pb = PathBuilder::new();
            pb.move_to(p1);
            let stroke: bool;
            match rng.gen_range::<u8, _>(0..3) {
                // line
                0 => {
                    let p2 = Point::new(rng.gen(), rng.gen());
                    pb.line_to(p2);
                    stroke = true;
                },
                // circle
                1 => {
                    let radius = rng.gen::<f64>() * 0.5;
                    pb.circle(radius);
                    stroke = rng.gen();
                },
                // polygon
                2 => {
                    for _ in 0..3 {
                        let p = Point::new(rng.gen(), rng.gen());
                        pb.line_to(p);
                    }
                    pb.line_to(p1);
                    stroke = rng.gen();
                },
                _ => {
                    stroke = false;
                },
            }
            let mut path = pb.build();
            if stroke {
                let style = StrokeStyle {
                    width: rng.gen_range(0.05..0.2),
                    line_join: LineJoin::Bevel,
                    line_cap: LineCap::Butt
                };
                path = path.stroke(style);
            }
            img = path.fill(&rasterizer,transform, fill_rule,color, img);
        }

        let jitter_x = rng.gen_range(-0.1..0.1);
        let jitter_y = rng.gen_range(-0.1..0.1);

        let transform = transform
            .pre_translate(jitter_x, jitter_y)
            .pre_translate(0.5, 0.5)
            .pre_rotate(angle)
            .pre_translate(-0.5, -0.5);

        // Draw a semi-transparent white circle to improve the contrast of our text
        let radius = rng.gen_range((font_size/(size64*2.0))..0.7);
        let color = white.with_alpha(rng.gen_range(0.5..1.0));
        let mut pb = PathBuilder::new();
        pb.move_to((0.5, 0.5));
        pb.circle(radius);
        let path = pb.build();
        img = path.fill(
            &rasterizer,
            transform,
            fill_rule,
            color,
            img
        );

        // Draw the glyph
        let font = self.fonts.choose(&mut rng).unwrap();
        let font_scale = font_size / (size64 * font.em_size);
        let glyph= font.get_cached_glyph(char);

        //println!("width, height: {}, {}", glyph.width, glyph.height);
        //println!("HBX, HBY: {}, {}", glyph.hbx, glyph.hby);
        let offset_x = (1.0 - (glyph.width * font_scale)) / 2.0 - glyph.hbx * font_scale;
        let offset_y = (1.0 - (glyph.height * font_scale)) / 2.0 + (glyph.height - glyph.hby) * font_scale;

        // The transform steps are listed in reverse order
        let transform = transform
            .pre_translate(offset_x, 1.0 - offset_y)
            .pre_scale(font_scale, -font_scale);
        let color = black.with_alpha(rng.gen_range(0.5..1.0));

        let path = glyph.to_path();
        img = path.fill(&rasterizer, transform, fill_rule, color, img);

        let shape = (self.img_size as usize, self.img_size as usize);
        let pixels = Array2::from_shape_vec(
            shape, img.iter().map(|px| px.red() as f32).collect()
        ).unwrap();

        TrainingSample {
            img: pixels,
            kanji_index,
            font_size: (font_size as i32),
            angle_category
        }
    }
}

#[pymethods]
impl RustRenderer {
    #[new]
    fn new(img_size: u32, font_paths: Vec<String>, all_kanji: Vec<char>, angle_categories: i32) -> PyResult<Self> {
        if font_paths.len() == 0 {
            return Err(exceptions::PyValueError::new_err(
                format!("Font list was empty. len(font_paths) must be greater than zero.")
            ));
        }
        if all_kanji.len() == 0 {
            return Err(exceptions::PyValueError::new_err(
                format!("Kanji list was empty. len(all_kanji) must be greater than zero.")
            ));
        }

        // Check that all our fonts are valid.
        // It's better to throw an exception as soon as we receive the invalid path(s) from Python
        let mut fonts = vec![];
        let lib = Library::init().unwrap();
        for path in font_paths {
            println!("Trying font at: {}", &path);
            match lib.new_face(&path, 0) {
                Err(_) => {
                    return Err(exceptions::PyValueError::new_err(
                        format!("Failed to load the font at: {}", &path)
                    ))
                },
                Ok(face) => {
                    fonts.push(FontFace {
                        path,
                        em_size: face.em_size() as f64,
                        glyphs: HashMap::new(),
                    });
                }
            }
        }

        // This is optional, but we might as well pre-populate the cache since we're going to need
        // it sooner or later.
        for font in fonts.iter_mut() {
            font.ensure_in_cache(all_kanji.as_slice());
        }

        let category_map = all_kanji.iter().enumerate().map(|(ii, kk)| {
            (*kk, ii as i32)
        }).collect();

        Ok(RustRenderer {
            fonts,
            img_size,
            kanji: all_kanji,
            category_map,
            angle_categories,
        })
    }

    #[getter(kanji)]
    fn get_kanji(&self) -> PyResult<Vec<char>> {
        Ok(self.kanji.clone())
    }

    #[setter(kanji)]
    fn set_kanji(&mut self, char_list: Vec<char>) -> PyResult<()> {
        if char_list.len() == 0 {
            return Err(exceptions::PyValueError::new_err(
                format!("Can't set kanji to an empty list.")
            ));
        }
        self.kanji = char_list;
        Ok(())
    }

    /// Generate a random training sample
    /// Returns a tuple of (img, kanji_index, font_size, angle_index)
    ///     img: a 2D numpy array of floating point pixel values
    ///     kanji_index: an integer giving the category label of the kanji
    ///     font_size: the font size as an integer
    ///     angle_index: the category label for the angle
    fn render<'a>(&self, py: Python<'a>) -> PyResult<(&'a PyArray<f32, ndarray::Dim<[usize; 2]>>, i32, i32, i32)> {
        let sample = py.allow_threads(|| {
            self.render_raw()
        });
        Ok((
            sample.img.to_pyarray(py),
            sample.kanji_index,
            sample.font_size,
            sample.angle_category
        ))
    }

    /// The same as render(), but returns a list of `batch_size` training samples.
    fn render_batch<'a>(&self, py: Python<'a>, batch_size: i32) -> PyResult<Vec<(&'a PyArray<f32, ndarray::Dim<[usize; 2]>>, i32, i32, i32)>> {
        let samples = py.allow_threads(|| {
            (0..batch_size).into_iter().map(|_| {
                self.render_raw()
            })
        }).map(|sample| {
            (
                sample.img.to_pyarray(py),
                sample.kanji_index,
                sample.font_size,
                sample.angle_category
            )
        }).collect();
        Ok(samples)
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn datagen(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustRenderer>()?;
    Ok(())
}
