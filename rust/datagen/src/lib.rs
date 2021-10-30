use ndarray::{Array1};
use numpy::{PyArray, ToPyArray};
use pyo3::prelude::*;
use glutin::{ContextBuilder, event_loop::EventLoop, dpi::PhysicalSize, Context, PossiblyCurrent};
use femtovg::{renderer::OpenGl, Canvas, Path, Paint, Color, Baseline, Align, FontId};
use rand;
use rand::Rng;

/// A class for efficiently generating training data.
#[pyclass(unsendable)]
struct RustRenderer {
    gl_context: Context<PossiblyCurrent>,
    canvas: Canvas<OpenGl>,
    fonts: Vec<FontId>,
    img_size: u32,
}

#[pymethods]
impl RustRenderer {
    #[new]
    fn new(img_size: u32, font_paths: Vec<String>) -> Self {
        assert!(font_paths.len() > 0);
        // Create our OpenGL context using glutin.
        // It would be really nice if we could create this context using the CPU's integrated GPU
        // instead of using the main dedicated GPU where the model training will be happening.
        let event_loop = EventLoop::new();
        let gl_context = ContextBuilder::new()
            .with_vsync(false)
            .with_multisampling(16)
            .build_headless(&event_loop, PhysicalSize::new(img_size, img_size)).unwrap();
        let gl_context = unsafe { gl_context.make_current().unwrap() };

        // Set up our femtovg canvas
        let renderer = OpenGl::new(|s| gl_context.get_proc_address(s) as *const _).unwrap();
        let mut canvas = Canvas::new(renderer).unwrap();
        canvas.set_size(img_size, img_size, 1.0);  // dpi doesn't matter so we set it to 1.0

        // Initialize fonts
        let fonts = font_paths.iter().map(|path| canvas.add_font(path).unwrap()).collect();

        RustRenderer {
            gl_context,
            canvas,
            fonts,
            img_size
        }
    }

    fn render<'a>(&mut self, py: Python<'a>, kanji: String, font_size: f32, angle: f32) -> PyResult<&'a PyArray<f32, ndarray::Dim<[usize; 2]>>> {
        let mut rng = rand::thread_rng();
        self.canvas.reset();
        self.canvas.clear_rect(0, 0, self.img_size, self.img_size, Color::rgb(255, 255, 255));

        let sizef = self.img_size as f32;

        // set up transforms so that we're working in a more sensible coordinate space
        // (0,0) is the middle of our image
        self.canvas.translate(sizef / 2.0, sizef / 2.0);

        // draw a whole bunch of random shapes to add noise to the image.
        for _ in 0..rng.gen_range(1..100) {
            let x = (rng.gen::<f32>() - 0.5) * sizef * 2.0;
            let y = (rng.gen::<f32>() - 0.5) * sizef * 2.0;
            let sx = (rng.gen::<f32>() - 0.5) * sizef * 2.0;
            let sy = (rng.gen::<f32>() - 0.5) * sizef * 2.0;
            let mut paint = Paint::color(Color::rgba(0, 0, 0, rng.gen()));
            paint.set_line_width(rng.gen_range(1.0..5.0));
            let mut path = Path::new();
            match rng.gen_range::<u8, _>(0..5) {
                0 => {
                    path.move_to(x, y);
                    path.line_to(sx, sy);
                    self.canvas.stroke_path(&mut path, paint);
                },
                1 => {
                    path.ellipse(x, y, sx, sy);
                    self.canvas.stroke_path(&mut path, paint);
                },
                2 => {
                    path.ellipse(x, y, sx, sy);
                    self.canvas.fill_path(&mut path, paint);
                },
                3 => {
                    path.rect(x, y, sx, sy);
                    self.canvas.stroke_path(&mut path, paint);
                },
                4 => {
                    path.rect(x, y, sx, sy);
                    self.canvas.fill_path(&mut path, paint);
                },
                _ => {},
            }
        }

        // draw a circle in 50% white so we're sure the kanji will be at least somewhat visible
        let mut path = Path::new();
        let radius = rng.gen_range(font_size..sizef) / 2.0;
        let paint = Paint::color(Color::rgba(255, 255, 255, rng.gen_range(128..=255)));
        path.circle(0.0, 0.0, radius);
        self.canvas.fill_path(&mut path, paint);

        // rotate only the rendered text
        // we don't want the model learning to estimate the rotation by looking at the background clutter
        self.canvas.rotate(angle);

        let font_index: usize = rng.gen_range(0..self.fonts.len());
        let mut paint = Paint::color(Color::black());
        paint.set_font(&[self.fonts[font_index]]);
        paint.set_font_size(font_size);
        paint.set_text_baseline(Baseline::Middle);
        paint.set_text_align(Align::Center);
        self.canvas.fill_text(0.0, 0.0, kanji, paint).unwrap();

        // finish sending draw commands and render everything
        self.canvas.flush();

        let image = self.canvas.screenshot().unwrap();
        let pixels = Array1::from_iter(image.pixels().map(|pixel| pixel.r as f32 / 255.0));
        let result = pixels.to_shape((self.img_size as usize, self.img_size as usize)).unwrap();
        Ok(result.to_pyarray(py))
    }
}



/// Returns a 2D numpy array containing an image.
#[pyfunction]
fn render_gl(py: Python) -> PyResult<&PyArray<f32, ndarray::Dim<[usize; 2]>>> {
    let width: u32 = 256;
    let height: u32 = 256;

    // Create our OpenGL context using glutin.
    let el = EventLoop::new();
    let context = ContextBuilder::new()
        .with_vsync(false)
        .with_multisampling(16)
        .build_headless(&el, PhysicalSize::new(width, height)).unwrap();
    let context = unsafe { context.make_current().unwrap() };

    // Set up our femtovg canvas
    let renderer = OpenGl::new(|s| context.get_proc_address(s) as *const _).unwrap();
    let mut canvas = Canvas::new(renderer).unwrap();
    canvas.set_size(width, height,1.0);  // dpi doesn't matter so we set it to 1.0

    // Initialize fonts
    let font_id = canvas.add_font("fonts/NotoSansJP-Bold.otf").unwrap();

    // ----- DRAWING STARTS HERE -----
    canvas.clear_rect(0, 0, width, height, Color::rgb(255, 255, 255));

    // set up transforms so that we're working in a more sensible coordinate space
    // (0,0) is the middle of our image
    // also apply any rotation at this stage
    canvas.translate(width as f32 / 2.0, height as f32 / 2.0);
    canvas.rotate(std::f32::consts::PI / 4.0);

    let mut path = Path::new();
    path.circle(0.0, 0.0, 116.0);
    canvas.fill_path(&mut path, Paint::color(Color::rgba(200, 200, 200, 255)));
    let mut line = Paint::color(Color::rgba(32, 32, 32, 255));
    line.set_line_width(5.0);
    canvas.stroke_path(&mut path, line);

    let mut paint = Paint::color(Color::black());
    paint.set_font(&[font_id]);
    paint.set_font_size(100.0);
    paint.set_text_baseline(Baseline::Middle);
    paint.set_text_align(Align::Center);
    canvas.fill_text(0.0, 0.0, "漢字", paint).unwrap();

    // finish sending draw commands and render everything
    canvas.flush();

    let image = canvas.screenshot().unwrap();
    let pixels = Array1::from_iter(image.pixels().map(|pixel| pixel.r as f32 / 255.0));
    let result = pixels.to_shape((width as usize, height as usize)).unwrap();
    Ok(result.to_pyarray(py))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn datagen(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustRenderer>()?;
    m.add_function(wrap_pyfunction!(render_gl, m)?)?;
    Ok(())
}
