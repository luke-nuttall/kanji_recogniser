extern crate andrew;

use ndarray::{Array1, Array2, Array3, Axis};
use numpy::{PyArray, ToPyArray};
use pyo3::exceptions;
use pyo3::prelude::*;
use glutin::{ContextBuilder, event_loop::EventLoop};
use femtovg::{renderer::OpenGl, Canvas, Path, Paint, Color, Baseline, Align};
use glutin::dpi::PhysicalSize;

/// Returns a 2D numpy array containing an image.
#[pyfunction]
fn render_gl(py: Python) -> PyResult<&PyArray<u8, ndarray::Dim<[usize; 2]>>> {
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

    canvas.flush();

    let image = canvas.screenshot().unwrap();
    let pixels = Array1::from_iter(image.pixels().map(|pixel| pixel.r));
    let result = pixels.to_shape((width as usize, height as usize)).unwrap();
    Ok(result.to_pyarray(py))
}

/// Returns a 2D numpy array containing an image.
#[pyfunction]
fn render_andy(py: Python) -> PyResult<&PyArray<u8, ndarray::Dim<[usize; 2]>>> {
    let width = 48;
    let height = 48;
    let mut buf: Vec<u8> = vec![255; 4 * width * height];
    let mut canvas = andrew::Canvas::new(&mut buf, width, height, 4 * width, andrew::Endian::native());
    let (block_w, block_h) = (width / 8, height / 8);
    for block_y in 0..9 {
        for block_x in 0..9 {
            let color = if (block_x + (block_y % 2)) % 2 == 0 {
                [255, 0, 100, 200]
            } else {
                [255, 255, 255, 255]
            };

            let block = andrew::shapes::rectangle::Rectangle::new(
                (block_w * block_x, block_h * block_y),
                (block_w, block_h),
                None,
                Some(color),
            );
            canvas.draw(&block);
        }
    }

    let font_data = andrew::text::load_font_file("fonts/NotoSansJP-Medium.otf");
    canvas.draw(&andrew::text::Text::new(
        (0, 0),
        [255, 0, 0, 0],
        &font_data,
        12.0,
        1.0,
        "abcdefghi",
    ));
    canvas.draw(&andrew::text::Text::new(
        (0, 12),
        [255, 0, 0, 0],
        &font_data,
        12.0,
        1.0,
        "jklmnopqr",
    ));
    canvas.draw(&andrew::text::Text::new(
        (0, 24),
        [255, 0, 0, 0],
        &font_data,
        12.0,
        1.0,
        "stuvwxyz",
    ));

    let image = match Array3::from_shape_vec((width, height, 4), buf) {
        Ok(result) => result,
        Err(_) => return Err(PyErr::new::<exceptions::PyException, _>("Internal error converting pixel buffer to array.")),
    };
    let result = image.index_axis(Axis(2), 0);
    Ok(result.to_pyarray(py))
}

/// Returns a blank array.
#[pyfunction]
fn get_blank_array(py: Python) -> PyResult<&PyArray<u8, ndarray::Dim<[usize; 2]>>> {
    let results = Array2::zeros((5, 5));
    Ok(results.to_pyarray(py))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn datagen(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_blank_array, m)?)?;
    m.add_function(wrap_pyfunction!(render_andy, m)?)?;
    m.add_function(wrap_pyfunction!(render_gl, m)?)?;
    Ok(())
}
