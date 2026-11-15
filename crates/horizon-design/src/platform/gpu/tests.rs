mod readback;

use super::*;
use readback::{PixelReadback, ReadbackLayout};

// These methods exist only in test builds. The live renderer does not expose
// a pixel-download API after removal of the snapshot feature.
impl Gpu {
    /// Downloads rendered pixels for color, alpha, and row-layout assertions.
    fn read_pixels(
        &mut self,
        scene: &Scene,
        width: u32,
        height: u32,
        background: Color,
    ) -> Result<Vec<u8>, String> {
        self.begin_read_pixels(scene, width, height, background)?
            .wait(&self.device, std::time::Duration::from_secs(5))
    }

    /// Starts a download for tests of nonblocking GPU mapping.
    fn begin_read_pixels(
        &mut self,
        scene: &Scene,
        width: u32,
        height: u32,
        background: Color,
    ) -> Result<PixelReadback, String> {
        self.check_device()?;

        let layout = ReadbackLayout::new(&self.device.limits(), width, height)?;
        let texture = self.render_texture(width, height, wgpu::TextureUsages::COPY_SRC);
        let render_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.render_scene(scene, &render_view, width, height, background)?;

        Ok(PixelReadback::copy(
            &self.device,
            &self.queue,
            &texture,
            layout,
        ))
    }
}

fn transparent_capabilities() -> wgpu::SurfaceCapabilities {
    wgpu::SurfaceCapabilities {
        formats: vec![
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
        ],
        present_modes: vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox],
        alpha_modes: vec![
            wgpu::CompositeAlphaMode::PostMultiplied,
            wgpu::CompositeAlphaMode::PreMultiplied,
        ],
        ..Default::default()
    }
}

#[test]
fn selects_transparent_non_srgb_configuration() {
    let mut capabilities = transparent_capabilities();
    let configuration =
        surface_configuration(&capabilities, &wgpu::Limits::default(), 640, 480).unwrap();
    assert_eq!(configuration.format, wgpu::TextureFormat::Bgra8Unorm);
    assert_eq!(
        configuration.alpha_mode,
        wgpu::CompositeAlphaMode::PreMultiplied
    );
    assert_eq!(configuration.present_mode, wgpu::PresentMode::Mailbox);
    capabilities.alpha_modes = vec![wgpu::CompositeAlphaMode::PostMultiplied];
    capabilities.present_modes = vec![wgpu::PresentMode::Fifo];
    let configuration =
        surface_configuration(&capabilities, &wgpu::Limits::default(), 640, 480).unwrap();
    assert_eq!(
        configuration.alpha_mode,
        wgpu::CompositeAlphaMode::PostMultiplied
    );
    assert_eq!(configuration.present_mode, wgpu::PresentMode::Fifo);
}

#[test]
fn rejects_opaque_incompatible_and_invalid_surface_configurations() {
    let limits = wgpu::Limits::default();
    for alpha_mode in [
        wgpu::CompositeAlphaMode::Opaque,
        wgpu::CompositeAlphaMode::Auto,
        wgpu::CompositeAlphaMode::Inherit,
    ] {
        let mut capabilities = transparent_capabilities();
        capabilities.alpha_modes = vec![alpha_mode];
        assert!(surface_configuration(&capabilities, &limits, 640, 480).is_err());
    }
    let mut capabilities = transparent_capabilities();
    capabilities.formats.clear();
    assert!(surface_configuration(&capabilities, &limits, 640, 480).is_err());
    capabilities.formats = vec![wgpu::TextureFormat::Rgba8UnormSrgb];
    assert!(surface_configuration(&capabilities, &limits, 640, 480).is_err());
    capabilities = transparent_capabilities();
    capabilities.present_modes.clear();
    assert!(surface_configuration(&capabilities, &limits, 640, 480).is_err());
    for (width, height) in [(0, 480), (640, 0), (limits.max_texture_dimension_2d + 1, 1)] {
        assert!(
            surface_configuration(&transparent_capabilities(), &limits, width, height).is_err()
        );
    }
}

#[test]
fn distinguishes_retry_reconfigure_and_recreation_outcomes() {
    for (status, expected) in [
        (
            wgpu::CurrentSurfaceTexture::Timeout,
            RenderOutcome::TimedOut,
        ),
        (
            wgpu::CurrentSurfaceTexture::Occluded,
            RenderOutcome::Occluded,
        ),
        (
            wgpu::CurrentSurfaceTexture::Outdated,
            RenderOutcome::Reconfigured,
        ),
        (
            wgpu::CurrentSurfaceTexture::Lost,
            RenderOutcome::RecreateSurface,
        ),
    ] {
        let FrameAcquisition::Skipped(outcome) = acquire_frame(status, false).unwrap() else {
            panic!("unavailable surface must not produce a frame");
        };
        assert_eq!(outcome, expected);
    }
    assert!(acquire_frame(wgpu::CurrentSurfaceTexture::Validation, false).is_err());
    for status in [
        wgpu::CurrentSurfaceTexture::Lost,
        wgpu::CurrentSurfaceTexture::Outdated,
        wgpu::CurrentSurfaceTexture::Timeout,
    ] {
        assert!(matches!(
            acquire_frame(status, true).unwrap(),
            FrameAcquisition::Skipped(RenderOutcome::RecreateDevice)
        ));
    }
}

#[test]
fn presentation_shader_passes_wgsl_validation() {
    let shader = wgpu::naga::front::wgsl::parse_str(BLIT_SHADER).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&shader)
    .unwrap();
}

/// Runs on a physical adapter or Mesa's software Vulkan driver. Kept opt-in so
/// ordinary unit tests do not require a graphics driver or compile GPU shaders.
#[test]
#[ignore = "requires a Vulkan adapter (software Vulkan is sufficient)"]
fn gpu_readback_and_presentation_preserve_color_and_alpha() {
    use std::time::{Duration, Instant};
    use vello::{
        kurbo::{Affine, Rect},
        peniko::Fill,
    };

    let instance = wgpu_instance();
    let mut gpu = Gpu::new(&instance, None).expect("a Vulkan adapter is required for this test");
    let validation_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgba8(200, 100, 50, 128),
        None,
        &Rect::new(0.0, 0.0, 65.0, 3.0),
    );

    for width in [1, 63, 64, 65] {
        let pixels = gpu
            .read_pixels(&scene, width, 3, Color::TRANSPARENT)
            .unwrap();
        assert_eq!(pixels.len(), width as usize * 3 * 4);
        for pixel in pixels.as_chunks::<4>().0 {
            assert_pixel_near(pixel, &[200, 100, 50, 128]);
        }
    }
    let background = gpu
        .read_pixels(&Scene::new(), 2, 2, Color::from_rgb8(12, 34, 56))
        .unwrap();
    assert_eq!(background, [12, 34, 56, 255].repeat(4));
    assert_eq!(
        gpu.read_pixels(&Scene::new(), 1, 1, Color::TRANSPARENT)
            .unwrap(),
        [0, 0, 0, 0]
    );
    assert!(gpu.read_pixels(&scene, 0, 1, Color::TRANSPARENT).is_err());
    assert!(
        gpu.read_pixels(
            &scene,
            gpu.device.limits().max_texture_dimension_2d + 1,
            1,
            Color::TRANSPARENT
        )
        .is_err()
    );

    let mut readback = gpu
        .begin_read_pixels(&scene, 65, 3, Color::TRANSPARENT)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(pixels) = readback.try_read(&gpu.device).unwrap() {
            assert_eq!(pixels.len(), 65 * 3 * 4);
            assert_pixel_near(&pixels[..4], &[200, 100, 50, 128]);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "nonblocking readback did not complete"
        );
        std::thread::yield_now();
    }
    assert!(readback.try_read(&gpu.device).is_err());

    // Use actual GPU render passes to catch double premultiplication, channel
    // swaps, wrong fragment entry points, and fullscreen-triangle coverage.
    let (source_view, bind_group) = gpu.offscreen(65, 3);
    gpu.render_scene(&scene, &source_view, 65, 3, Color::TRANSPARENT)
        .unwrap();
    for format in [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8Unorm,
    ] {
        for premultiply_alpha in [false, true] {
            let pipeline = gpu.blit_pipeline(BlitPipelineKey {
                format,
                premultiply_alpha,
            });
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("presentation test"),
                size: wgpu::Extent3d {
                    width: 65,
                    height: 3,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let target_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encode_blit(&mut encoder, &target_view, &pipeline, &bind_group);
            gpu.queue.submit([encoder.finish()]);
            let layout = ReadbackLayout::new(&gpu.device.limits(), 65, 3).unwrap();
            let pixels = PixelReadback::copy(&gpu.device, &gpu.queue, &texture, layout)
                .wait(&gpu.device, Duration::from_secs(5))
                .unwrap();
            let mut expected = if premultiply_alpha {
                [100, 50, 25, 128]
            } else {
                [200, 100, 50, 128]
            };
            if format == wgpu::TextureFormat::Bgra8Unorm {
                expected.swap(0, 2);
            }
            for pixel in pixels.as_chunks::<4>().0 {
                assert_pixel_near(pixel, &expected);
            }
        }
    }
    assert_eq!(gpu.blit_pipelines.len(), 4);
    let existing = gpu.blit_pipeline(BlitPipelineKey {
        format: wgpu::TextureFormat::Rgba8Unorm,
        premultiply_alpha: true,
    });
    assert_eq!(gpu.blit_pipelines.len(), 4);
    assert_eq!(
        existing,
        gpu.blit_pipelines[&BlitPipelineKey {
            format: wgpu::TextureFormat::Rgba8Unorm,
            premultiply_alpha: true
        }]
    );
    assert!(pollster::block_on(validation_scope.pop()).is_none());

    gpu.device.destroy();
    gpu.device.poll(wgpu::PollType::Poll).unwrap();
    assert!(gpu.device_lost.load(Ordering::Acquire));
    assert!(
        gpu.begin_read_pixels(&scene, 1, 1, Color::TRANSPARENT)
            .is_err()
    );
}

fn assert_pixel_near(actual: &[u8], expected: &[u8; 4]) {
    for (actual_channel, expected_channel) in actual.iter().zip(expected) {
        assert!(
            actual_channel.abs_diff(*expected_channel) <= 1,
            "pixel {actual:?}, expected {expected:?}"
        );
    }
}

/// Reports wall-clock latency without imposing hardware-dependent thresholds.
/// Includes driver overhead and GPU completion, but excludes compositor pacing.
#[test]
#[ignore = "manual GPU performance check; requires a Vulkan adapter"]
fn profile_gpu_rendering_and_readback() {
    use std::time::{Duration, Instant};
    use vello::{
        kurbo::{Affine, Rect},
        peniko::Fill,
    };

    let setup_started = Instant::now();
    let instance = wgpu_instance();
    let mut gpu = Gpu::new(&instance, None).unwrap();
    let initialization = setup_started.elapsed();
    let validation_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let (width, height) = (1280, 720);
    let (render_view, bind_group) = gpu.offscreen(width, height);
    let pipeline = gpu.blit_pipeline(BlitPipelineKey {
        format: RENDER_FORMAT,
        premultiply_alpha: true,
    });
    let output = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("performance check"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RENDER_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let mut scene = Scene::new();
    for row in 0..8 {
        for column in 0..12 {
            let left = f64::from(column * 100);
            let top = f64::from(row * 80);
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::from_rgba8(80, 140, 220, 180),
                None,
                &Rect::new(left, top, left + 90.0, top + 70.0),
            );
        }
    }
    let mut samples = Vec::new();
    for iteration in 0..13 {
        let frame_started = Instant::now();
        gpu.render_scene(&scene, &render_view, width, height, Color::TRANSPARENT)
            .unwrap();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encode_blit(&mut encoder, &output_view, &pipeline, &bind_group);
        let submission = gpu.queue.submit([encoder.finish()]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(5)),
            })
            .unwrap();
        if iteration >= 3 {
            samples.push(frame_started.elapsed());
        }
    }
    samples.sort_unstable();
    let readback_started = Instant::now();
    let pixels = gpu
        .read_pixels(&scene, width, height, Color::TRANSPARENT)
        .unwrap();
    let readback_time = readback_started.elapsed();
    assert_eq!(pixels.len(), width as usize * height as usize * 4);
    assert!(pollster::block_on(validation_scope.pop()).is_none());
    eprintln!(
        "GPU: {}; initialization: {initialization:?}; 1280×720 render + blit median: {:?}; test render + readback: {readback_time:?}",
        gpu.description,
        samples[samples.len() / 2]
    );
}

/// Produces an inspectable specimen using the same controlled fonts as text tests.
#[test]
#[ignore = "visual typography check; requires a Vulkan adapter"]
fn typography_specimen() {
    use crate::{DrawOptions, TextStyle, render::text::tests::fixture_system};
    use horizon_theme::TextRole;
    use std::io::Write;
    use vello::kurbo::Point;

    let (mut text, typography) = fixture_system();
    let mut scene = Scene::new();
    let output_scale = 1.25;
    let mut y = 24.0;
    for role in TextRole::ALL {
        text.draw(
            &typography,
            &mut scene,
            role.name(),
            &TextStyle::default(),
            None,
            DrawOptions {
                origin: Point::new(24.0, y),
                output_scale,
                color: Color::from_rgb8(95, 95, 95),
                ..Default::default()
            },
        )
        .unwrap();
        let sample = if role == TextRole::Display {
            "Horizon 09:41"
        } else {
            "Horizon — Aa Bb 0123456789"
        };
        let style = TextStyle::new(role);
        let measured = text
            .measure(&typography, sample, &style, None, output_scale)
            .unwrap();
        let drawn = text
            .draw(
                &typography,
                &mut scene,
                sample,
                &style,
                None,
                DrawOptions {
                    origin: Point::new(150.0, y),
                    output_scale,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(drawn, measured);
        y += measured.height + 16.0;
    }
    for (label, sample, style, width) in [
        (
            "tabular digits",
            "09:41  11:11  88:88",
            TextStyle::default().feature("tnum", 1),
            None,
        ),
        (
            "wrapped",
            "Typography follows the theme. Measurement and drawing share a layout, including explicit newlines.\nA second paragraph.",
            TextStyle::default(),
            Some(350.0),
        ),
        (
            "fallback",
            "Horizon Καλημέρα Привет א",
            TextStyle::default().font(horizon_theme::FontDefinition::new([
                "Inter",
                "Inter Variable",
                "Liberation Mono",
            ])),
            None,
        ),
        (
            "overrides",
            "Italic • weight 550 • tracking",
            TextStyle::default()
                .slant(horizon_theme::FontSlant::Italic)
                .weight(550.0)
                .tracking(0.04),
            None,
        ),
    ] {
        text.draw(
            &typography,
            &mut scene,
            label,
            &TextStyle::default(),
            None,
            DrawOptions {
                origin: Point::new(24.0, y),
                output_scale,
                color: Color::from_rgb8(95, 95, 95),
                ..Default::default()
            },
        )
        .unwrap();
        let metrics = text
            .draw(
                &typography,
                &mut scene,
                sample,
                &style,
                width,
                DrawOptions {
                    origin: Point::new(150.0, y),
                    output_scale,
                    ..Default::default()
                },
            )
            .unwrap();
        y += metrics.height + 16.0;
    }
    for scale in [1.0, 1.25, 2.0] {
        text.set_text_scale(scale).unwrap();
        let metrics = text
            .draw(
                &typography,
                &mut scene,
                &format!("Text scale {scale}×"),
                &TextStyle::default(),
                None,
                DrawOptions {
                    origin: Point::new(150.0, y),
                    output_scale,
                    ..Default::default()
                },
            )
            .unwrap();
        y += metrics.height + 16.0;
    }
    let width = 1200;
    let height = ((y + 24.0) * output_scale).ceil() as u32;
    let instance = wgpu_instance();
    let mut gpu = Gpu::new(&instance, None).expect("a Vulkan adapter is required for this test");
    let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pixels = gpu
        .read_pixels(&scene, width, height, Color::WHITE)
        .unwrap();
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .any(|p| p[0] < 100 && p[1] < 100 && p[2] < 100)
    );
    assert!(pollster::block_on(validation.pop()).is_none());
    let path = std::env::temp_dir().join("horizon-typography.ppm");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    write!(file, "P6\n{width} {height}\n255\n").unwrap();
    for pixel in pixels.as_chunks::<4>().0 {
        file.write_all(&pixel[..3]).unwrap();
    }
    file.flush().unwrap();
    eprintln!("typography specimen: {}", path.display());
}
