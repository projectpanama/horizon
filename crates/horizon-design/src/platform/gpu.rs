mod cache;

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
};
use vello::{
    AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene, peniko::Color, wgpu,
};

use cache::PipelineCache;

const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const BLIT_SHADER: &str = r"
@group(0) @binding(0) var source_texture: texture_2d<f32>;

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle covers the target without an internal seam.
    let position = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    return vec4<f32>(position * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

@fragment
fn premultiplied_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let color = textureLoad(source_texture, vec2<i32>(position.xy), 0);
    return vec4<f32>(color.rgb * color.a, color.a);
}

@fragment
fn straight_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(source_texture, vec2<i32>(position.xy), 0);
}
";

/// Creates the instance used to select adapters and create surfaces.
pub(crate) fn wgpu_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        display: None,
        backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::from_env_or_default(),
    })
}

/// Creates a surface that retains or borrows its window and display handle owner.
///
/// Pass a shared owner to obtain an owned surface, or borrow an owner to tie the
/// surface's lifetime to it. Handle implementations must keep the underlying
/// Wayland display and its surface valid for the lifetime of those handles.
pub(crate) fn create_surface<'window>(
    instance: &wgpu::Instance,
    window: impl Into<wgpu::SurfaceTarget<'window>>,
) -> Result<wgpu::Surface<'window>, String> {
    instance
        .create_surface(window)
        .map_err(|error| format!("can't create a GPU surface: {error}"))
}

/// The work performed by a render request and any recovery the caller must arrange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub(crate) enum RenderOutcome {
    /// The scene was rendered and the resulting frame was submitted for display.
    Presented,
    /// Try again on a later redraw; a timeout is not a rendering failure.
    TimedOut,
    /// Resume redraws when the window becomes visible again.
    Occluded,
    /// The configuration was refreshed; schedule another redraw.
    Reconfigured,
    /// Drop the target and create a new surface from its handle owner.
    RecreateSurface,
    /// Recreate the GPU and every target and resource associated with it.
    RecreateDevice,
}

/// A device and Vello renderer shared by all compatible surfaces in the process.
pub(crate) struct Gpu {
    /// Device used for all GPU resource creation and submission.
    pub device: wgpu::Device,
    /// Queue used to submit rendering and presentation commands.
    pub queue: wgpu::Queue,
    /// GPU name and backend for diagnostics.
    pub description: String,
    adapter: wgpu::Adapter,
    renderer: Renderer,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_pipeline_layout: wgpu::PipelineLayout,
    blit_shader: wgpu::ShaderModule,
    blit_pipelines: HashMap<BlitPipelineKey, wgpu::RenderPipeline>,
    pipeline_cache: PipelineCache,
    device_lost: std::sync::Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BlitPipelineKey {
    format: wgpu::TextureFormat,
    premultiply_alpha: bool,
}

impl Gpu {
    /// Opens an adapter compatible with `compatible_surface`, if supplied.
    ///
    /// Device, renderer, and presentation-shader initialization are synchronous.
    /// Create this once during setup. Additional surfaces must be compatible with
    /// the selected adapter.
    pub fn new(
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, String> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface,
            apply_limit_buckets: false,
        }))
        .map_err(|error| format!("no usable GPU: {error}"))?;

        let adapter_info = adapter.get_info();
        let description = format!("{} ({:?})", adapter_info.name, adapter_info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("horizon"),
            required_features: adapter.features()
                & (wgpu::Features::CLEAR_TEXTURE | wgpu::Features::PIPELINE_CACHE),
            // Keep Vello's baseline limits while allowing the adapter's native
            // texture dimensions for large displays and scaled surfaces.
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            ..Default::default()
        }))
        .map_err(|error| format!("can't open the GPU: {error}"))?;

        let device_lost = std::sync::Arc::new(AtomicBool::new(false));
        let callback_device_lost = device_lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            callback_device_lost.store(true, Ordering::Release);
            if reason != wgpu::DeviceLostReason::Destroyed {
                log::error!("GPU device lost: {message}");
            }
        });

        let pipeline_cache = PipelineCache::new(&device, &adapter_info);

        let renderer = Renderer::new(
            &device,
            RendererOptions {
                antialiasing_support: AaSupport::area_only(),
                pipeline_cache: pipeline_cache.handle.clone(),
                ..Default::default()
            },
        )
        .map_err(|error| format!("Vello initialization: {error}"))?;
        pipeline_cache.save();

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("presentation texture"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("presentation"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            immediate_size: 0,
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("presentation"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        Ok(Self {
            device,
            queue,
            description,
            adapter,
            renderer,
            blit_bind_group_layout,
            blit_pipeline_layout,
            blit_shader,
            blit_pipelines: HashMap::new(),
            pipeline_cache,
            device_lost,
        })
    }

    /// Creates presentation resources for a transparent surface.
    ///
    /// The surface must support an 8-bit non-sRGB format and an explicit
    /// transparent alpha mode. Zero or unsupported dimensions are rejected
    /// before the surface is configured.
    pub fn target<'window>(
        &mut self,
        surface: wgpu::Surface<'window>,
        width: u32,
        height: u32,
    ) -> Result<Target<'window>, String> {
        self.check_device()?;

        let configuration = surface_configuration(
            &surface.get_capabilities(&self.adapter),
            &self.device.limits(),
            width,
            height,
        )?;

        let blit_pipeline = self.blit_pipeline(BlitPipelineKey {
            format: configuration.format,
            premultiply_alpha: configuration.alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied,
        });

        let (render_view, blit_bind_group) = self.offscreen(width, height);

        surface.configure(&self.device, &configuration);

        Ok(Target {
            surface,
            configuration,
            render_view,
            blit_bind_group,
            blit_pipeline,
        })
    }

    /// Resizes a target and recreates its intermediate texture for new dimensions.
    ///
    /// Unchanged dimensions are a no-op. Zero or device-limit-exceeding sizes
    /// are rejected. Suspend redraws for hidden or unconfigured surfaces until
    /// the window system provides a usable size.
    pub fn resize(&self, target: &mut Target<'_>, width: u32, height: u32) -> Result<(), String> {
        self.check_device()?;

        validate_dimensions(&self.device.limits(), width, height)?;

        if (width, height) == (target.configuration.width, target.configuration.height) {
            return Ok(());
        }

        let (render_view, blit_bind_group) = self.offscreen(width, height);

        target.configuration.width = width;
        target.configuration.height = height;
        target
            .surface
            .configure(&self.device, &target.configuration);
        target.render_view = render_view;
        target.blit_bind_group = blit_bind_group;

        Ok(())
    }

    /// Renders and presents a scene, or reports why presentation was skipped.
    ///
    /// The surface is acquired before Vello rendering so unavailable surfaces do not
    /// consume rendering work. Recovery outcomes must be handled by the window owner.
    /// Recreating a lost surface requires its live window and display handles.
    pub fn render(
        &mut self,
        target: &mut Target<'_>,
        scene: &Scene,
    ) -> Result<RenderOutcome, String> {
        self.render_with_pre_present(target, scene, || {})
    }

    /// Renders a scene and runs window-system preparation before presentation.
    ///
    /// The hook is not called for skipped acquisition or rendering failures.
    /// Wayland callers can request their next frame callback here so the
    /// presentation commit includes that request.
    pub fn render_with_pre_present(
        &mut self,
        target: &mut Target<'_>,
        scene: &Scene,
        before_present: impl FnOnce(),
    ) -> Result<RenderOutcome, String> {
        if self.device_lost.load(Ordering::Acquire) {
            return Ok(RenderOutcome::RecreateDevice);
        }

        // Acquire before rasterizing so hidden or unavailable surfaces consume
        // no Vello work. A successfully acquired frame is dropped on any error.
        let acquisition = acquire_frame(
            target.surface.get_current_texture(),
            self.device_lost.load(Ordering::Acquire),
        )?;

        let (frame, reconfigure_after_present) = match acquisition {
            FrameAcquisition::Ready {
                frame,
                reconfigure_after_present,
            } => (frame, reconfigure_after_present),
            FrameAcquisition::Skipped(RenderOutcome::Reconfigured) => {
                target
                    .surface
                    .configure(&self.device, &target.configuration);
                return Ok(RenderOutcome::Reconfigured);
            }
            FrameAcquisition::Skipped(outcome) => return Ok(outcome),
        };

        self.render_scene(
            scene,
            &target.render_view,
            target.configuration.width,
            target.configuration.height,
            Color::TRANSPARENT,
        )?;

        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("presentation"),
            });

        encode_blit(
            &mut encoder,
            &frame_view,
            &target.blit_pipeline,
            &target.blit_bind_group,
        );

        self.queue.submit([encoder.finish()]);
        before_present();
        self.queue.present(frame);

        if reconfigure_after_present {
            // Present consumes the frame; no old swapchain texture may remain
            // alive when configure waits for the GPU and replaces the swapchain.
            drop(frame_view);
            target
                .surface
                .configure(&self.device, &target.configuration);
        }

        Ok(RenderOutcome::Presented)
    }

    /// Returns or creates the presentation pipeline for a format and alpha mode.
    fn blit_pipeline(&mut self, key: BlitPipelineKey) -> wgpu::RenderPipeline {
        let pipeline = self.blit_pipelines.entry(key).or_insert_with(|| {
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("presentation"),
                    layout: Some(&self.blit_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &self.blit_shader,
                        entry_point: Some("vertex"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &self.blit_shader,
                        entry_point: Some(if key.premultiply_alpha {
                            "premultiplied_fragment"
                        } else {
                            "straight_fragment"
                        }),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: key.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: self.pipeline_cache.handle.as_ref(),
                });
            // Only a newly compiled pipeline changes the cache. This method is
            // called during target setup, never while presenting a frame.
            self.pipeline_cache.save();
            pipeline
        });
        pipeline.clone()
    }

    /// Creates the intermediate render view and its presentation bind group.
    fn offscreen(&self, width: u32, height: u32) -> (wgpu::TextureView, wgpu::BindGroup) {
        let render_view = self
            .render_texture(width, height, wgpu::TextureUsages::TEXTURE_BINDING)
            .create_view(&wgpu::TextureViewDescriptor::default());
        let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("presentation texture"),
            layout: &self.blit_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&render_view),
            }],
        });
        (render_view, blit_bind_group)
    }

    /// Creates a single-layer straight-alpha RGBA8 render texture.
    fn render_texture(&self, width: u32, height: u32, usage: wgpu::TextureUsages) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Vello target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | usage,
            format: RENDER_FORMAT,
            view_formats: &[],
        })
    }

    /// Rejects operations after the device-lost callback has fired.
    fn check_device(&self) -> Result<(), String> {
        if self.device_lost.load(Ordering::Acquire) {
            Err("GPU device lost; recreate the GPU and all of its targets".into())
        } else {
            Ok(())
        }
    }

    /// Renders a scene into an intermediate texture using Vello.
    fn render_scene(
        &mut self,
        scene: &Scene,
        render_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        background: Color,
    ) -> Result<(), String> {
        self.renderer
            .render_to_texture(
                &self.device,
                &self.queue,
                scene,
                render_view,
                &RenderParams {
                    base_color: background,
                    width,
                    height,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .map_err(|error| format!("Vello rendering: {error}"))
    }
}

/// A surface and its reusable intermediate texture and presentation pipeline.
///
/// The surface lifetime keeps a borrowed handle owner alive. Passing an owned
/// handle provider to [`create_surface`] instead lets wgpu retain that owner.
pub(crate) struct Target<'window> {
    surface: wgpu::Surface<'window>,
    configuration: wgpu::SurfaceConfiguration,
    render_view: wgpu::TextureView,
    blit_bind_group: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
}

#[derive(Debug)]
enum FrameAcquisition {
    Ready {
        frame: wgpu::SurfaceTexture,
        reconfigure_after_present: bool,
    },
    Skipped(RenderOutcome),
}

/// Acquires a surface texture and translates wgpu status into recovery actions.
fn acquire_frame(
    texture: wgpu::CurrentSurfaceTexture,
    device_lost: bool,
) -> Result<FrameAcquisition, String> {
    if device_lost {
        return Ok(FrameAcquisition::Skipped(RenderOutcome::RecreateDevice));
    }

    Ok(match texture {
        wgpu::CurrentSurfaceTexture::Success(frame) => FrameAcquisition::Ready {
            frame,
            reconfigure_after_present: false,
        },
        wgpu::CurrentSurfaceTexture::Suboptimal(frame) => FrameAcquisition::Ready {
            frame,
            reconfigure_after_present: true,
        },
        wgpu::CurrentSurfaceTexture::Timeout => FrameAcquisition::Skipped(RenderOutcome::TimedOut),
        wgpu::CurrentSurfaceTexture::Occluded => FrameAcquisition::Skipped(RenderOutcome::Occluded),
        wgpu::CurrentSurfaceTexture::Outdated => {
            FrameAcquisition::Skipped(RenderOutcome::Reconfigured)
        }
        wgpu::CurrentSurfaceTexture::Lost => {
            FrameAcquisition::Skipped(RenderOutcome::RecreateSurface)
        }
        wgpu::CurrentSurfaceTexture::Validation => {
            return Err(
                "surface acquisition failed GPU validation; inspect wgpu diagnostics".into(),
            );
        }
    })
}

/// Validates nonzero texture dimensions against device limits.
fn validate_dimensions(limits: &wgpu::Limits, width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("GPU target dimensions must be nonzero".into());
    }
    if width > limits.max_texture_dimension_2d || height > limits.max_texture_dimension_2d {
        return Err(format!(
            "GPU target {width}×{height} exceeds the {} pixel texture limit",
            limits.max_texture_dimension_2d
        ));
    }
    Ok(())
}

/// Selects a transparent, non-sRGB surface configuration supported by the GPU.
fn surface_configuration(
    capabilities: &wgpu::SurfaceCapabilities,
    limits: &wgpu::Limits,
    width: u32,
    height: u32,
) -> Result<wgpu::SurfaceConfiguration, String> {
    validate_dimensions(limits, width, height)?;

    let format = capabilities
        .formats
        .iter()
        .copied()
        .find(|format| {
            matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
            )
        })
        .ok_or("surface is incompatible with this GPU or offers no 8-bit non-sRGB format")?;

    let alpha_mode = [
        wgpu::CompositeAlphaMode::PreMultiplied,
        wgpu::CompositeAlphaMode::PostMultiplied,
    ]
    .into_iter()
    .find(|mode| capabilities.alpha_modes.contains(mode))
    .ok_or("surface offers no explicit transparent alpha mode")?;

    Ok(wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Fifo)
        {
            wgpu::PresentMode::Fifo
        } else {
            return Err("surface offers no supported presentation mode".into());
        },
        color_space: wgpu::SurfaceColorSpace::Auto,
        desired_maximum_frame_latency: 2,
        alpha_mode,
        view_formats: vec![],
    })
}

/// Encodes the fullscreen presentation pass, including alpha conversion.
fn encode_blit(
    encoder: &mut wgpu::CommandEncoder,
    target_view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("presentation"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}
