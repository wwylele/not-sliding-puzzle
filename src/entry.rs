use bytemuck::*;
use glam::*;
use log::*;
use rfd::*;
use std::{borrow::Cow, future::*, io::Cursor, mem::size_of, pin::*, sync::*, task::*};
use web_time::*;
use wgpu::{util::*, *};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::*,
    event_loop::{EventLoop, EventLoopProxy},
    window::Window,
};

use crate::board::*;
use crate::res;

pub enum UserEvent {
    PicturePickerWake,
}

fn load_texture(
    bytes: &[u8],
    device: &Device,
    queue: &Queue,
    srgb: bool,
) -> Option<(Texture, f32 /*w/h*/)> {
    let img = image::io::Reader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .decode()
        .ok()?
        .to_rgba8();

    let texture_size = Extent3d {
        width: img.width(),
        height: img.height(),
        depth_or_array_layers: 1,
    };

    let format = if srgb {
        TextureFormat::Rgba8UnormSrgb
    } else {
        TextureFormat::Rgba8Unorm
    };

    let texture = device.create_texture(&TextureDescriptor {
        label: None,
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        img.as_raw(),
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(texture_size.width * 4),
            rows_per_image: None,
        },
        texture_size,
    );
    Some((texture, img.width() as f32 / img.height() as f32))
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum ButtonId {
    Small = 0,
    Large = 1,

    Classic = 2,
    Horsey = 3,
    Full = 4,

    Reveal = 5,
    Picture = 6,
}

const BUTTON_COUNT: i32 = 7;

#[derive(Copy, Clone, PartialEq, Eq)]
enum ButtonGroup {
    Size,
    Rule,
    Misc,
}

struct Button {
    id: ButtonId,
    group: ButtonGroup,
    transform: Mat4,
    transform_rev: Mat4,
    selected: bool,
}

impl Button {
    fn new(id: ButtonId, group: ButtonGroup) -> Button {
        Button {
            id,
            group,
            transform: Mat4::IDENTITY,
            transform_rev: Mat4::IDENTITY,
            selected: false,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Zeroable, Pod)]
struct Uniform {
    // These can be just Mat3 but I don't want to deal with weird alignment
    transform: Mat4,
    tex_transform: Mat4,
    mask_transform: Mat4,
    light_transform: Mat4,

    morph: f32,

    // manual paddings
    p0: f32,
    p1: f32,
    p2: f32,
}

struct PicturePickerWaker {
    // Technically this doesn't need to be a Mutex given this is in a single-threaded program.
    // But the safe construction of Waker from Wake requires Arc, which requires Sync,
    // and EventLoopProxy is inconsistently ?Sync across different platforms.
    // To avoid unsafe, let's just slap a Mutex on this cold path.
    event_proxy: Mutex<EventLoopProxy<UserEvent>>,
}

impl Wake for PicturePickerWaker {
    fn wake(self: Arc<Self>) {
        let _ = self
            .event_proxy
            .lock()
            .unwrap()
            .send_event(UserEvent::PicturePickerWake);
    }
}

struct Game<'window> {
    window: &'window Window,
    device: Device,
    surface: Surface<'window>,
    queue: Queue,
    #[allow(dead_code)]
    instance: Instance,
    #[allow(dead_code)]
    adapter: Adapter,
    #[allow(dead_code)]
    shader: ShaderModule,
    #[allow(dead_code)]
    pipeline_layout: PipelineLayout,
    config: SurfaceConfiguration,
    uniform_buffer: Buffer,
    vertex_buffer: Buffer,
    sampler: Sampler,
    mask_sampler: Sampler,
    mask_texture_view: TextureView,
    light_texture_view: TextureView,
    bind_group_layout: BindGroupLayout,

    render_pipeline: RenderPipeline,
    bind_group_piece: BindGroup,
    bind_group_button: BindGroup,

    current_rule: Rule,
    current_size: BoardSize,
    board: Board,

    buttons: Vec<Button>,
    button_hover: Option<ButtonId>,
    window_width: u32,
    window_height: u32,
    board_transform: Mat4,
    board_transform_inv: Mat4,
    picture_transform: Mat4,

    prev_time: Instant,
    animating: bool,

    picture_picker_task: Option<Pin<Box<dyn Future<Output = Option<Vec<u8>>>>>>,
    picture_picker_waker: Arc<PicturePickerWaker>,
}

fn create_bind_group(
    device: &Device,
    bind_group_layout: &BindGroupLayout,
    uniform_buffer: &Buffer,
    texture_view: &TextureView,
    sampler: &Sampler,
    mask_texture_view: &TextureView,
    mask_sampler: &Sampler,
    light_texture_view: &TextureView,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(texture_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(mask_texture_view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::Sampler(mask_sampler),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::TextureView(light_texture_view),
            },
        ],
    })
}

fn get_picture_transform(aspect_ratio: f32) -> Mat4 {
    if aspect_ratio > 1.0 {
        Mat4::from_translation(vec3((1.0 - 1.0 / aspect_ratio) / 2.0, 0.0, 0.0))
            * Mat4::from_scale(vec3(1.0 / aspect_ratio, 1.0, 1.0))
    } else {
        Mat4::from_translation(vec3(0.0, (1.0 - aspect_ratio) / 2.0, 0.0))
            * Mat4::from_scale(vec3(1.0, aspect_ratio, 1.0))
    }
}

impl<'window> Game<'window> {
    pub async fn new(
        window: &'window Window,
        event_proxy: EventLoopProxy<UserEvent>,
    ) -> Game<'window> {
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);

        let instance = Instance::default();

        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                    required_limits: Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Load the shaders from disk
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        // Declare binding points with the shaders
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Define pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VertexBufferLayout {
                    array_stride: 8,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: size_of::<Uniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load textures and samplers
        let (texture, aspect_ratio) =
            load_texture(res::EXAMPLE_PICTURE, &device, &queue, true).unwrap();
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        let picture_transform = get_picture_transform(aspect_ratio);

        let (mask_texture, _) = load_texture(res::PIECE_MASK_PNG, &device, &queue, false).unwrap();
        let mask_texture_view = mask_texture.create_view(&TextureViewDescriptor::default());

        let (light_texture, _) = load_texture(res::BLOCK_PNG, &device, &queue, true).unwrap();
        let light_texture_view = light_texture.create_view(&TextureViewDescriptor::default());

        let (button_texture, _) = load_texture(res::BUTTON_PNG, &device, &queue, true).unwrap();
        let button_texture_view = button_texture.create_view(&TextureViewDescriptor::default());

        let (button_mask, _) = load_texture(res::WHITE_PNG, &device, &queue, true).unwrap();
        let button_mask_view = button_mask.create_view(&TextureViewDescriptor::default());

        let (button_deco, _) = load_texture(res::BUTTON_DECO_PNG, &device, &queue, true).unwrap();
        let button_deco_view = button_deco.create_view(&TextureViewDescriptor::default());

        let sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let mask_sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..Default::default()
        });

        // Define bind groups
        let bind_group_piece = create_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &texture_view,
            &sampler,
            &mask_texture_view,
            &mask_sampler,
            &light_texture_view,
        );

        let bind_group_button = create_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &button_texture_view,
            &sampler,
            &button_mask_view,
            &mask_sampler,
            &button_deco_view,
        );

        // Initial config. Enable VSync
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        config.present_mode = PresentMode::Fifo;
        surface.configure(&device, &config);

        let vertex_buffer_content: &[f32] = &[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: cast_slice(vertex_buffer_content),
            usage: BufferUsages::VERTEX,
        });

        // Initialize first game
        let current_rule = Rule::Horsey;
        let current_size = BoardSize::Small;
        let board = Board::new_shuffle(current_size, current_rule);

        let buttons = vec![
            Button::new(ButtonId::Small, ButtonGroup::Size),
            Button::new(ButtonId::Large, ButtonGroup::Size),
            Button::new(ButtonId::Classic, ButtonGroup::Rule),
            Button::new(ButtonId::Horsey, ButtonGroup::Rule),
            Button::new(ButtonId::Full, ButtonGroup::Rule),
            Button::new(ButtonId::Reveal, ButtonGroup::Misc),
            Button::new(ButtonId::Picture, ButtonGroup::Misc),
        ];

        let event_proxy = Mutex::new(event_proxy);
        let picture_picker_waker = Arc::new(PicturePickerWaker { event_proxy });

        let mut game = Game {
            window,
            device,
            queue,
            surface,
            instance,
            adapter,
            shader,
            pipeline_layout,
            config,
            uniform_buffer,
            vertex_buffer,
            sampler,
            mask_sampler,
            mask_texture_view,
            light_texture_view,
            bind_group_layout,

            render_pipeline,
            bind_group_piece,
            bind_group_button,

            current_rule,
            current_size,
            board,

            buttons,
            button_hover: None,
            window_width: 1,
            window_height: 1,
            board_transform: Mat4::IDENTITY,
            board_transform_inv: Mat4::IDENTITY,
            picture_transform,

            prev_time: Instant::now(),
            animating: true,

            picture_picker_task: None,
            picture_picker_waker,
        };
        game.update_button();
        game
    }

    fn resume_animation(&mut self) {
        if !self.animating {
            self.animating = true;
            self.prev_time = Instant::now();
            self.window.request_redraw();
        }
    }

    // Handle window resize event
    fn resized(&mut self, new_size: PhysicalSize<u32>) {
        self.resume_animation();

        // Reposition board
        let boarder_width = 40.0;
        self.window_width = new_size.width.max(1);
        self.window_height = new_size.height.max(1);
        self.config.width = self.window_width;
        self.config.height = self.window_height;
        self.surface.configure(&self.device, &self.config);

        let window_width = self.window_width as f32;
        let window_height = self.window_height as f32;

        let outer_aspect =
            (window_width - boarder_width * 2.0) / (window_height - boarder_width * 2.0);
        let inner_aspect = self.board.width() as f32 / self.board.height() as f32;
        let board_pixel_width;
        let board_pixel_height;
        if outer_aspect > inner_aspect {
            board_pixel_height = window_height - boarder_width * 2.0;
            board_pixel_width =
                self.board.width() as f32 / self.board.height() as f32 * board_pixel_height;
        } else {
            board_pixel_width = window_width - boarder_width * 2.0;
            board_pixel_height =
                self.board.height() as f32 / self.board.width() as f32 * board_pixel_width;
        }
        self.board_transform = Mat4::from_scale(vec3(
            board_pixel_width / window_width,
            board_pixel_height / window_height,
            1.0,
        ));
        self.board_transform_inv = self.board_transform.inverse();

        // Reposition buttons
        let button_top = (window_height - board_pixel_height) * 0.5;
        let button_bottom = button_top - boarder_width;
        let mut button_left = (window_width - board_pixel_width) * 0.5;
        let mut prev_group = None;
        for button in &mut self.buttons {
            if let Some(prev_group) = prev_group {
                if prev_group == button.group {
                    button_left += boarder_width * 1.5;
                } else {
                    button_left += boarder_width * 1.6;
                }
            }

            let button_right = button_left + boarder_width * 1.5;

            let button_x = (button_left + button_right) * 0.5 / window_width * 2.0 - 1.0;
            let button_y = (button_top + button_bottom) * 0.5 / window_height * 2.0 - 1.0;
            let button_x_scale = (button_right - button_left) / window_width;
            let button_y_scale = (button_top - button_bottom) / window_height;

            button.transform = Mat4::from_translation(vec3(button_x, button_y, 1.0))
                * Mat4::from_scale(vec3(button_x_scale, button_y_scale, 1.0));
            button.transform_rev = button.transform.inverse();

            prev_group = Some(button.group);
        }
    }

    // Make sure all buttons display the correct select state
    fn update_button(&mut self) {
        for button in &mut self.buttons {
            match button.id {
                ButtonId::Small => button.selected = self.current_size == BoardSize::Small,
                ButtonId::Large => button.selected = self.current_size == BoardSize::Large,
                ButtonId::Classic => button.selected = self.current_rule == Rule::Classic,
                ButtonId::Horsey => button.selected = self.current_rule == Rule::Horsey,
                ButtonId::Full => button.selected = self.current_rule == Rule::Full,
                ButtonId::Reveal => button.selected = self.board.reveal,
                _ => (),
            }
        }
    }

    fn initiate_picture_change(&mut self) {
        if self.picture_picker_task.is_some() {
            warn!("Already picking file");
            return;
        }
        let task = async {
            let Some(file) = AsyncFileDialog::new()
                .add_filter("Image", &["bmp", "png", "jpg", "jpeg", "gif"])
                .set_title("Pick a custom picture")
                .pick_file()
                .await
            else {
                return None;
            };
            Some(file.read().await)
        };
        self.picture_picker_task = Some(Box::pin(task) as Pin<Box<_>>);
        self.poll_picture_change();
    }

    fn poll_picture_change(&mut self) {
        let Some(task) = &mut self.picture_picker_task else {
            warn!("Polling when there is no file picker");
            return;
        };
        match task.as_mut().poll(&mut Context::from_waker(&Waker::from(
            self.picture_picker_waker.clone(),
        ))) {
            Poll::Ready(image_data) => {
                self.picture_picker_task = None;

                let Some(image_data) = image_data else {
                    warn!("Failed to pick any file");
                    return;
                };

                info!("Received image data");

                let Some((texture, aspect_ratio)) =
                    load_texture(&image_data, &self.device, &self.queue, true)
                else {
                    warn!("Unable to decode image");
                    return;
                };
                let texture_view = texture.create_view(&TextureViewDescriptor::default());

                self.picture_transform = get_picture_transform(aspect_ratio);
                self.bind_group_piece = create_bind_group(
                    &self.device,
                    &self.bind_group_layout,
                    &self.uniform_buffer,
                    &texture_view,
                    &self.sampler,
                    &self.mask_texture_view,
                    &self.mask_sampler,
                    &self.light_texture_view,
                );
                self.resume_animation();
            }
            Poll::Pending => (),
        }
    }

    // Handle mouse click event
    fn clicked(&mut self) {
        self.resume_animation();

        // Check if we clicked a button
        if let Some(button) = self.button_hover {
            match button {
                ButtonId::Small => {
                    self.current_size = BoardSize::Small;
                    self.board = Board::new_shuffle(self.current_size, self.current_rule);
                }
                ButtonId::Large => {
                    self.current_size = BoardSize::Large;
                    self.board = Board::new_shuffle(self.current_size, self.current_rule);
                }
                ButtonId::Classic => {
                    self.current_rule = Rule::Classic;
                    self.board = Board::new_shuffle(self.current_size, self.current_rule);
                }
                ButtonId::Horsey => {
                    self.current_rule = Rule::Horsey;
                    self.board = Board::new_shuffle(self.current_size, self.current_rule);
                }
                ButtonId::Full => {
                    self.current_rule = Rule::Full;
                    self.board = Board::new_shuffle(self.current_size, self.current_rule);
                }
                ButtonId::Reveal => {
                    if self.current_rule != Rule::Classic {
                        self.board.reveal = !self.board.reveal;
                    }
                }
                ButtonId::Picture => {
                    self.initiate_picture_change();
                }
            }
            self.update_button();
            return;
        }

        // Check if we clicked a piece
        let Some(click_pos) = self.board.hover else {
            return;
        };
        if self.board.empty() == click_pos {
            return;
        }
        if self.board.can_move(click_pos) {
            self.board.move_piece(click_pos, false);
        } else {
            self.board.shake(click_pos);
        }
    }

    // Handle cursor move event
    fn cursor_moved(&mut self, PhysicalPosition { x, y }: PhysicalPosition<f64>) {
        // Check if we are at a piece
        let old_board_hover = self.board.hover;
        let cursor = vec2(
            x as f32 / self.window_width as f32 * 2.0 - 1.0,
            1.0 - y as f32 / self.window_height as f32 * 2.0,
        );
        let board_cursor = self.board_transform_inv * cursor.extend(0.0).extend(1.0);
        if board_cursor.x.abs() > 1.0 || board_cursor.y.abs() > 1.0 {
            self.board.hover = None;
        } else {
            let grid_cursor_f = (board_cursor.xy() + vec2(1.0, 1.0)) / 2.0
                * vec2(self.board.width() as f32, self.board.height() as f32);
            let grid_x = grid_cursor_f.x.clamp(0.0, self.board.width() as f32) as u32;
            let grid_y = grid_cursor_f.y.clamp(0.0, self.board.height() as f32) as u32;
            self.board.hover = Some(uvec2(grid_x, grid_y));
        }

        // Check if we are at a button
        let old_button_hover = self.button_hover;
        self.button_hover = None;
        for button in &self.buttons {
            let button_cursor = button.transform_rev * cursor.extend(0.0).extend(1.0);
            if button_cursor.x.abs() <= 1.0 && button_cursor.y.abs() <= 1.0 {
                self.button_hover = Some(button.id);
                break;
            }
        }

        // If we have changed any focus, start redraw
        if old_board_hover != self.board.hover || old_button_hover != self.button_hover {
            self.resume_animation();
        }
    }

    // Draw next frame, and request for next redraw if we are still in animation
    fn redraw(&mut self) {
        // Move forward animation
        let cur_time = Instant::now();
        let delta = cur_time - self.prev_time;
        self.prev_time = cur_time;
        self.animating = self.board.delta(delta);

        // Get the frame to draw
        let frame = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let view = frame.texture.create_view(&TextureViewDescriptor::default());

        let vertex_buffer_slice = self.vertex_buffer.slice(..);

        // Clear background to black
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let _ = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        self.queue.submit(Some(encoder.finish()));

        let common_tex_transform =
            Mat4::from_translation(vec3(0.5, 0.5, 0.0)) * Mat4::from_scale(vec3(0.5, -0.5, 1.0));

        // Draw buttons
        for button in &self.buttons {
            let tex_transform = Mat4::from_scale(vec3(1.0 / BUTTON_COUNT as f32, 1.0, 1.0))
                * Mat4::from_translation(vec3(button.id as i32 as f32, 0.0, 0.0))
                * common_tex_transform;
            let button_deco_transform =
                (Some(button.id) == self.button_hover) as i32 * 2 + button.selected as i32;
            let light_transform = Mat4::from_scale(vec3(1.0 / 4.0, 1.0, 1.0))
                * Mat4::from_translation(vec3(button_deco_transform as f32, 0.0, 0.0))
                * common_tex_transform;
            let uniform = Uniform {
                transform: button.transform,
                tex_transform,
                mask_transform: Mat4::IDENTITY,
                light_transform,
                morph: 0.0,
                p0: 0.0,
                p1: 0.0,
                p2: 0.0,
            };
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytes_of(&uniform));
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&self.render_pipeline);

                rpass.set_bind_group(0, &self.bind_group_button, &[]);
                rpass.set_vertex_buffer(0, vertex_buffer_slice);
                rpass.draw(0..4, 0..1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        // Sort pieces by their priority
        let mut indices: Vec<UVec2> = (0..self.board.height())
            .flat_map(|y| (0..self.board.width()).map(move |x| uvec2(x, y)))
            .filter(|pos| *pos != self.board.empty())
            .collect();
        indices.sort_by_key(|pos| self.board.cell(*pos).as_ref().unwrap().priority());

        // Draw all pieces
        for pos in indices {
            let Some(piece) = &self.board.cell(pos) else {
                continue;
            };
            let mut current_pos = pos.as_vec2();
            if let Some(moving) = &piece.moving {
                current_pos =
                    Vec2::lerp(moving.previous_pos.as_vec2(), current_pos, moving.progress);
            }

            let translate = current_pos * 2.0
                - vec2(
                    self.board.width() as f32 - 1.0,
                    self.board.height() as f32 - 1.0,
                );
            let source_translate = piece.source_pos.as_vec2() * 2.0
                - vec2(
                    self.board.width() as f32 - 1.0,
                    self.board.height() as f32 - 1.0,
                );
            let grid_scale = Mat4::from_scale(vec3(
                1.0 / self.board.width() as f32,
                1.0 / self.board.height() as f32,
                1.0,
            ));

            let shake = (piece.shake_progress * 10.0).sin() * 0.2;
            let shake = Mat4::from_translation(vec3(shake, 0.0, 0.0));

            let transform = self.board_transform
                * grid_scale
                * Mat4::from_translation(translate.extend(0.0))
                * shake;

            let tex_transform = self.picture_transform
                * common_tex_transform
                * grid_scale
                * Mat4::from_translation(source_translate.extend(0.0));

            let mask_transform = Mat4::from_scale(vec3(1.0 / 4.0, 1.0, 1.0))
                * Mat4::from_translation(vec3(piece.kind as i32 as f32, 0.0, 0.0))
                * common_tex_transform;

            let uniform = Uniform {
                transform,
                tex_transform,
                mask_transform,
                light_transform: common_tex_transform,
                morph: piece.morph_progress,
                p0: 0.0,
                p1: 0.0,
                p2: 0.0,
            };
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytes_of(&uniform));
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&self.render_pipeline);

                rpass.set_bind_group(0, &self.bind_group_piece, &[]);
                rpass.set_vertex_buffer(0, vertex_buffer_slice);
                rpass.draw(0..4, 0..1);
            }

            self.queue.submit(Some(encoder.finish()));
        }

        // Present the frame, and request for the next frame
        frame.present();
        if self.animating {
            self.window.request_redraw();
        }
    }

    // Run the game
    pub fn run(&mut self, event_loop: EventLoop<UserEvent>) {
        event_loop
            .run(move |event, target| {
                if let Event::UserEvent(user_event) = event {
                    match user_event {
                        UserEvent::PicturePickerWake => self.poll_picture_change(),
                    }
                } else if let Event::WindowEvent {
                    window_id: _,
                    event,
                } = event
                {
                    match event {
                        WindowEvent::Resized(new_size) => self.resized(new_size),
                        WindowEvent::MouseInput {
                            state: ElementState::Pressed,
                            button: MouseButton::Left,
                            ..
                        } => self.clicked(),
                        WindowEvent::Touch(Touch {
                            location,
                            phase: TouchPhase::Started,
                            ..
                        }) => {
                            self.cursor_moved(location);
                            self.clicked();
                        }
                        WindowEvent::CursorLeft { .. } => {
                            self.cursor_moved(PhysicalPosition::new(-1.0, -1.0))
                        }
                        WindowEvent::CursorMoved { position, .. } => self.cursor_moved(position),
                        WindowEvent::RedrawRequested => self.redraw(),
                        WindowEvent::CloseRequested => target.exit(),
                        _ => {}
                    };
                }
            })
            .unwrap();
    }
}

pub async fn run(event_loop: EventLoop<UserEvent>, window: Window) {
    let event_proxy = event_loop.create_proxy();
    let mut game = Game::new(&window, event_proxy).await;
    game.run(event_loop);
}
