use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub embedding_size: u32,
    pub num_embeddings: u32,
}
pub struct GlobalWgpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: Option<wgpu::ComputePipeline>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub embeddings_buffer: Option<wgpu::Buffer>,
    pub search_buffer: Option<wgpu::Buffer>,
    pub result_buffer: Option<wgpu::Buffer>,
}

thread_local! {
    pub static GLOBAL_WGPU: RefCell<Option<GlobalWgpu>> = RefCell::new(None);
}

pub static DEVICE_INITIALIZED: AtomicBool = AtomicBool::new(false);
pub static PIPELINE_INITIALIZED: AtomicBool = AtomicBool::new(false);

async fn init_wgpu() -> GlobalWgpu {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    GlobalWgpu {
        device,
        queue,
        pipeline: None,
        bind_group: None,
        embeddings_buffer: None,
        search_buffer: None,
        result_buffer: None,
    }
}

pub fn init_pipeline(flattened_embeddings: &mut Vec<f32>, uniforms: Uniforms) -> () {
    // over-allocate the size of the buffer by 2x so we have room for future embeddings
    let total_cap: usize = flattened_embeddings.len() * 2;

    flattened_embeddings.resize(total_cap, 0.0);
    GLOBAL_WGPU.with(|g| {
        if let Some(ref mut global_wgpu) = *g.borrow_mut() {
            let device = &global_wgpu.device;
            let embeddings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Embeddings Buffer"),
                contents: bytemuck::cast_slice(&flattened_embeddings),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let search_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Search Buffer"),
                size: 1024,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Result Buffer"),
                size: 1024,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: embeddings_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: search_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: result_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let shader_code: &str = include_str!("./shaders/embedding_lookup.wgsl");

            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Embedding Lookup"),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

            let compute_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let compute_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &shader_module,
                    entry_point: "main",
                });

            global_wgpu.pipeline = Some(compute_pipeline);
            global_wgpu.bind_group = Some(bind_group);
            global_wgpu.embeddings_buffer = Some(embeddings_buffer);
            PIPELINE_INITIALIZED.store(true, Ordering::SeqCst);
        } else {
        }
    });
}

pub(crate) async fn setup_global_wgpu() {
    if !DEVICE_INITIALIZED.load(Ordering::SeqCst) {
        let global_wgpu = init_wgpu().await;
        GLOBAL_WGPU.with(|g| {
            if g.borrow().is_none() {
                *g.borrow_mut() = Some(global_wgpu);
                DEVICE_INITIALIZED.store(true, Ordering::SeqCst);
            }
        });
    }
}

pub(crate) fn load_embeddings_gpu(flattened_embeddings: &mut Vec<f32>) -> () {
    GLOBAL_WGPU.with(|g| {
        if let Some(g) = &*g.borrow() {
            let queue = &g.queue;
            let embeddings_buffer = &g.embeddings_buffer;
            if let Some(embeddings_buffer) = embeddings_buffer {
                queue.write_buffer(
                    embeddings_buffer,
                    0,
                    bytemuck::cast_slice(&flattened_embeddings),
                );
            }
        }
    });
}

pub(crate) fn lookup_embeddings_gpu() -> () {
    if PIPELINE_INITIALIZED.load(Ordering::SeqCst) {
        GLOBAL_WGPU.with(|g| {
            if let Some(g) = &*g.borrow() {
                let device = &g.device;
                let queue = &g.queue;
                let pipeline = &g
                    .pipeline
                    .as_ref()
                    .expect("Compute pipeline not initialized");
                let bind_group = &g.bind_group.as_ref().expect("Bind group not initialized");

                let mut command_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Compute Command Encoder"),
                    });

                {
                    let mut compute_pass =
                        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Compute Pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(pipeline);
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.dispatch_workgroups(1, 1, 1); // Adjust based on your needs
                }

                queue.submit(Some(command_encoder.finish()));
            }
        })
    }
}
