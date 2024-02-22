use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub embedding_size: u32,
    pub num_embeddings: u32,
}

#[derive(Clone)]
pub struct GlobalWgpu {
    pub device: Rc<wgpu::Device>,
    pub queue: Rc<wgpu::Queue>,
    pub pipeline: Option<Rc<wgpu::ComputePipeline>>,
    pub bind_group: Option<Rc<wgpu::BindGroup>>,
    pub embeddings_buffer: Option<Rc<wgpu::Buffer>>,
    pub search_buffer: Option<Rc<wgpu::Buffer>>,
    pub result_buffer: Option<Rc<wgpu::Buffer>>,
    pub readback_buffer: Option<Rc<wgpu::Buffer>>,
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
        device: device.into(),
        queue: queue.into(),
        pipeline: None.into(),
        bind_group: None.into(),
        embeddings_buffer: None.into(),
        search_buffer: None.into(),
        result_buffer: None.into(),
        readback_buffer: None.into(),
    }
}

pub fn init_pipeline(flattened_embeddings: &mut Vec<f32>, uniforms: Uniforms) -> () {
    // over-allocate the size of the buffer by 2x so we have room for future embeddings
    let total_cap: usize = flattened_embeddings.len() * 2;

    flattened_embeddings.resize(total_cap, 0.0);
    GLOBAL_WGPU.with(|g| {
        let mut gw = g.borrow_mut();
        if let Some(ref mut global_wgpu) = *gw {
            let device = &global_wgpu.device;
            let embeddings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Embeddings Buffer"),
                contents: bytemuck::cast_slice(&flattened_embeddings),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let search_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Search Buffer"),
                size: 2000,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Result Buffer"),
                size: 2000,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Readback Buffer"),
                size: 2000,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, // For copying data into it and reading back on CPU
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

            global_wgpu.pipeline = Some(compute_pipeline.into());
            global_wgpu.bind_group = Some(bind_group.into());
            global_wgpu.embeddings_buffer = Some(embeddings_buffer.into());
            global_wgpu.result_buffer = Some(result_buffer.into());
            global_wgpu.readback_buffer = Some(readback_buffer.into());
            global_wgpu.search_buffer = Some(search_buffer.into());
            PIPELINE_INITIALIZED.store(true, Ordering::SeqCst);
        }
    });
}

pub(crate) async fn setup_global_wgpu() {
    if !DEVICE_INITIALIZED.load(Ordering::SeqCst) {
        let new_global_wgpu = init_wgpu().await;

        GLOBAL_WGPU.with(|g| {
            let mut global_wgpu = g.borrow_mut();

            if global_wgpu.is_none() {
                *global_wgpu = Some(new_global_wgpu);
                DEVICE_INITIALIZED.store(true, Ordering::SeqCst);
            } else {
                // this should never run unless the device is initialized twice
                panic!("Global WGPU already initialized");
            }
        });
    }
}

pub fn load_embeddings_gpu(flattened_embeddings: &[f32]) {
    GLOBAL_WGPU.with(|g| {
        if let Some(global_wgpu) = &*g.borrow() {
            let queue = &global_wgpu.queue;
            if let Some(embeddings_buffer) = global_wgpu.embeddings_buffer.as_ref() {
                queue.write_buffer(
                    embeddings_buffer,
                    0,
                    bytemuck::cast_slice(flattened_embeddings),
                );
            }
        }
    });
}

pub(crate) async fn lookup_embeddings_gpu(lookup_embedding: &[f32]) -> Result<Vec<f32>, String> {
    if PIPELINE_INITIALIZED.load(Ordering::SeqCst) {
        if let Some(global_wgpu) = GLOBAL_WGPU.with(|g| g.borrow().clone()) {
            let device = &global_wgpu.device;
            let queue = &global_wgpu.queue;
            let pipeline = &global_wgpu.pipeline.unwrap();
            let bind_group = &global_wgpu.bind_group.unwrap();
            let result_buffer = &global_wgpu.result_buffer.unwrap();
            let readback_buffer = &global_wgpu.readback_buffer.unwrap();
            let search_buffer = &global_wgpu.search_buffer.unwrap();

            queue.write_buffer(&search_buffer, 0, bytemuck::cast_slice(lookup_embedding));

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

                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(64, 1, 1);
            }

            command_encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, 2000);

            queue.submit(Some(command_encoder.finish()));

            let readback_buffer_slice = readback_buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            readback_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            match receiver.receive().await {
                Some(Ok(())) => {
                    let data_raw = readback_buffer_slice.get_mapped_range();
                    if data_raw.len() % std::mem::size_of::<f32>() == 0 {
                        let data: &[f32] = bytemuck::cast_slice(&data_raw);
                        Ok(data.to_vec())
                    } else {
                        Err("Data length is not aligned with f32 size".to_string())
                    }
                }
                _ => Err("Failed to read back buffer".to_string()),
            }
        } else {
            Err("Global WGPU not found".to_string())
        }
    } else {
        Err("Pipeline not initialized".to_string())
    }
}
