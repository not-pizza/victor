use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

use crate::db::Embedding;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    embedding_size: u32,
    num_embeddings: u32,
}
pub struct GlobalWgpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: Option<wgpu::ComputePipeline>,
}

thread_local! {
    pub static GLOBAL_WGPU: RefCell<Option<GlobalWgpu>> = RefCell::new(None);
}

static DEVICE_INITIALIZED: AtomicBool = AtomicBool::new(false);
static PIPELINE_INITIALIZED: AtomicBool = AtomicBool::new(false);

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
    }
}

fn init_pipeline(device: &wgpu::Device, flattened_embeddings: &Vec<f32>, uniforms: Uniforms) -> () {
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let embeddings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Embeddings Buffer"),
        contents: bytemuck::cast_slice(&flattened_embeddings),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: embeddings_buffer.as_entire_binding(),
            },
        ],
    });

    let shader_code: &str = include_str!("./shaders/embedding_lookup.wgsl");

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Embedding Lookup"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &shader_module,
        entry_point: "main",
    });

    GLOBAL_WGPU.with(|global_wgpu_cell| {
        if let Some(ref mut global_wgpu) = *global_wgpu_cell.borrow_mut() {
            global_wgpu.pipeline = Some(compute_pipeline);
            PIPELINE_INITIALIZED.store(true, Ordering::SeqCst);
        } else {
        }
    });
}

pub(crate) async fn setup_global_wgpu() {
    if !DEVICE_INITIALIZED.load(Ordering::SeqCst) {
        let global_wgpu = init_wgpu().await;
        GLOBAL_WGPU.with(|global_wgpu_cell| {
            if global_wgpu_cell.borrow().is_none() {
                *global_wgpu_cell.borrow_mut() = Some(global_wgpu);
                DEVICE_INITIALIZED.store(true, Ordering::SeqCst);
            }
        });
    }
}

pub(crate) fn load_embeddings_gpu(device: &wgpu::Device, embeddings: Vec<Embedding>) -> () {
    let uniforms = Uniforms {
        embedding_size: embeddings[0].vector.len() as u32,
        num_embeddings: embeddings.len() as u32,
    };

    // need the clusters flat before putting them in the gpu buffer
    let flattened_embeddings = embeddings
        .into_iter()
        .map(|embedding| embedding.vector)
        .flatten()
        .collect::<Vec<f32>>();

    if !PIPELINE_INITIALIZED.load(Ordering::SeqCst) {
        init_pipeline(device, &flattened_embeddings, uniforms);
    }
}
