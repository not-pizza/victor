@group(0) @binding(0) var<storage, read> embeddingsBuffer: array<f32>;
@group(0) @binding(1) var<storage, read> embeddingBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> resultsBuffer: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

struct Uniforms {
    embedding_size: u32,
    num_embeddings: u32,
};

fn euclidean(a_start: u32, b_start: u32, size: u32) -> f32 {
    var distance: f32 = 0.0;
    for (var i: u32 = 0; i < size; i = i + 1) {
        let diff: f32 = embeddingsBuffer[a_start + i] - embeddingBuffer[b_start + i];
        distance = distance + diff * diff;
    }
    return sqrt(distance);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u.embedding_size) {
        return;
    }

    var min_distance: f32 = 1e20;
    var min_index: u32 = 0;

    for (var embedding_index: u32 = 0; embedding_index < u.num_embeddings; embedding_index = embedding_index + 1) {
        let distance: f32 = euclidean(embedding_index * u.embedding_size, 0, u.embedding_size);
        if (distance < min_distance) {
            min_distance = distance;
            min_index = embedding_index;
        }
    }

    if (global_id.x == 0) {
        for (var i: u32 = 0; i < u.embedding_size; i = i + 1) {
            resultsBuffer[i] = embeddingsBuffer[min_index * u.embedding_size + i];
        }
    }
}