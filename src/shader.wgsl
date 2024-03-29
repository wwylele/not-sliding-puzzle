
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) mask_coords: vec2<f32>,
    @location(2) light_coords: vec2<f32>,
}

struct Uniform {
    transform: mat4x4<f32>,
    tex_transform: mat4x4<f32>,
    mask_transform: mat4x4<f32>,
    light_transform: mat4x4<f32>,
    morph: f32,
    p0: f32,
    p1: f32,
    p2: f32,
}

@group(0) @binding(0)
var<uniform> uni: Uniform;

@group(0) @binding(1)
var image: texture_2d<f32>;

@group(0) @binding(2)
var image_sampler: sampler;

@group(0) @binding(3)
var mask: texture_2d<f32>;

@group(0) @binding(4)
var mask_sampler: sampler;

@group(0) @binding(5)
var light: texture_2d<f32>;

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VertexOutput {
    let xyzw = vec4<f32>(position, 0.0, 1.0);
    var result: VertexOutput;
    result.position = uni.transform * xyzw;
    result.tex_coords = (uni.tex_transform * xyzw).xy;
    result.mask_coords = (uni.mask_transform * xyzw).xy;
    result.light_coords = (uni.light_transform * xyzw).xy;
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(image, image_sampler, vertex.tex_coords);
    let mask = textureSample(mask, mask_sampler, vertex.mask_coords).x;
    let light = textureSample(light, image_sampler, vertex.light_coords);
    if mask > 1.0 - uni.morph {
        discard;
    }
    return color * (1 - light.w) + vec4<f32>(light.xyz, 1.0) * light.w;
}
