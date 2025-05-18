@group(0) @binding(0) var<storage, read_write> buffers: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> current_offset: u32;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> dispatches: array<vec3<u32>>;

fn write_to_buffer(buffer_index: u32, coord: vec3<u32>, value: f32) {
    let buffer = buffers[buffer_index];
    if (any(coord >= buffer.xyz)) {
        return;
    }
    let index = coord.x + coord.y * buffer.x + coord.z * buffer.x * buffer.y;
    values[buffer.w + index] = value;
}

fn read_buffer(buffer_index: u32, coord: vec3<u32>) -> f32 {
    let buffer = buffers[buffer_index];
    let clamped_coord = min(coord, buffer.xyz-1);
    let index = clamped_coord.x + clamped_coord.y * buffer.x + clamped_coord.z * buffer.x * buffer.y;
    return values[buffer.w + index];
}

fn div_ceil(value: vec3<u32>, divisor: u32) -> vec3<u32> {
    return (value + divisor - 1) / divisor;
}

fn allocate(location: u32, size: vec3<u32>) {
    buffers[location] = vec4(size, current_offset);
    current_offset += size.x * size.y * size.z;
    dispatches[0] = size;//div_ceil(size, 4);
}
