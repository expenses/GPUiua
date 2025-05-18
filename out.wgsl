struct Buffer {
    size: vec4<u32>,
    offset: u32,
};

@group(0) @binding(0) var<storage, read_write> buffers: array<Buffer>;
@group(0) @binding(1) var<storage, read_write> current_offset: u32;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> dispatches: array<vec3<u32>>;

fn coord_to_index(coord: vec4<u32>, size: vec4<u32>) -> u32 {
    return coord.x +
        coord.y * size.x +
        coord.z * size.x * size.y +
        coord.w * size.x * size.y * size.z;
}

fn index_to_coord(index: u32, size: vec4<u32>) -> vec4<u32> {
    return vec4(
        (index) % size.x,
        (index / size.x) % size.y,
        (index / size.x / size.y) % size.z,
        (index / size.x / size.y / size.z)
    );
}

fn write_to_buffer(buffer_index: u32, coord: vec4<u32>, value: f32) {
    let buffer = buffers[buffer_index];
    if (any(coord >= buffer.size)) {
        return;
    }
    let index = coord_to_index(coord, buffer.size);
    values[buffer.offset + index] = value;
}

fn read_buffer(buffer_index: u32, coord: vec4<u32>) -> f32 {
    let buffer = buffers[buffer_index];
    let clamped_coord = min(coord, buffer.size-1);
    let index = coord_to_index(clamped_coord, buffer.size);
    return values[buffer.offset + index];
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    return (value + divisor - 1) / divisor;
}

fn allocate(location: u32, size: vec4<u32>) {
    buffers[location].size = size;
    buffers[location].offset = current_offset;
    let length = size.x * size.y * size.z * size.w;
    current_offset += length;
    dispatches[0] = vec3(div_ceil(length, 64), 1, 1);
}
