struct Data {
    stack_len: u32,
    current_offset: u32,
    dipped_len: u32,
}

@group(0) @binding(0) var<storage, read_write> buffers: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> data: Data;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> dispatches: array<vec3<u32>>;

fn size_to_length(size: vec3<u32> ) -> u32 {
    return size.x * size.y * size.z;
}

fn load_buffer(stack_depth: u32) -> vec4<u32> {
    return buffers[data.stack_len - 1 - stack_depth];
}

fn write_to_buffer_no_overflow(stack_depth: u32, coord: vec3<u32>, value: f32) {
    let buffer = load_buffer(stack_depth);
    if (any(coord >= buffer.xyz)) {
        return;
    }
    let index = coord.x + coord.y * buffer.x + coord.z * buffer.x * buffer.y;
    values[buffer.w + index] = value;
}

fn write_to_buffer(stack_depth: u32, coord: vec3<u32>, value: f32) {
    let buffer = load_buffer(stack_depth);
    let clamped_coord = min(coord, buffer.xyz-1);
    let index = clamped_coord.x + clamped_coord.y * buffer.x + clamped_coord.z * buffer.x * buffer.y;
    values[buffer.w + index] = value;
}

fn read_buffer(stack_depth: u32, coord: vec3<u32>) -> f32 {
    let buffer = load_buffer(stack_depth);
    let clamped_coord = min(coord, buffer.xyz-1);
    let index = clamped_coord.x + clamped_coord.y * buffer.x + clamped_coord.z * buffer.x * buffer.y;
    return values[buffer.w + index];
}

fn div_ceil(value: vec3<u32>, divisor: u32) -> vec3<u32> {
    return (value + divisor - 1) / divisor;
}

fn allocate(size: vec3<u32>) -> u32 {
    let length = size_to_length(size);
    let offset = data.current_offset;
    let stack_index = data.stack_len;
    data.current_offset += length;
    data.stack_len += 1;
    buffers[stack_index] = vec4(size, offset);
    dispatches[0] = div_ceil(size, 4);
    return offset;
}

fn pop() {
    data.stack_len -= 1;
    buffers[data.stack_len-1] = buffers[data.stack_len];
}

const DIP_OFFSET: u32 = 100;

fn dip() {
    buffers[DIP_OFFSET + data.dipped_len] = buffers[data.stack_len-1];
    data.dipped_len += 1;
    data.stack_len -= 1;
}

fn undip() {
    data.stack_len += 1;
    buffers[data.stack_len - 1] = buffers[DIP_OFFSET + data.dipped_len - 1];
    data.dipped_len -= 1;
}

fn dup() {
    data.stack_len += 1;
    buffers[data.stack_len-1] = buffers[data.stack_len-2];
}

fn allocate_range() {
    allocate(vec3<u32>(u32(values[load_buffer(0).w]), 1, 1));
}

fn allocate_copy() {
    allocate(load_buffer(0).xyz);
}

fn allocate_max() {
    allocate(max(load_buffer(0).xyz, load_buffer(1).xyz));
}
