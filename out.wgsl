@group(0) @binding(0) var<storage, read_write> buffers: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> stack_len: u32;
@group(0) @binding(2) var<storage, read_write> current_offset: u32;
@group(0) @binding(3) var<storage, read_write> values: array<f32>;
@group(0) @binding(4) var<storage, read_write> dispatches: array<vec3<u32>>;

fn size_to_length(size: vec3<u32> ) -> u32 {
    return size.x * size.y * size.z;
}

fn load_buffer(stack_depth: u32) -> vec4<u32> {
    return buffers[stack_len - 1 - stack_depth];
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

fn allocate(size: vec3<u32>) -> u32 {
    let length = size_to_length(size);
    let offset = current_offset;
    let stack_index = stack_len;
    current_offset += length;
    stack_len += 1;
    buffers[stack_index] = vec4(size, offset);
    dispatches[0] = size;
    return offset;
}

fn pop() {
    stack_len -= 1;
    buffers[stack_len-1] = buffers[stack_len];
}

fn dup() {
    stack_len += 1;
    buffers[stack_len-1] = buffers[stack_len-2];
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

@compute @workgroup_size(1,1,1)
fn rev(@builtin(global_invocation_id) thread_id : vec3<u32>) {
    write_to_buffer(0, thread_id, read_buffer(1, vec3<u32>(load_buffer(1).x - 1 - thread_id.x, thread_id.yz)));
}