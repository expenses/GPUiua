uint size_to_length(uint3 size) {
    return size.x * size.y * size.z;
}

struct Buffer {
    uint3 size;
    float* ptr;

    property uint length {
        get {
            return size_to_length(size);
        }
    }

    float* get(uint3 coord) {
        let clamped_coord = min(coord, size-1);
        let index = clamped_coord.x + clamped_coord.y * size.x + clamped_coord.z * size.x * size.y;
        return &ptr[index];
    }
};

struct PushConstant {
    uint3* dispatches;
    uint* stack_size;
    Buffer* buffers;
    float* values;
    uint* offset;
    uint set_value;
}

[vk::push_constant]
PushConstant push_constant;

Buffer* buffer_at_stack_depth(uint stack_depth) {
    return &push_constant.buffers[push_constant.stack_size[0] - 1 - stack_depth];
}

float load_buffer_at_stack_depth(uint stack_depth, uint3 index) {
    return *buffer_at_stack_depth(stack_depth).get(index);
}

void write_buffer(uint3 index, float value) {
    buffer_at_stack_depth(0).get(index)[0] = value;
}

float* allocate(uint3 size) {
    let length = size_to_length(size);
    let ptr = &push_constant.values[*push_constant.offset];
    printf("Allocating %u @ %lu\n", length, (uint64_t)ptr);
    let stack_index = push_constant.stack_size[0];
    *push_constant.stack_size += 1;
    push_constant.buffers[stack_index].ptr = ptr;
    push_constant.buffers[stack_index].size = size;
    *push_constant.offset += length;
    push_constant.dispatches[0] = size;
    return ptr;
}

[shader("compute")]
void allocate_scalar() {
    let ptr = allocate(uint3(1, 1, 1));
    ptr[0] = push_constant.set_value;
}


[shader("compute")]
void length(uint3 thread_id: SV_DispatchThreadID) {
    buffer_at_stack_depth(0).get(thread_id)[0] = buffer_at_stack_depth(1).length;
}


[shader("compute")]
void clear(uint3 thread_id: SV_DispatchThreadID) {
    buffer_at_stack_depth(0).get(thread_id)[0] = 0;
}

[shader("compute")]
void allocate_fold() {
    let size = buffer_at_stack_depth(0).size;
    allocate(uint3(1, size.yz));
    push_constant.dispatches[1] = size;
}

[shader("compute")]
void range_allocation() {
    allocate(uint3(uint(load_buffer_at_stack_depth(0, 0)), 1, 1));
}

[shader("compute")]
void range(uint3 thread_id: SV_DispatchThreadID) {
    let buffer = buffer_at_stack_depth(0);
    buffer_at_stack_depth(0).ptr[thread_id.x] = thread_id.x;
}

[shader("compute")]
void allocate_copy() {
    allocate(buffer_at_stack_depth(0).size);
}

[shader("compute")]
void allocate_max() {
    allocate(max(buffer_at_stack_depth(0).size, buffer_at_stack_depth(1).size));
}

[shader("compute")]
void allocate_transpose() {
    allocate(max(buffer_at_stack_depth(0).size, buffer_at_stack_depth(1).size.yxz));
}


[shader("compute")]
void add(uint3 thread_id: SV_DispatchThreadID) {
    write_buffer(
        thread_id,
        load_buffer_at_stack_depth(2, thread_id) + load_buffer_at_stack_depth(1, thread_id)
    );
}

[shader("compute")]
void div(uint3 thread_id: SV_DispatchThreadID) {
    write_buffer(
        thread_id,
        load_buffer_at_stack_depth(1, thread_id) / load_buffer_at_stack_depth(2, thread_id)
    );
}

[shader("compute")]
void fold_add(uint3 thread_id: SV_DispatchThreadID) {
    let val  = load_buffer_at_stack_depth(1, thread_id);
    InterlockedAdd(*buffer_at_stack_depth(0).get(uint3(0)), val);
}

[shader("compute")]
void max_vals(uint3 thread_id: SV_DispatchThreadID) {
    write_buffer(
        thread_id,
        max(load_buffer_at_stack_depth(1, thread_id), load_buffer_at_stack_depth(2, thread_id))
    );
}

[shader("compute")]
void copy(uint3 thread_id: SV_DispatchThreadID) {
    write_buffer(
        thread_id,
        load_buffer_at_stack_depth(1, thread_id)
    );
}


[shader("compute")]
void rev(uint3 thread_id: SV_DispatchThreadID) {
    let coord = uint3(buffer_at_stack_depth(1).size.x - 1 - thread_id.x, thread_id.y, thread_id.z);
    let val  = load_buffer_at_stack_depth(1, coord);
    write_buffer(
        thread_id,
        val
    );
}

[shader("compute")]
void table_eq(uint3 thread_id: SV_DispatchThreadID) {
    write_buffer(
        thread_id,
        float(load_buffer_at_stack_depth(1, thread_id) == load_buffer_at_stack_depth(2, thread_id.yxz))
    );
}
