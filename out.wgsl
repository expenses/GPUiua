@group(0) @binding(0) var<storage, read_write> buffers: array<Buffer>;
@group(0) @binding(1) var<storage, read_write> current_offset: u32;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> dispatches: array<vec3<u32>>;

alias Coord = array<u32, 4>;

struct Buffer {
    size: Coord,
    offset: u32,
};

fn coord_clamp(coord: Coord, size: Coord) -> Coord {
    return Coord(
        min(coord[0], size[0] - 1),
        min(coord[1], size[1] - 1),
        min(coord[2], size[2] - 1),
        min(coord[3], size[3] - 1)
    );
}

fn coord_max(a: Coord, b: Coord) -> Coord {
    return Coord(
        max(a[0], b[0]),
        max(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3])
    );
}

fn coord_any_ge(a: Coord, b: Coord) -> bool {
    return a[0] >= b[0]
        || a[1] >= b[1]
        || a[2] >= b[2]
        || a[3] >= b[3];
}

fn coord_to_index(coord: Coord, size: Coord) -> u32 {
    return coord[0]
         + coord[1] * size[0]
         + coord[2] * size[0] * size[1]
         + coord[3] * size[0] * size[1] * size[2];
}

fn index_to_coord(index: u32, size: Coord) -> Coord {
    return Coord(
        (index) % size[0],
        (index / size[0]) % size[1],
        (index / size[0] / size[1]) % size[2],
        (index / size[0] / size[1] / size[2])
    );
}

fn coord_prod(coord: Coord) -> u32 {
    return coord[0]
         * coord[1]
         * coord[2]
         * coord[3];
}

fn write_to_buffer(buffer_index: u32, coord: Coord, value: f32) {
    let buffer = buffers[buffer_index];
    if (coord_any_ge(coord, buffer.size)) {
        return;
    }
    let index = coord_to_index(coord, buffer.size);
    values[buffer.offset + index] = value;
}

fn read_buffer(buffer_index: u32, coord: Coord) -> f32 {
    let buffer = buffers[buffer_index];
    let clamped_coord = coord_clamp(coord, buffer.size);
    let index = coord_to_index(clamped_coord, buffer.size);
    return values[buffer.offset + index];
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    return (value + divisor - 1) / divisor;
}

fn allocate(location: u32, size: Coord) {
    buffers[location].size = size;
    buffers[location].offset = current_offset;
    let length = coord_prod(size);
    current_offset += length;
    dispatches[0] = vec3(div_ceil(length, 64), 1, 1);
}
