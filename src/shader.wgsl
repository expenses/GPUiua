struct Data {
    current_offset: u32,
};

@group(0) @binding(0) var<storage, read_write> buffers: array<Buffer>;
@group(0) @binding(1) var<storage, read_write> data: Data;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> dispatches: array<vec3<u32>>;

// Random functions from https://marktension.nl/blog/my_favorite_wgsl_random_func_so_far/
// Not perfect but good enough and I'm lazy.

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm for u32.
fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
fn float_construct_from_u32(m_in: u32) -> f32 {
    let ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    let ieeeOne = 0x3F800000u;      // 1.0 in IEEE binary32

    var m = m_in;
    m &= ieeeMantissa;              // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                   // Add fractional part to 1.0

    let f = bitcast<f32>(m);        // Range [1:2]
    return f - 1.0;                 // Range [0:1]
}

// Pseudo-random value in half-open range [0:1] from a u32 seed.
fn random_uniform(seed: u32) -> f32 {
    return float_construct_from_u32(hash_u32(seed));
}

alias Coord = array<u32, 4>;

struct Buffer {
    size: Coord,
    offset: u32,
};

fn coord_clamp(coord: Coord, size: Coord) -> Coord {
    return Coord(
        min(coord[0], size[0] - 1),
        min(coord[1], size[1]),
        min(coord[2], size[2]),
        min(coord[3], size[3])
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

fn coord_any_gt(a: Coord, b: Coord) -> bool {
    return a[0] > b[0]
        || a[1] > b[1]
        || a[2] > b[2]
        || a[3] > b[3];
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
         * max(coord[1], 1)
         * max(coord[2], 1)
         * max(coord[3], 1);
}

fn coord_reverse(coord: Coord, size: Coord) -> Coord {
    return Coord(
        size[0] - 1 - coord[0],
        coord[1],
        coord[2],
        coord[3]
    );
}

fn coord_table(a: Coord, b: Coord) -> Coord {
    // todo!
    if (any(max(vec3(a[1], a[2], a[3]), vec3(b[1], b[2], b[3])) != vec3(0))) {
        return Coord(0,0,0,0);
    }

    return Coord(
        a[0],
        b[0],
        0,
        0
    );
}


fn coord_transpose(coord: Coord) -> Coord {
    return Coord(
        coord[1],
        coord[0],
        coord[2],
        coord[3]
    );
}

fn coord_plus_x(coord: Coord, x: f32) -> Coord {
    return Coord(
        u32(f32(coord[0]) + x),
        coord[1],
        coord[2],
        coord[3]
    );
}

fn write_to_buffer(buffer_index: u32, coord: Coord, value: f32) {
    let buffer = buffers[buffer_index];
    if (coord_any_gt(coord, buffer.size)) {
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
    buffers[location].offset = data.current_offset;
    let length = coord_prod(size);
    data.current_offset += length;
}

fn dispatch_for_buffer(buffer_index: u32) {
    let length = coord_prod(buffers[buffer_index].size);
    dispatches[0] = vec3(div_ceil(length, 64), 1, 1);
}

fn add(x: f32, y: f32) -> f32 {
    return x + y;
}

fn eq(x: f32, y: f32) -> f32 {
    return f32(x == y);
}

fn div(x: f32, y: f32) -> f32 {
    return x / y;
}

fn mul(x: f32, y: f32) -> f32 {
    return x * y;
}

fn sub(x: f32, y: f32) -> f32 {
    return x - y;
}

fn ge(x: f32, y: f32) -> f32 {
    return f32(x >= y);
}

fn gt(x: f32, y: f32) -> f32 {
    return f32(x > y);
}

fn lt(x: f32, y: f32) -> f32 {
    return f32(x < y);
}

fn le(x: f32, y: f32) -> f32 {
    return f32(x <= y);
}

fn ne(x: f32, y: f32) -> f32 {
    return f32(x != y);
}

fn not(v: f32) -> f32 {
    return 1 - v;
}

fn neg(v: f32) -> f32 {
    return -v;
}

fn modulus(x: f32, y: f32) -> f32 {
    // https://cs.stackexchange.com/questions/53031/what-is-the-difference-between-modulo-and-modulus#comment334015_53033
    return (((x % y) + y) % y);
}
