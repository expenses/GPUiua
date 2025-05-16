struct ShaderCode {
    code: String,
    func_index: u32,
}

impl ShaderCode {
    fn push_function(&mut self, code: &str) -> u32 {
        self.code.push_str(&format!(
            "
@compute @workgroup_size(1,1,1)
fn func_{}(@builtin(global_invocation_id) thread_id : vec3<u32>) {{
    {};
}}
",
            self.func_index, code
        ));
        let func_index = self.func_index;
        self.func_index += 1;
        func_index
    }

    fn push_scalar(&mut self, value: f32) -> u32 {
        self.push_function(&format!("values[allocate(vec3(1,1,1))] = {}", value))
    }

    fn write(&mut self, code: &str) -> u32 {
        self.push_function(&format!("write_to_buffer(0, thread_id, {})", code))
    }

    fn allocate(&mut self, code: &str) -> u32 {
        self.push_function(&format!("allocate({})", code))
    }
}

use std::collections::HashMap;

use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"\s+")]
enum Token {
    #[token("⇡")]
    Range,
    #[token("⊞")]
    Table,
    #[token("=")]
    Eq,
    #[token(".")]
    Dup,
    #[token("⇌")]
    Rev,
    #[token("⊸")]
    On,
    #[token("↥")]
    Max,
    #[regex("#[^\n]*")]
    Comment,
    #[regex("[0-9]+", |lex| lex.slice().parse::<f32>().unwrap())]
    Value(f32),
}

#[derive(Debug)]
enum PipelineRef {
    Num(u32),
    Name(&'static str),
}

#[derive(Debug)]
enum Operation {
    Set {
        allocate: PipelineRef,
        write: PipelineRef,
    },
    Scalar(u32),
}

fn main() {
    let code = "
        ⇡5  # 0 through 4
        ⊞=. # Identity matrix
        ⊸⇌  # Reverse
        ↥   # Max
        ";

    let instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();

    let shader_str = std::fs::read_to_string("out.wgsl").unwrap();

    let mut shader_code = ShaderCode {
        code: shader_str,
        func_index: 0,
    };

    let lines_reversed = code
        .lines()
        .rev()
        .flat_map(|string| string.chars().chain(std::iter::once('\n')))
        .collect::<String>();
    println!("{}", &lines_reversed);
    let mut lexer = Token::lexer(&lines_reversed).peekable();

    let mut operations = Vec::new();

    while let Some(token) = lexer.next() {
        let (first_id, second_id) = if lexer.peek() == Some(&Ok(Token::Dup)) {
            let _ = lexer.next();
            (0, 0)
        } else {
            (0, 1)
        };

        match token.unwrap() {
            Token::Table => {
                let next = lexer.next().unwrap().unwrap();
                let (first_id, second_id) = if lexer.peek() == Some(&Ok(Token::Dup)) {
                    let _ = lexer.next();
                    (0, 0)
                } else {
                    (0, 1)
                };
                let allocate = PipelineRef::Num(shader_code.allocate(&format!(
                    "max(load_buffer({}).xyz, load_buffer({}).yxz)",
                    first_id, second_id
                )));
                operations.push(Operation::Set {
                    allocate,
                    write: match next {
                        Token::Eq => PipelineRef::Num(shader_code.write(&format!(
                            "f32(read_buffer({}, thread_id) == read_buffer({}, thread_id.yxz))",
                            first_id + 1,
                            second_id + 1
                        ))),
                        _ => panic!(),
                    },
                });
            }
            Token::Max => {
                operations.push(Operation::Set {
                    allocate: PipelineRef::Name("allocate_copy"),
                    write: PipelineRef::Num(shader_code.write(&format!(
                        "max(read_buffer({}, thread_id), read_buffer({}, thread_id))",
                        first_id + 1,
                        second_id + 1
                    ))),
                });
            }
            Token::Rev => {
                operations.push(Operation::Set {
                    allocate: PipelineRef::Name("allocate_copy"),
                    write: PipelineRef::Name("rev"),
                });
            }
            Token::Value(value) => {
                operations.push(Operation::Scalar(shader_code.push_scalar(value)))
            }
            Token::Range => {
                operations.push(Operation::Set {
                    allocate: PipelineRef::Name("allocate_range"),
                    write: PipelineRef::Name("write_range"),
                });
            }
            Token::On | Token::Comment => {}
            _ => {
                dbg!(&token);
            }
        };
    }

    //println!("{}", &shader_code.code);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_code.code.into()),
    });

    let entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[entry(0), entry(1), entry(2), entry(3), entry(4)],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let create_pipeline = |entry_point| -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    let mut named_pipelines = HashMap::new();

    let mut add = |name| {
        named_pipelines.insert(name, create_pipeline(name));
    };

    add("allocate_range");
    add("write_range");
    add("allocate_copy");
    add("rev");

    let pipelines: Vec<_> = (0..shader_code.func_index)
        .map(|i| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(&format!("func_{}", i)),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect();

    let create_buffer = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };

    let buffers = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * 4 * 100,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buffers_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * 4 * 100,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stack_len = create_buffer(4);
    let current_offset = create_buffer(4);
    let dispatches = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * 3 * 2,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });
    let values = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * 1000,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let values_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * 1000,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: stack_len.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: current_offset.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dispatches.as_entire_binding(),
            },
        ],
    });

    let mut command_encoder = device.create_command_encoder(&Default::default());
    let mut pass = command_encoder.begin_compute_pass(&Default::default());

    pass.set_bind_group(0, &bind_group, &[]);

    for op in operations.iter().rev() {
        match op {
            Operation::Scalar(num) => {
                pass.set_pipeline(&pipelines[*num as usize]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            Operation::Set { allocate, write } => {
                pass.set_pipeline(match allocate {
                    PipelineRef::Num(num) => &pipelines[*num as usize],
                    PipelineRef::Name(name) => &named_pipelines[name],
                });
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(match write {
                    PipelineRef::Num(num) => &pipelines[*num as usize],
                    PipelineRef::Name(name) => &named_pipelines[name],
                });
                pass.dispatch_workgroups_indirect(&dispatches, 0);
            }
        }
    }

    drop(pass);
    command_encoder.copy_buffer_to_buffer(&values, 0, &values_readback, 0, values.size());
    command_encoder.copy_buffer_to_buffer(&buffers, 0, &buffers_readback, 0, buffers.size());
    let buffer = command_encoder.finish();
    let submit = queue.submit([buffer]);
    values_readback.map_async(wgpu::MapMode::Read, .., |res| {
        //dbg!(res);
    });
    buffers_readback.map_async(wgpu::MapMode::Read, .., |res| ());
    device
        .poll(wgpu::PollType::WaitForSubmissionIndex(submit))
        .unwrap();
    let buffers_range = buffers_readback.get_mapped_range(..);
    let buffers = cast_slice::<[u32; 4]>(&buffers_range);
    let values = values_readback.get_mapped_range(..);
    let values = cast_slice::<f32>(&values);
    for (i, &[x, y, z, offset]) in buffers
        .iter()
        .take_while(|&buffer| *buffer != [0_u32; 4])
        .enumerate()
    {
        println!("{}:", i);
        if (y, z) == (1, 1) {
            println!(
                "{:?}",
                &values[offset as usize..offset as usize + x as usize]
            );
        } else if z == 1 {
            let mut offset = offset as usize;
            for _ in 0..y {
                println!(
                    "{:?}",
                    &values[offset as usize..offset as usize + x as usize]
                );
                offset += x as usize;
            }
        }
    }
}

fn cast_slice<T>(slice: &[u8]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const T,
            slice.len() / std::mem::size_of::<T>(),
        )
    }
}
