struct ShaderCode {
    code: String,
    func_index: u32,
    operations: Vec<PipelineRef>,
    scalar_function: String,
}

impl ShaderCode {
    fn push_function(&mut self, code: &str, workgroup_size: u32) -> u32 {
        self.code.push_str(&format!(
            "
@compute @workgroup_size({0},{0},{0})
fn func_{1}(@builtin(global_invocation_id) thread_id : vec3<u32>) {{
    {2}
}}
",
            workgroup_size, self.func_index, code
        ));
        let func_index = self.func_index;
        self.func_index += 1;
        func_index
    }

    fn finish_scalar_function(&mut self) {
        if !self.scalar_function.is_empty() {
            let scalar_function = std::mem::take(&mut self.scalar_function);
            let id = self.push_function(&scalar_function, 1);
            self.operations.push(PipelineRef::Scalar(id));
        }
    }

    fn push_scalar_function_line(&mut self, line: &str) {
        self.scalar_function.push_str(&format!("{};\n", line));
    }

    fn push_scalar(&mut self, value: f32) {
        self.push_scalar_function_line(&format!("values[allocate(vec3(1,1,1))] = {}", value));
    }

    fn write(&mut self, code: &str, num_operands: u32) {
        self.finish_scalar_function();
        let num = self.push_function(&format!("write_to_buffer(0, thread_id, {});", code), 4);
        self.operations.push(PipelineRef::Num(num));
        for _ in 0..num_operands {
            self.push_scalar_function_line("pop()");
        }
    }

    // overflow needs to be handled e.g. with a range 3, where the dispatch size of 4x4x4 will cause 3 to be written to the last value not 2.
    fn write_no_overflow(&mut self, code: &str, num_operands: u32) {
        self.finish_scalar_function();
        let num = self.push_function(
            &format!("write_to_buffer_no_overflow(0, thread_id, {});", code),
            4,
        );
        self.operations.push(PipelineRef::Num(num));
        for _ in 0..num_operands {
            self.push_scalar_function_line("pop()");
        }
    }
}

use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"\s+")]
enum Token {
    #[token("⇡")]
    Range,
    #[token("⊞")]
    Table,
    #[token("+")]
    Add,
    #[token("=")]
    Eq,
    #[token(".")]
    Dup,
    #[token("⊙")]
    Dip,
    #[token("÷")]
    Div,
    #[token("×")]
    Mul,
    #[token("⇌")]
    Rev,
    #[token("⊸")]
    On,
    #[token("↥")]
    Max,
    #[token("↧")]
    Min,
    #[token("~")]
    Back,
    #[token("-")]
    Sub,
    #[token("≠")]
    Ne,
    #[regex("#[^\n]*")]
    Comment,
    #[regex("[0-9]+", |lex| lex.slice().parse::<f32>().unwrap())]
    Value(f32),
}

#[derive(Debug)]
enum PipelineRef {
    Scalar(u32),
    Num(u32),
}

fn main() {
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();

    let mut shader_code = ShaderCode {
        code: include_str!("../out.wgsl").into(),
        func_index: 0,
        operations: Vec::new(),
        scalar_function: String::new(),
    };

    let input = std::env::args().nth(1).unwrap();

    let mut tokens: Vec<_> = Token::lexer(&input).collect();
    tokens.reverse();

    let mut lexer = tokens.into_iter().peekable();

    while let Some(token) = lexer.next() {
        let token = token.unwrap();

        let mut read_1_index = 1;
        let mut read_2_index = 2;

        if lexer.peek() == Some(&Ok(Token::Back)) {
            std::mem::swap(&mut read_1_index, &mut read_2_index);
            let _ = lexer.next();
        }

        let mut dips = 0;

        while lexer.peek() == Some(&Ok(Token::Dip)) {
            dips += 1;
            shader_code.push_scalar_function_line("dip()");
            let _ = lexer.next();
        }

        let mut read_1 = format!("read_buffer({}, thread_id)", read_1_index);
        let mut read_2 = format!("read_buffer({}, thread_id)", read_2_index);
        let mut diadic_alloc_method = "allocate_max()".into();

        if lexer.peek() == Some(&Ok(Token::Table)) {
            read_1 = format!("read_buffer({}, thread_id)", read_1_index);
            read_2 = format!("read_buffer({}, thread_id.yxz)", read_2_index);
            diadic_alloc_method = format!(
                "allocate(max(load_buffer({}).xyz, load_buffer({}).yxz))",
                read_1_index - 1,
                read_2_index - 1
            );

            let _ = lexer.next();
            true
        } else {
            false
        };

        match token {
            Token::Value(value) => {
                shader_code.push_scalar(value);
            }
            Token::Dup => {
                shader_code.push_scalar_function_line("dup()");
            }
            Token::Range => {
                shader_code.push_scalar_function_line("allocate_range()");
                shader_code.write_no_overflow("f32(thread_id.x)", 1);
            }
            Token::Div => {
                shader_code.push_scalar_function_line(&diadic_alloc_method);
                shader_code.write(&format!("{} / {}", read_2, read_1), 2);
            }
            Token::Add => {
                shader_code.push_scalar_function_line(&diadic_alloc_method);
                shader_code.write(&format!("{} + {}", read_1, read_2), 2);
            }
            Token::Mul => {
                shader_code.push_scalar_function_line(&diadic_alloc_method);
                shader_code.write(&format!("{} * {}", read_1, read_2), 2);
            }
            Token::Eq => {
                shader_code.push_scalar_function_line(&diadic_alloc_method);
                shader_code.write(&format!("f32({} == {})", read_1, read_2), 2);
            }
            Token::Max => {
                shader_code.push_scalar_function_line(&diadic_alloc_method);
                shader_code.write(&format!("max({}, {})", read_1, read_2), 2);
            }
            Token::Min => {
                shader_code.push_scalar_function_line(&diadic_alloc_method);
                shader_code.write(&format!("min({}, {})", read_1, read_2), 2);
            }
            Token::Rev => {
                shader_code.push_scalar_function_line("allocate_copy()");
                shader_code.write(
                    &format!(
                        "read_buffer({0}, vec3(load_buffer({0}).x - 1 - thread_id.x, thread_id.yz))",
                        read_1_index
                    ),
                    1,
                );
            }
            _ => panic!(),
        }

        for _ in 0..dips {
            shader_code.push_scalar_function_line("undip()");
        }
    }

    shader_code.finish_scalar_function();

    println!("{}", &shader_code.code);

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
        entries: &[entry(0), entry(1), entry(2), entry(3)],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

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

    // min_storage_buffer_offset_alignment of 256.
    let data_and_buffers_len = 256 + 4 * 4 * 200;
    let data_and_buffers = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: data_and_buffers_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let data_and_buffers_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: data_and_buffers_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dispatches = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * 3 * 2,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });
    let values = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024 * 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let values_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024 * 1024,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &data_and_buffers,
                    offset: 256,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &data_and_buffers,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dispatches.as_entire_binding(),
            },
        ],
    });

    let mut command_encoder = device.create_command_encoder(&Default::default());
    let mut pass = command_encoder.begin_compute_pass(&Default::default());

    pass.set_bind_group(0, &bind_group, &[]);

    for op in shader_code.operations.iter() {
        match op {
            PipelineRef::Scalar(num) => {
                pass.set_pipeline(&pipelines[*num as usize]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            PipelineRef::Num(num) => {
                pass.set_pipeline(&pipelines[*num as usize]);
                pass.dispatch_workgroups_indirect(&dispatches, 0);
            }
        }
    }

    drop(pass);
    command_encoder.copy_buffer_to_buffer(&values, 0, &values_readback, 0, values.size());
    command_encoder.copy_buffer_to_buffer(
        &data_and_buffers,
        0,
        &data_and_buffers_readback,
        0,
        data_and_buffers.size(),
    );
    let buffer = command_encoder.finish();
    let submit = queue.submit([buffer]);
    let (mut values_tx, values_rx) = async_oneshot::oneshot::<()>();
    values_readback.map_async(wgpu::MapMode::Read, .., move |res| {
        res.unwrap();
        values_tx.send(()).unwrap();
    });
    let (mut buffers_tx, buffers_rx) = async_oneshot::oneshot::<()>();
    data_and_buffers_readback.map_async(wgpu::MapMode::Read, .., move |res| {
        res.unwrap();
        buffers_tx.send(()).unwrap();
    });
    device
        .poll(wgpu::PollType::WaitForSubmissionIndex(submit))
        .unwrap();
    pollster::block_on(values_rx).unwrap();
    pollster::block_on(buffers_rx).unwrap();
    let buffers_range = data_and_buffers_readback.get_mapped_range(..);
    let stack_len = cast_slice::<u32>(&buffers_range)[0];
    let buffers = cast_slice::<[u32; 4]>(&buffers_range[256..]);
    let values = values_readback.get_mapped_range(..);
    let values = cast_slice::<f32>(&values);
    for (i, &arr @ [x, y, z, offset]) in buffers.iter().take(stack_len as _).enumerate() {
        println!("{} {:?}:", i, arr);
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
