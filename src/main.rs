struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

impl Context {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

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

        Self {
            device,
            queue,
            bind_group_layout,
            pipeline_layout,
        }
    }

    async fn run(&self, shader_code: &str, num_steps: usize) {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

        let pipelines: Vec<_> = (0..num_steps)
            .map(|i| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&self.pipeline_layout),
                        module: &shader,
                        entry_point: Some(&format!("step_{}", i)),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            })
            .collect();

        // min_storage_buffer_offset_alignment of 256.
        let data_and_buffers_len = 256 + 4 * 4 * 200;
        let data_and_buffers = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data_and_buffers_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let data_and_buffers_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data_and_buffers_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dispatches = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4 * 3 * 2,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let values = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let values_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
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

        let mut command_encoder = self.device.create_command_encoder(&Default::default());
        let mut pass = command_encoder.begin_compute_pass(&Default::default());

        pass.set_bind_group(0, &bind_group, &[]);

        for step in 0..num_steps {
            pass.set_pipeline(&pipelines[step]);
            if step % 2 == 1 {
                pass.dispatch_workgroups_indirect(&dispatches, 0);
            } else {
                pass.dispatch_workgroups(1, 1, 1);
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
        let submit = self.queue.submit([buffer]);
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
        self.device
            .poll(wgpu::PollType::WaitForSubmissionIndex(submit))
            .unwrap();
        values_rx.await.unwrap();
        buffers_rx.await.unwrap();
        let buffers_range = data_and_buffers_readback.get_mapped_range(..);
        let buffers = cast_slice::<[u32; 4]>(&buffers_range[256..]);
        let values = values_readback.get_mapped_range(..);
        let values = cast_slice::<f32>(&values);
        //dbg!(&values[..20]);
        for (i, &arr @ [x, y, z, offset]) in buffers
            .iter()
            .take_while(|&&buffer| buffer != [0; 4])
            .enumerate()
        {
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
}

use std::collections::HashMap;

use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"\s+")]
enum Token {
    #[regex(r"add|\+")]
    Add,
    #[regex(r"mul|\*")]
    Mul,
    #[regex(r"div")]
    Div,
    #[regex(r"=")]
    Eq,
    #[token("range")]
    Range,
    #[token("table")]
    Table,
    #[token("sin")]
    Sin,
    #[token("abs")]
    Abs,
    #[token("rev")]
    Rev,
    #[token("max")]
    Max,
    #[token("back")]
    Back,
    #[token(r"dup")]
    Dup,
    #[regex("#[^\n]*")]
    Comment,
    #[regex("[0-9]+", |lex| lex.slice().parse::<f32>().unwrap())]
    Value(f32),
}

#[derive(Clone, Debug)]
enum StackItem {
    Buffer(u32),
    Other { str: String, is_output: bool },
}

impl StackItem {
    fn size(&self) -> String {
        match &self {
            Self::Buffer(id) => format!("buffers[{}].xyz", id),
            Self::Other { .. } => format!("vec3(1, 1, 1)"),
        }
    }

    fn is_output(&self) -> bool {
        match &self {
            Self::Buffer(_) => false,
            Self::Other { is_output, .. } => *is_output,
        }
    }

    fn str(&self, coord: &str) -> String {
        match &self {
            Self::Buffer(id) => format!("read_buffer({}, {})", id, coord),
            Self::Other { str, .. } => str.to_string(),
        }
    }
}

#[derive(Debug)]
enum ModifierAccess {
    Normal,
    Transpose,
}

#[derive(Default, Debug)]
struct Parser {
    output_size: String,
    functions: Vec<Vec<String>>,
    stack: Vec<StackItem>,
    num_allocated: u32,
    modifier_expecting_op: Option<ModifierAccess>,
}

impl Parser {
    fn pop(&mut self) -> StackItem {
        self.stack.pop().unwrap()
    }

    fn peek(&self, depth: usize) -> &StackItem {
        self.stack.get(self.stack.len() - 1 - depth).unwrap()
    }

    fn modifier_expecting_op(&mut self) -> Option<ModifierAccess> {
        self.modifier_expecting_op.take()
    }

    fn finish_write(&mut self) {
        let mut admin_commands = Vec::new();
        let mut write_commands = Vec::new();
        let mut command_to_buffer_index = HashMap::new();

        for item in &mut self.stack {
            if !item.is_output() {
                continue;
            }

            if let Some(index) = command_to_buffer_index.get(&item.str("thread_id")) {
                admin_commands.push(format!(
                    "buffers[{}] = buffers[{}]",
                    self.num_allocated, index
                ));
            } else {
                command_to_buffer_index.insert(item.str("thread_id"), self.num_allocated);

                admin_commands.push(format!(
                    "allocate({}, {})",
                    self.num_allocated, self.output_size
                ));
                write_commands.push(format!(
                    "write_to_buffer({}, thread_id, {})",
                    self.num_allocated,
                    item.str("thread_id")
                ));
            }

            *item = StackItem::Buffer(self.num_allocated);

            self.num_allocated += 1;
        }

        if !admin_commands.is_empty() {
            self.functions.push(admin_commands);
        }
        if !write_commands.is_empty() {
            self.functions.push(write_commands);
        }

        /*self.stack.retain(|item| {
            if !(item.is_output() && item.is_dup()) {
                return true;
            }

            self.commands.push("dup()".into());

            false
        });*/
    }

    fn handle_monadic<F: Fn(String) -> String>(&mut self, func: F) {
        let v = self.pop();
        let modifier_expecting_op = self.modifier_expecting_op();
        let is_output = v.is_output() || modifier_expecting_op.is_some();
        let access = match modifier_expecting_op {
            Some(ModifierAccess::Normal) | None => "thread_id",
            Some(ModifierAccess::Transpose) => "thread_id.yxz",
        };
        self.stack.push(StackItem::Other {
            str: func(v.str(access)),
            is_output,
        })
    }

    fn handle_diadic<F: Fn(String, String) -> String>(&mut self, back: bool, func: F) {
        let mut a = self.pop();
        let mut b = self.pop();
        if back {
            std::mem::swap(&mut a, &mut b);
        }
        let modifier_expecting_op = self.modifier_expecting_op();
        let is_output = a.is_output() || b.is_output() || modifier_expecting_op.is_some();
        let first_access = match modifier_expecting_op {
            Some(ModifierAccess::Normal) | None => "thread_id",
            Some(ModifierAccess::Transpose) => "thread_id.yxz",
        };
        self.stack.push(StackItem::Other {
            str: func(b.str(first_access), a.str("thread_id")),
            is_output,
        })
    }
}

fn main() {
    let input = std::env::args().nth(1).unwrap();

    let mut parser = Parser::default();

    let mut lexer = Token::lexer(&input);

    while let Some(token) = lexer.next() {
        let mut token = token.unwrap();

        let mut back = false;

        while token == Token::Back {
            back = !back;
            token = lexer.next().unwrap().unwrap();
        }

        match token {
            Token::Value(val) => parser.stack.push(StackItem::Other {
                str: val.to_string(),
                is_output: false,
            }),
            Token::Dup => {
                let top = parser.stack.last().unwrap().clone();
                parser.stack.push(top);
            }
            Token::Range => {
                parser.finish_write();
                let len = parser.pop();
                parser.output_size = format!("vec3({}, 1, 1)", len.str("unreachable!"));
                parser.stack.push(StackItem::Other {
                    str: "f32(thread_id.x)".to_owned(),
                    is_output: true,
                });
            }
            Token::Table => {
                parser.finish_write();
                let a = parser.peek(0);
                let b = parser.peek(1);
                parser.output_size = format!("max({}, {}.yxz)", a.size(), b.size());
                parser.modifier_expecting_op = Some(ModifierAccess::Transpose);
            }
            Token::Rev => {
                parser.finish_write();
                let v = parser.peek(0);
                let size = v.size();
                let str = v.str(&format!(
                    "vec3({}.x - 1 - thread_id.x, thread_id.yz)",
                    v.size()
                ));
                parser.stack.push(StackItem::Other {
                    str: str,
                    is_output: true,
                });
                parser.output_size = size;
            }
            Token::Add => {
                parser.handle_diadic(back, |a, b| format!("({} + {})", a, b));
            }
            Token::Eq => {
                parser.handle_diadic(back, |a, b| format!("f32({} == {})", a, b));
            }
            Token::Div => {
                parser.handle_diadic(back, |a, b| format!("({} / {})", a, b));
            }
            Token::Mul => {
                parser.handle_diadic(back, |a, b| format!("({} * {})", a, b));
            }
            Token::Max => {
                parser.handle_diadic(back, |a, b| format!("max({}, {})", a, b));
            }
            Token::Sin => {
                parser.handle_monadic(|v| format!("sin({})", v));
            }
            Token::Abs => {
                parser.handle_monadic(|v| format!("abs({})", v));
            }
            Token::Back => unreachable!(),
            Token::Comment => {}
        }
    }

    let mut code = include_str!("../out.wgsl").to_string();

    parser.finish_write();

    for (i, function) in parser.functions.iter().enumerate() {
        code.push_str(&format!("@compute @workgroup_size(1,1,1) fn step_{}(@builtin(global_invocation_id) thread_id : vec3<u32>) {{\n", i));
        for line in function {
            code.push_str(line);
            code.push_str(";\n");
        }
        code.push_str("}\n");
    }

    println!("{}", code);

    let context = pollster::block_on(Context::new());

    pollster::block_on(context.run(&code, parser.functions.len()));
}

fn cast_slice<T>(slice: &[u8]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const T,
            slice.len() / std::mem::size_of::<T>(),
        )
    }
}
