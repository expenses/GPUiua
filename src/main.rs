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

        for (step, pipeline) in pipelines.iter().enumerate() {
            pass.set_pipeline(pipeline);
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
        let buffers = cast_slice::<[u32; 8]>(&buffers_range[256..]);
        let values = values_readback.get_mapped_range(..);
        let values = cast_slice::<f32>(&values);
        //dbg!(&values[..20]);
        for (i, &arr @ [x, y, z, w, offset, ..]) in buffers
            .iter()
            .take_while(|&&buffer| buffer != [0; 8])
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
                    println!("{:?}", &values[offset..offset + x as usize]);
                    offset += x as usize;
                }
            }
        }
    }
}

use std::collections::HashMap;

use logos::Logos;

#[derive(Debug)]
enum OpType {
    Add,
    Mul,
    Div,
    Eq,
    Sin,
    Abs,
    Max,
}

#[derive(Debug)]
struct Operation {
    ty: OpType,
    back: bool,
}

#[derive(Debug)]
enum Code {
    Operation(Operation),
    Value(f32),
    Table(Operation),
    Rev,
    Dup,
    Range,
}

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"\s+")]
enum Token {
    #[regex(r"add|\+")]
    Add,
    #[regex(r"mul|\*")]
    Mul,
    #[regex(r"div|÷")]
    Div,
    #[regex(r"eq|=")]
    Eq,
    #[regex(r"range|⇡")]
    Range,
    #[regex(r"table|⊞")]
    Table,
    #[regex(r"sin|∿")]
    Sin,
    #[regex(r"abs|⌵")]
    Abs,
    #[regex(r"rev|⇌")]
    Rev,
    #[regex(r"max|↥")]
    Max,
    #[regex(r"back|˜")]
    Back,
    #[regex(r"dup|\.")]
    Dup,
    #[regex("#[^\n]*")]
    Comment,
    #[token("\n", priority = 3)]
    LineBreak,
    #[regex(r"[0-9]+\.?[0-9]*", |lex| lex.slice().parse::<f32>().unwrap())]
    Value(f32),
}

fn parse_as_op(token: Token) -> Option<OpType> {
    match token {
        Token::Eq => Some(OpType::Eq),
        Token::Abs => Some(OpType::Abs),
        Token::Add => Some(OpType::Add),
        Token::Mul => Some(OpType::Mul),
        Token::Div => Some(OpType::Div),
        Token::Sin => Some(OpType::Sin),
        Token::Max => Some(OpType::Max),
        Token::LineBreak
        | Token::Dup
        | Token::Value(_)
        | Token::Back
        | Token::Range
        | Token::Table
        | Token::Rev
        | Token::Comment => None,
    }
}

type Line = Vec<Code>;

fn parse_code(input: &str) -> Vec<Line> {
    let mut code = Vec::new();
    let mut line = Vec::new();

    let mut lexer = Token::lexer(input);

    while let Some(token) = lexer.next() {
        let mut token = match token {
            Ok(token) => token,
            Err(error) => panic!("{:?}: {:?}", error, code),
        };

        let mut back = false;

        while token == Token::Back {
            back = !back;
            token = lexer.next().unwrap().unwrap();
        }

        match token {
            Token::Comment => {}
            Token::Back => unreachable!(),
            Token::Value(value) => line.push(Code::Value(value)),
            Token::Range => line.push(Code::Range),
            Token::Dup => line.push(Code::Dup),
            Token::Rev => line.push(Code::Rev),
            Token::Table => line.push(Code::Table({
                let mut token = lexer.next().unwrap().unwrap();
                let mut back = false;

                while token == Token::Back {
                    back = !back;
                    token = lexer.next().unwrap().unwrap();
                }

                Operation {
                    ty: match parse_as_op(token) {
                        Some(op) => op,
                        None => panic!("{:?}", token),
                    },
                    back,
                }
            })),
            Token::LineBreak => {
                code.push(std::mem::take(&mut line));
            }
            other => line.push(Code::Operation(Operation {
                ty: match parse_as_op(other) {
                    Some(op) => op,
                    None => panic!("{:?}", other),
                },
                back,
            })),
        }
    }

    if !line.is_empty() {
        code.push(line);
    }

    code
}

#[derive(Clone, Debug)]
enum StackItem {
    Buffer(u32),
    Other { str: String, is_output: bool },
}

impl StackItem {
    fn size(&self) -> String {
        match &self {
            Self::Buffer(id) => format!("buffers[{}].size", id),
            Self::Other { .. } => "vec4(1)".to_string(),
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
                    "thread_id = index_to_coord(thread.x, buffers[{}].size)",
                    self.num_allocated
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
            Some(ModifierAccess::Transpose) => "thread_id.yxzw",
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
            Some(ModifierAccess::Transpose) => "thread_id.yxzw",
        };
        self.stack.push(StackItem::Other {
            str: func(b.str(first_access), a.str("thread_id")),
            is_output,
        })
    }

    fn handle_operation(&mut self, operation: &Operation) {
        match operation.ty {
            OpType::Add => {
                self.handle_diadic(operation.back, |a, b| format!("({} + {})", a, b));
            }
            OpType::Eq => {
                self.handle_diadic(operation.back, |a, b| format!("f32({} == {})", a, b));
            }
            OpType::Div => {
                self.handle_diadic(operation.back, |a, b| format!("({} / {})", a, b));
            }
            OpType::Mul => {
                self.handle_diadic(operation.back, |a, b| format!("({} * {})", a, b));
            }
            OpType::Max => {
                self.handle_diadic(operation.back, |a, b| format!("max({}, {})", a, b));
            }
            OpType::Sin => {
                self.handle_monadic(|v| format!("sin({})", v));
            }
            OpType::Abs => {
                self.handle_monadic(|v| format!("abs({})", v));
            }
        }
    }
}

fn main() {
    let input = std::env::args().nth(1).unwrap();

    let code = parse_code(&input);
    let mut parser = Parser::default();

    for line in code.iter() {
        for code in line.iter().rev() {
            match code {
                Code::Value(val) => parser.stack.push(StackItem::Other {
                    str: val.to_string(),
                    is_output: false,
                }),
                Code::Dup => {
                    let top = parser.stack.last().unwrap().clone();
                    parser.stack.push(top);
                }
                Code::Range => {
                    parser.finish_write();
                    let len = parser.pop();
                    parser.output_size = format!("vec4({}, 1, 1, 1)", len.str("unreachable!"));
                    parser.stack.push(StackItem::Other {
                        str: "f32(thread_id.x)".to_owned(),
                        is_output: true,
                    });
                }
                Code::Table(op) => {
                    parser.finish_write();
                    let a = parser.peek(0);
                    let b = parser.peek(1);
                    parser.output_size = format!("max({}, {}.yxzw)", a.size(), b.size());
                    parser.modifier_expecting_op = Some(ModifierAccess::Transpose);
                    parser.handle_operation(op);
                }
                Code::Rev => {
                    parser.finish_write();
                    let v = parser.peek(0);
                    let size = v.size();
                    let str = v.str(&format!(
                        "vec4({}.x - 1 - thread_id.x, thread_id.yzw)",
                        v.size()
                    ));
                    parser.stack.push(StackItem::Other {
                        str,
                        is_output: true,
                    });
                    parser.output_size = size;
                }
                Code::Operation(operation) => parser.handle_operation(operation),
            }
        }
    }

    let mut code = include_str!("../out.wgsl").to_string();

    parser.finish_write();

    for (i, function) in parser.functions.iter().enumerate() {
        if i % 2 == 1 {
            code.push_str(&format!("@compute @workgroup_size(64,1,1) fn step_{}(@builtin(global_invocation_id) thread : vec3<u32>) {{\nvar thread_id: vec4<u32>;\n", i));
        } else {
            code.push_str(&format!(
                "@compute @workgroup_size(1,1,1) fn step_{}() {{\n",
                i
            ));
        }
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
