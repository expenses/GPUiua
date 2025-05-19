struct ShaderModule {
    code: String,
    num_functions: usize,
    final_stack_len: usize,
}

fn generate_module(blocks: Vec<CodeBlock>) -> Result<ShaderModule, CodeBlock> {
    let mut parser = Parser::default();

    for code in &blocks {
        let mut back = false;
        let mut dipped = Vec::new();

        for modifier in &code.modifiers {
            match modifier {
                Modifier::Gap => {
                    parser.stack.pop();
                }
                Modifier::Dip => {
                    //parser.finish_write();
                    dipped.push(parser.stack.pop().unwrap());
                }
                Modifier::Back => back = !back,
                Modifier::Table => {
                    parser.finish_write().map_err(|()| code.clone())?;
                    let a = parser.peek(0);
                    let b = parser.peek(1);
                    parser.output_size = format!(
                        "coord_max({0}, Coord({1}[1], {1}[0], {1}[2], {1}[3]))",
                        a.size(),
                        b.size()
                    );
                    parser.modifier_expecting_op = Some(ModifierAccess::Transpose);
                }
            }
        }

        match code.code {
            FunctionOrOp::Function(_) => panic!(),
            FunctionOrOp::Op(Op::Value(val)) => parser.stack.push(StackItem::Other {
                str: format!("f32({})", val),
                is_output: false,
            }),
            FunctionOrOp::Op(Op::Range) => {
                parser.finish_write().map_err(|()| code.clone())?;
                let len = parser.pop().map_err(|()| code.clone())?;
                parser.output_size = format!("Coord(u32({}), 1, 1, 1)", len.str("unreachable!"));
                parser.stack.push(StackItem::Other {
                    str: "f32(thread_id[0])".to_owned(),
                    is_output: true,
                });
            }
            FunctionOrOp::Op(Op::Rev) => {
                parser.finish_write().map_err(|()| code.clone())?;
                let v = parser.peek(0);
                let size = v.size();
                let str = v.str(&format!(
                    "Coord({}[0] - 1 - thread_id[0], thread_id[1], thread_id[2], thread_id[3])",
                    v.size()
                ));
                parser.delayed_pops.push(parser.stack.len() - 1);
                parser.stack.push(StackItem::Other {
                    str,
                    is_output: true,
                });
                parser.output_size = size;
            }

            FunctionOrOp::Op(Op::BasicOp(ref operation)) => parser
                .handle_operation(back, *operation)
                .map_err(|()| code.clone())?,
        }

        while let Some(item) = dipped.pop() {
            parser.stack.push(item);
        }
    }

    parser
        .finish()
        .map_err(|()| blocks.last().unwrap().clone())?;

    let mut code = include_str!("../out.wgsl").to_string();

    for (i, function) in parser.functions.iter().enumerate() {
        if i % 2 == 1 {
            code.push_str(&format!("@compute @workgroup_size(64,1,1) fn step_{}(@builtin(global_invocation_id) thread : vec3<u32>) {{\n", i));
            code.push_str("var thread_id: Coord;\n");
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

    Ok(ShaderModule {
        code,
        num_functions: parser.functions.len(),
        final_stack_len: parser.stack.len(),
    })
}

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
            label: Some("bind_group_layout"),
            entries: &[entry(0), entry(1), entry(2), entry(3)],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
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

    async fn run(&self, module: &ShaderModule) -> Vec<ReadBackValue> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shader"),
                source: wgpu::ShaderSource::Wgsl((&module.code).into()),
            });

        let pipelines: Vec<_> = (0..module.num_functions)
            .map(|i| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(&format!("step_{}", i)),
                        layout: Some(&self.pipeline_layout),
                        module: &shader,
                        entry_point: Some(&format!("step_{}", i)),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            })
            .collect();

        // min_storage_buffer_offset_alignment of 256.
        let buffers_len = 4 * 4 * 200;
        let buffers = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffers"),
            size: buffers_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffers_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffers_readback"),
            size: buffers_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dispatches = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dispatches"),
            size: 4 * 3 * 2,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let temp_data = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp_data"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let values_size = 1024 * 1024 * 4;
        let values = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("values"),
            size: values_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let values_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("values_readback"),
            size: values_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // webgpu on chrome doesn't let you bind an indirect + writeable storage buffer. Sad!
        let dummy_dispatches = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_dispatches"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let admin_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("admin_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_data.as_entire_binding(),
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

        let write_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("write_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_dispatches.as_entire_binding(),
                },
            ],
        });

        let mut command_encoder = self.device.create_command_encoder(&Default::default());
        let mut pass = command_encoder.begin_compute_pass(&Default::default());

        for (step, pipeline) in pipelines.iter().enumerate() {
            pass.set_pipeline(pipeline);
            if step % 2 == 1 {
                pass.set_bind_group(0, &write_bind_group, &[]);
                pass.dispatch_workgroups_indirect(&dispatches, 0);
            } else {
                pass.set_bind_group(0, &admin_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        drop(pass);
        command_encoder.copy_buffer_to_buffer(&values, 0, &values_readback, 0, values.size());
        command_encoder.copy_buffer_to_buffer(&buffers, 0, &buffers_readback, 0, buffers.size());
        let buffer = command_encoder.finish();
        let submit = self.queue.submit([buffer]);
        let (mut values_tx, values_rx) = async_oneshot::oneshot::<()>();
        values_readback.map_async(wgpu::MapMode::Read, .., move |res| {
            res.unwrap();
            values_tx.send(()).unwrap();
        });
        let (mut buffers_tx, buffers_rx) = async_oneshot::oneshot::<()>();
        buffers_readback.map_async(wgpu::MapMode::Read, .., move |res| {
            res.unwrap();
            buffers_tx.send(()).unwrap();
        });
        self.device
            .poll(wgpu::PollType::WaitForSubmissionIndex(submit))
            .unwrap();
        values_rx.await.unwrap();
        buffers_rx.await.unwrap();
        let buffers_range = buffers_readback.get_mapped_range(..);
        let buffers = cast_slice::<[u32; 5]>(&buffers_range);
        let values = values_readback.get_mapped_range(..);
        let values = cast_slice::<f32>(&values);

        buffers
            .iter()
            .take(module.final_stack_len)
            .map(|&[x, y, z, w, offset, ..]| ReadBackValue {
                size: [x, y, z, w],
                values: values[offset as usize..offset as usize + (x * y * z * w) as usize]
                    .to_vec(),
            })
            .collect()
    }

    async fn run_string(
        &self,
        string: &str,
        left_to_right: bool,
    ) -> Result<Vec<ReadBackValue>, String> {
        let module =
            generate_module(parse_code(string, left_to_right).map_err(|err| format!("{:?}", err))?)
                .map_err(|err| format!("{:?}", err))?;
        Ok(self.run(&module).await)
    }

    async fn run_string_and_get_string_output(&self, string: &str, left_to_right: bool) -> String {
        let values = match self.run_string(string, left_to_right).await {
            Ok(values) => values,
            Err(error) => return format!("{:?}", error),
        };

        let mut output = String::new();

        for value in values {
            match value.size {
                [1, 1, 1, 1] => output.push_str(&format!("{}\n", value.values[0])),
                [_, 1, 1, 1] => output.push_str(&format!("{:?}\n", value.values)),
                [_, y, 1, 1] => {
                    for chunk in value.values.chunks(y as usize) {
                        output.push_str(&format!("{:?}\n", chunk));
                    }
                    output.push_str("\n");
                }

                _ => {}
            }
        }

        output
    }
}

use std::{collections::HashMap, ops::Range};

use logos::Logos;

#[derive(Debug, Clone, Copy)]
enum OpType {
    Add,
    Mul,
    Div,
    Eq,
    Sin,
    Sub,
    Round,
    Abs,
    Max,
    Floor,
    Ceil,
    Dup,
    Pop,
}

#[derive(Debug, Clone)]
struct CodeBlock {
    modifiers: Vec<Modifier>,
    span: std::ops::Range<usize>,
    code: FunctionOrOp,
}

#[derive(Debug, Clone)]
enum FunctionOrOp {
    Function(Vec<CodeBlock>),
    Op(Op),
}

#[derive(Debug, Clone, Copy)]
enum Op {
    BasicOp(OpType),
    Value(f32),
    Range,
    Rev,
}

#[derive(Debug, Clone, Copy)]
enum Modifier {
    Table,
    Back,
    Gap,
    Dip,
}

enum TokenType {
    Modifier(Modifier),
    Op(Op),
}

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"[ \t]+")]
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
    #[regex(r"round|⁅")]
    Round,
    #[regex("sub|-")]
    Sub,
    #[regex(r"back|˜")]
    Back,
    #[regex(r"dup|\.")]
    Dup,
    #[regex(r"gap")]
    Gap,
    #[regex(r"dip")]
    Dip,
    #[regex(r"pop")]
    Pop,
    #[regex(r"floor|⌊")]
    Floor,
    #[regex(r"ceil|⌈")]
    Ceil,
    #[regex("#[^\n]*")]
    Comment,
    #[token("\n")]
    LineBreak,
    #[regex(r"[0-9]+(\.[0-9]+)?", |lex| lex.slice().parse::<f32>().unwrap())]
    Value(f32),
}

fn parse(token: Token) -> Option<TokenType> {
    Some(match token {
        Token::Eq => TokenType::Op(Op::BasicOp(OpType::Eq)),
        Token::Abs => TokenType::Op(Op::BasicOp(OpType::Abs)),
        Token::Add => TokenType::Op(Op::BasicOp(OpType::Add)),
        Token::Mul => TokenType::Op(Op::BasicOp(OpType::Mul)),
        Token::Div => TokenType::Op(Op::BasicOp(OpType::Div)),
        Token::Sin => TokenType::Op(Op::BasicOp(OpType::Sin)),
        Token::Max => TokenType::Op(Op::BasicOp(OpType::Max)),
        Token::Sub => TokenType::Op(Op::BasicOp(OpType::Sub)),
        Token::Round => TokenType::Op(Op::BasicOp(OpType::Round)),
        Token::Floor => TokenType::Op(Op::BasicOp(OpType::Floor)),
        Token::Ceil => TokenType::Op(Op::BasicOp(OpType::Ceil)),
        Token::Dup => TokenType::Op(Op::BasicOp(OpType::Dup)),
        Token::Pop => TokenType::Op(Op::BasicOp(OpType::Pop)),
        Token::Table => TokenType::Modifier(Modifier::Table),
        Token::Back => TokenType::Modifier(Modifier::Back),
        Token::Gap => TokenType::Modifier(Modifier::Gap),
        Token::Dip => TokenType::Modifier(Modifier::Dip),
        Token::Range => TokenType::Op(Op::Range),
        Token::Rev => TokenType::Op(Op::Rev),
        Token::Value(value) => TokenType::Op(Op::Value(value)),
        Token::Comment | Token::LineBreak => return None,
    })
}

fn parse_code(code: &str, left_to_right: bool) -> Result<Vec<CodeBlock>, Range<usize>> {
    let mut parsed_code = Vec::new();
    for line in code.lines() {
        let blocks = parse_code_blocks(Token::lexer(line).spanned(), left_to_right)?;
        parsed_code.extend_from_slice(&blocks);
    }

    Ok(parsed_code)
}

fn parse_code_blocks(
    mut lexer: logos::SpannedIter<Token>,
    left_to_right: bool,
) -> Result<Vec<CodeBlock>, Range<usize>> {
    let mut blocks = Vec::new();

    while let Some(block) = parse_code_block(&mut lexer)? {
        blocks.push(block);
    }

    if !left_to_right {
        blocks.reverse();
    }

    Ok(blocks)
}

fn parse_code_block(
    lexer: &mut logos::SpannedIter<Token>,
) -> Result<Option<CodeBlock>, Range<usize>> {
    let mut modifiers = Vec::new();
    let mut span_start = None;

    for (token, span) in lexer {
        let token = match token {
            Ok(token) => token,
            Err(()) => return Err(span),
        };

        span_start = Some(span_start.unwrap_or(span.start));

        match parse(token) {
            None => {}
            Some(TokenType::Modifier(modifier)) => modifiers.push(modifier),
            Some(TokenType::Op(op)) => {
                return Ok(Some(CodeBlock {
                    modifiers,
                    span: span_start.unwrap_or(span.start)..span.end,
                    code: FunctionOrOp::Op(op),
                }));
            }
        }
    }

    Ok(None)
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
            Self::Other { .. } => "Coord(1, 1, 1, 1)".to_string(),
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
    modifier_expecting_op: Option<ModifierAccess>,
    upcoming_admin_commands: Vec<String>,
    delayed_pops: Vec<usize>,
}

impl Parser {
    fn pop(&mut self) -> Result<StackItem, ()> {
        if self.delayed_pops.len() >= self.stack.len() {
            return Err(());
        }
        self.stack.pop().ok_or(())
    }

    fn peek(&self, depth: usize) -> &StackItem {
        self.stack.get(self.stack.len() - 1 - depth).unwrap()
    }

    fn modifier_expecting_op(&mut self) -> Option<ModifierAccess> {
        self.modifier_expecting_op.take()
    }

    fn finish_write(&mut self) -> Result<(), ()> {
        let mut admin_commands = std::mem::take(&mut self.upcoming_admin_commands);
        let mut write_commands = Vec::new();
        let mut command_to_buffer_index = HashMap::new();

        let mut temporary_buffer_index = 50;

        for (stack_pos, item) in self.stack.iter_mut().enumerate() {
            if !item.is_output() {
                continue;
            }

            if let Some(index) = command_to_buffer_index.get(&item.str("thread_id")) {
                self.upcoming_admin_commands
                    .push(format!("buffers[{}] = buffers[{}]", stack_pos, index));
            } else {
                command_to_buffer_index.insert(item.str("thread_id"), temporary_buffer_index);

                admin_commands.push(format!(
                    "allocate({}, {})",
                    temporary_buffer_index, self.output_size
                ));
                write_commands.push(format!(
                    "thread_id = index_to_coord(thread.x, buffers[{}].size)",
                    temporary_buffer_index
                ));
                write_commands.push(format!(
                    "write_to_buffer({}, thread_id, {})",
                    temporary_buffer_index,
                    item.str("thread_id")
                ));
                self.upcoming_admin_commands.push(format!(
                    "buffers[{}] = buffers[{}]",
                    stack_pos, temporary_buffer_index
                ));
                temporary_buffer_index += 1;
            }

            *item = StackItem::Buffer(stack_pos as u32);
        }

        if !admin_commands.is_empty() {
            self.functions.push(admin_commands);
        }
        if !write_commands.is_empty() {
            self.functions.push(write_commands);
        }

        for i in self.delayed_pops.drain(..) {
            self.stack.swap_remove(i);
            self.upcoming_admin_commands
                .push(format!("buffers[{}] = buffers[{}]", i - 1, i));
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<(), ()> {
        self.finish_write()?;
        for (i, item) in self.stack.iter().enumerate() {
            match item {
                StackItem::Other { str, is_output } => {
                    assert!(!is_output);
                    self.upcoming_admin_commands
                        .push(format!("allocate({}, Coord(1,1,1,1))", i));
                    self.upcoming_admin_commands
                        .push(format!("write_to_buffer({}, Coord(0,0,0,0), {})", i, str));
                }
                StackItem::Buffer(_) => {}
            }
        }
        if !self.upcoming_admin_commands.is_empty() {
            self.finish_write()?;
        }
        Ok(())
    }

    fn handle_monadic<F: Fn(String) -> String>(&mut self, func: F) -> Result<(), ()> {
        let v = self.pop()?;
        let modifier_expecting_op = self.modifier_expecting_op();
        let is_output = v.is_output() || modifier_expecting_op.is_some();
        let access = match modifier_expecting_op {
            Some(ModifierAccess::Normal) | None => "thread_id",
            Some(ModifierAccess::Transpose) => {
                "Coord(thread_id[1], thread_id[0], thread_id[2], thread_id[3])"
            }
        };
        self.stack.push(StackItem::Other {
            str: func(v.str(access)),
            is_output,
        });
        Ok(())
    }

    fn handle_diadic<F: Fn(String, String) -> String>(
        &mut self,
        back: bool,
        func: F,
    ) -> Result<(), ()> {
        let mut a = self.pop()?;
        let mut b = self.pop()?;
        if back {
            std::mem::swap(&mut a, &mut b);
        }
        let modifier_expecting_op = self.modifier_expecting_op();
        let is_output = a.is_output() || b.is_output() || modifier_expecting_op.is_some();
        let first_access = match modifier_expecting_op {
            Some(ModifierAccess::Normal) | None => "thread_id",
            Some(ModifierAccess::Transpose) => {
                "Coord(thread_id[1], thread_id[0], thread_id[2], thread_id[3])"
            }
        };
        self.stack.push(StackItem::Other {
            str: func(b.str(first_access), a.str("thread_id")),
            is_output,
        });
        Ok(())
    }

    fn handle_operation(&mut self, back: bool, operation: OpType) -> Result<(), ()> {
        match operation {
            OpType::Dup => {
                let top = self.stack.last().ok_or(())?.clone();
                self.stack.push(top);
                Ok(())
            }
            OpType::Pop => {
                self.pop()?;
                Ok(())
            }
            OpType::Add => self.handle_diadic(back, |a, b| format!("({} + {})", a, b)),
            OpType::Eq => self.handle_diadic(back, |a, b| format!("f32({} == {})", a, b)),
            OpType::Div => self.handle_diadic(back, |a, b| format!("({} / {})", a, b)),
            OpType::Mul => self.handle_diadic(back, |a, b| format!("({} * {})", a, b)),
            OpType::Sub => self.handle_diadic(back, |a, b| format!("({} - {})", a, b)),
            OpType::Max => self.handle_diadic(back, |a, b| format!("max({}, {})", a, b)),
            OpType::Sin => self.handle_monadic(|v| format!("sin({})", v)),
            OpType::Abs => self.handle_monadic(|v| format!("abs({})", v)),
            OpType::Round => self.handle_monadic(|v| format!("round({})", v)),
            OpType::Floor => self.handle_monadic(|v| format!("floor({})", v)),
            OpType::Ceil => self.handle_monadic(|v| format!("ceil({})", v)),
        }
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let input = std::env::args().nth(1).unwrap();
        pollster::block_on(async move {
            let output = Context::new()
                .await
                .run_string_and_get_string_output(&input, false)
                .await;
            for line in output.lines() {
                println!("{}", line);
            }
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Trace).expect("could not initialize logger");
        use leptos::prelude::*;
        wasm_bindgen_futures::spawn_local(async {
            let context = std::rc::Rc::new(Context::new().await);
            let (text, set_text) = create_signal(String::new());
            leptos::mount::mount_to_body(move || {
                view! {
                    <textarea
                        //prop:value=text
                        on:input=move |input| {
                            let context = context.clone();
                            wasm_bindgen_futures::spawn_local(async move {
                                set_text.set(context.run_string_and_get_string_output(&event_target_value(&input), false).await);
                            });
                            //log::info!("{:?}", event_target_value(&input))
                        }
                        //on:keydown=on_key_down
                        //placeholder="Type your message and press Shift+Enter to submit"
                        //class="textarea"
                    />
                    <br/>
                    <textarea prop:value=text/>

                }
            });
        });
    }
}

#[derive(Debug, PartialEq)]
struct ReadBackValue {
    size: [u32; 4],
    values: Vec<f32>,
}

impl ReadBackValue {
    fn scalar(value: f32) -> Self {
        Self {
            size: [1; 4],
            values: vec![value],
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

#[cfg(test)]
fn assert_output(string: &str, output: Vec<ReadBackValue>) {
    pollster::block_on(async {
        let context = Context::new().await;
        assert_eq!(context.run_string(string, false).await.unwrap(), output);
    })
}

#[test]
fn identity_matrix_cross() {
    assert_output(
        "max rev dup table = dup range 5",
        vec![ReadBackValue {
            size: [5, 5, 1, 1],
            #[rustfmt::skip]
            values: vec![
                1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }],
    );
}

#[test]
fn scalar_values() {
    assert_output(
        "16.6 3",
        vec![ReadBackValue::scalar(3.0), ReadBackValue::scalar(16.6)],
    );
}
