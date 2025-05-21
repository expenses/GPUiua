use rand::Rng;

fn generate_module(code: Vec<FunctionOrOp>) -> ShaderModule {
    let mut rng = rand::rngs::StdRng::from_seed([0; 32]);

    let (dag, final_stack) = parse_code_to_dag(code);
    let mut full_eval_to: BTreeMap<daggy::NodeIndex, usize> = final_stack
        .iter()
        .enumerate()
        .map(|(i, &index)| (index, i))
        .collect();
    let mut size_to_buffer: HashMap<Size, usize> = final_stack
        .iter()
        .enumerate()
        .map(|(i, &index)| (dag[index].size, i))
        .collect();
    let mut dummy_buffer_index = final_stack.len();

    {
        let mut walk_stack = final_stack.clone();
        while let Some(index) = walk_stack.pop() {
            for (_, parent) in dag.parents(index).iter(&dag) {
                walk_stack.push(parent);
            }

            let node = &dag[index];
            match node {
                Node {
                    op: NodeOp::Rand,
                    size: Size::Scalar,
                }
                | Node {
                    op: NodeOp::Reduce(_),
                    ..
                } => {
                    full_eval_to.insert(index, dummy_buffer_index);
                    size_to_buffer.insert(node.size, dummy_buffer_index);
                    dummy_buffer_index += 1;
                }
                _ => {}
            }
        }
    }

    let mut code_builder = CodeBuilder {
        functions: vec![FunctionPair {
            admin_lines: vec![],
            work_lines: Default::default(),
            dispatching_on_buffer_index: None,
        }],
        aux_functions: Default::default(),
        next_dispatch_index: 0,
        size_to_function_index: Default::default(),
    };

    let mut node_to_allocation = HashMap::new();

    for (&item, &i) in &full_eval_to {
        let size = evaluate_size(
            &mut EvaluationContext {
                dag: &dag,
                functions: &mut code_builder.aux_functions,
                full_eval_to: &full_eval_to,
                size_to_buffer: &size_to_buffer,
                rng: &mut rng,
            },
            true,
            item,
        );
        code_builder.functions[0]
            .admin_lines
            .push(format!("allocate({}, {})", i, size));
        node_to_allocation.insert(item, i);

        match &dag[item].size {
            Size::Scalar => {
                code_builder.functions[0]
                    .admin_lines
                    .push("random_seed += 1".to_string());
                code_builder.functions[0].admin_lines.push(format!(
                    "write_to_buffer({}, Coord(0,0,0,0), {})",
                    i,
                    evaluate_scalar(
                        &mut EvaluationContext {
                            dag: &dag,
                            functions: &mut code_builder.aux_functions,
                            full_eval_to: &full_eval_to,
                            size_to_buffer: &size_to_buffer,

                            rng: &mut rng,
                        },
                        true,
                        item
                    )
                ));
            }
            _ => {
                let dispatch_index = *code_builder
                    .size_to_function_index
                    .entry(dag[item].size)
                    .or_insert_with(|| {
                        let dispatch_index = code_builder.next_dispatch_index;
                        code_builder.next_dispatch_index += 1;

                        code_builder.functions[dispatch_index]
                            .admin_lines
                            .push(format!("dispatch_for_buffer({})", i));

                        code_builder.functions.push(Default::default());
                        dispatch_index
                    });

                code_builder.functions[dispatch_index].dispatching_on_buffer_index = Some(i);

                code_builder.functions[dispatch_index]
                    .work_lines
                    .push(format!(
                        "write_to_buffer({}, thread_coord, {})",
                        i,
                        evaluate_scalar(
                            &mut EvaluationContext {
                                dag: &dag,
                                functions: &mut code_builder.aux_functions,
                                full_eval_to: &full_eval_to,
                                size_to_buffer: &size_to_buffer,

                                rng: &mut rng,
                            },
                            true,
                            item
                        )
                    ));
            }
        }
    }

    for (i, item) in final_stack.iter().enumerate() {
        let allocation = node_to_allocation[item];
        if i == allocation {
            continue;
        }
        code_builder.functions[0]
            .admin_lines
            .push(format!("buffers[{}] = buffers[{}]", i, allocation));
    }

    let mut shader = include_str!("../out.wgsl").to_string();

    for (_, function) in code_builder.aux_functions {
        shader.push_str(&function);
        shader.push('\n');
    }

    let mut step_index = 0;

    for pair in &code_builder.functions {
        if pair.admin_lines.is_empty() && pair.work_lines.is_empty() {
            continue;
        }

        shader.push_str(&format!(
            "\n@compute @workgroup_size(1,1,1) fn step_{}() {{\n",
            step_index
        ));
        shader.push_str(&format!(
            "var random_seed = u32({});\n",
            rng.random::<u32>()
        ));
        for line in &pair.admin_lines {
            shader.push_str(&format!("{};\n", line));
        }
        shader.push_str("}\n");
        step_index += 1;
        shader.push_str(&format!("@compute @workgroup_size(64,1,1) fn step_{}(@builtin(global_invocation_id) thread: vec3<u32>) {{\n", step_index));
        if let Some(index) = pair.dispatching_on_buffer_index {
            shader.push_str(&format!("let dispatch_size = buffers[{}].size;\n", index));
            shader.push_str("let thread_coord = index_to_coord(thread.x, dispatch_size);\n");
            shader.push_str("if (coord_any_ge(thread_coord, dispatch_size)) {return;}\n");
            shader.push_str(&format!(
                "let random_seed = thread.x + {};\n",
                rng.random::<u32>()
            ));
        }

        for line in pair.work_lines.iter() {
            shader.push_str(&format!("{};\n", line));
        }
        shader.push_str("}\n");
        step_index += 1;
    }

    ShaderModule {
        code: shader,
        num_functions: step_index,
        final_stack_len: final_stack.len(),
    }
}

struct ShaderModule {
    code: String,
    num_functions: usize,
    final_stack_len: usize,
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
            size: 4 * 2,
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
        let module = generate_module(
            parse_code(string, left_to_right)
                .map_err(|(str, span)| format!("'{}' {:?}", str, span))?,
        );
        log::debug!("{}", module.code);
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
                [x, _, 1, 1] => {
                    for chunk in value.values.chunks(x as usize) {
                        output.push_str(&format!("{:?}\n", chunk));
                    }
                    output.push('\n');
                }
                _ => panic!(),
            }
        }

        output
    }
}

use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
};

use daggy::NodeIndex;
use logos::Logos;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MonadicOp {
    Sin,
    Round,
    Abs,
    Floor,
    Ceil,
    Not,
    Sqrt,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DyadicOp {
    Add,
    Mul,
    Div,
    Eq,
    Sub,
    Max,
    Gt,
    Ge,
    Lt,
    Le,
    Ne,
}

#[derive(Debug, Clone, Copy)]
enum StackOp {
    Dup,
    Pop,
    Ident,
}

#[derive(Clone, Debug)]
enum FunctionOrOp {
    Op(Op),
    Function {
        modifier: Modifier,
        code: Vec<FunctionOrOp>,
    },
}

impl FunctionOrOp {
    #[allow(unused)]
    fn stack_delta(&self) -> i32 {
        match self {
            Self::Op(Op::Monadic(_)) => 0,
            Self::Op(Op::Dyadic(_)) => -1,
            Self::Op(Op::Value(_)) => 1,
            Self::Op(Op::Rand) => 1,
            Self::Op(Op::Stack(StackOp::Dup)) => 1,
            Self::Op(Op::Stack(StackOp::Ident)) => 0,
            Self::Op(Op::Stack(StackOp::Pop)) => -1,
            Self::Op(Op::Len | Op::Rev | Op::Range) => 0,
            Self::Function { modifier, code } => {
                let modifier = match *modifier {
                    Modifier::Back => 0,
                    Modifier::Dip => 0,
                    Modifier::Table => 0,
                    Modifier::Gap => -1,
                    Modifier::By => 1,
                    Modifier::Reduce => 1,
                };

                modifier + code.iter().map(|op| op.stack_delta()).sum::<i32>()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Monadic(MonadicOp),
    Dyadic(DyadicOp),
    Stack(StackOp),
    Value(f32),
    Range,
    Rev,
    Rand,
    Len,
}

#[derive(Debug, Clone, Copy)]
enum Modifier {
    Table,
    Back,
    Gap,
    Dip,
    By,
    Reduce,
}

enum TokenType<'a> {
    Modifier(Modifier),
    Op(Op),
    AssignedOp(&'a str),
}

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"[ \t]+")]
enum Token<'source> {
    #[regex(r"[a-zA-Z_]+[a-zA-Z_0-9]*", priority = 0)]
    String(&'source str),
    #[regex(r"add|\+")]
    Add,
    #[regex(r"mul|\*")]
    Mul,
    #[regex(r"rand|⚂")]
    Rand,
    #[regex(r"div|÷")]
    Div,
    #[token("eq")]
    Eq,
    #[token("=")]
    EqualSign,
    #[token("←")]
    Assignment,
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
    #[regex(r"not|¬")]
    Not,
    #[regex("sub|-")]
    Sub,
    #[regex(r"back|˜")]
    Back,
    #[regex(r"dup|\.")]
    Dup,
    #[regex(r"gap|⋅")]
    Gap,
    #[regex(r"dip|⊙")]
    Dip,
    #[regex(r"pop|◌")]
    Pop,
    #[regex(r"floor|⌊")]
    Floor,
    #[regex(r"ceil|⌈")]
    Ceil,
    #[regex(r"ident|∘")]
    Ident,
    #[regex(r"by|⊸")]
    By,
    #[regex(r"gt|>")]
    Gt,
    #[regex(r"ge|≥")]
    Ge,
    #[regex(r"lt|<")]
    Lt,
    #[regex(r"le|≤")]
    Le,
    #[regex(r"ne|≠")]
    Ne,
    #[regex(r"neg|¯")]
    Neg,
    #[regex(r"sqrt|√")]
    Sqrt,
    #[regex(r"len")]
    Len,
    #[regex(r"/")]
    Reduce,
    #[regex(r"[0-9]+(\.[0-9]+)?", |lex| lex.slice().parse::<f32>().unwrap())]
    Value(f32),
    #[token("(")]
    OpenParen,
    #[token(")")]
    CloseParen,
}

fn parse(token: Token) -> Option<TokenType> {
    Some(match token {
        Token::Abs => TokenType::Op(Op::Monadic(MonadicOp::Abs)),
        Token::Not => TokenType::Op(Op::Monadic(MonadicOp::Not)),
        Token::Neg => TokenType::Op(Op::Monadic(MonadicOp::Neg)),
        Token::Sin => TokenType::Op(Op::Monadic(MonadicOp::Sin)),
        Token::Ceil => TokenType::Op(Op::Monadic(MonadicOp::Ceil)),
        Token::Round => TokenType::Op(Op::Monadic(MonadicOp::Round)),
        Token::Floor => TokenType::Op(Op::Monadic(MonadicOp::Floor)),
        Token::Gt => TokenType::Op(Op::Dyadic(DyadicOp::Gt)),
        Token::Ge => TokenType::Op(Op::Dyadic(DyadicOp::Ge)),
        Token::Lt => TokenType::Op(Op::Dyadic(DyadicOp::Lt)),
        Token::Le => TokenType::Op(Op::Dyadic(DyadicOp::Le)),
        Token::Ne => TokenType::Op(Op::Dyadic(DyadicOp::Ne)),
        Token::Eq | Token::EqualSign => TokenType::Op(Op::Dyadic(DyadicOp::Eq)),
        Token::Add => TokenType::Op(Op::Dyadic(DyadicOp::Add)),
        Token::Mul => TokenType::Op(Op::Dyadic(DyadicOp::Mul)),
        Token::Div => TokenType::Op(Op::Dyadic(DyadicOp::Div)),
        Token::Max => TokenType::Op(Op::Dyadic(DyadicOp::Max)),
        Token::Sub => TokenType::Op(Op::Dyadic(DyadicOp::Sub)),
        Token::Sqrt => TokenType::Op(Op::Monadic(MonadicOp::Sqrt)),
        Token::Dup => TokenType::Op(Op::Stack(StackOp::Dup)),
        Token::Pop => TokenType::Op(Op::Stack(StackOp::Pop)),
        Token::Ident => TokenType::Op(Op::Stack(StackOp::Ident)),
        Token::By => TokenType::Modifier(Modifier::By),
        Token::Gap => TokenType::Modifier(Modifier::Gap),
        Token::Dip => TokenType::Modifier(Modifier::Dip),
        Token::Back => TokenType::Modifier(Modifier::Back),
        Token::Table => TokenType::Modifier(Modifier::Table),
        Token::Reduce => TokenType::Modifier(Modifier::Reduce),
        Token::Rev => TokenType::Op(Op::Rev),
        Token::Rand => TokenType::Op(Op::Rand),
        Token::Range => TokenType::Op(Op::Range),
        Token::Len => TokenType::Op(Op::Len),
        Token::Value(value) => TokenType::Op(Op::Value(value)),
        Token::String(string) => TokenType::AssignedOp(string),
        Token::OpenParen | Token::CloseParen | Token::Assignment => return None,
    })
}

fn parse_code(code: &str, left_to_right: bool) -> Result<Vec<FunctionOrOp>, (&str, Range<usize>)> {
    let mut parsed_code = Vec::new();
    let mut assignments: HashMap<&str, Vec<FunctionOrOp>> = HashMap::new();
    for line in code.lines() {
        let line = line
            .split_once('#')
            .map(|(before_comment, _)| before_comment)
            .unwrap_or(line);
        let mut lexer = Token::lexer(line).spanned().peekable();

        if let Some((Ok(Token::String(name)), span)) = lexer.peek().cloned() {
            if assignments.contains_key(name) {
                let blocks = parse_code_blocks(lexer, left_to_right, &assignments)
                    .map_err(|span| (line, span))?;
                parsed_code.extend_from_slice(&blocks);
            } else {
                let _ = lexer.next().unwrap();
                match lexer.next() {
                    Some((Ok(Token::EqualSign | Token::Assignment), _)) => {}
                    Some((_, span)) => return Err((line, span)),
                    None => {
                        return Err((line, span));
                    }
                }
                let blocks =
                    parse_code_blocks(lexer, true, &assignments).map_err(|span| (line, span))?;
                assignments.insert(name, blocks);
            }
        } else {
            let blocks = parse_code_blocks(lexer, left_to_right, &assignments)
                .map_err(|span| (line, span))?;
            parsed_code.extend_from_slice(&blocks);
        }
    }

    Ok(parsed_code)
}

fn parse_code_blocks<'a>(
    mut lexer: std::iter::Peekable<logos::SpannedIter<'a, Token<'a>>>,
    left_to_right: bool,
    assignments: &HashMap<&str, Vec<FunctionOrOp>>,
) -> Result<Vec<FunctionOrOp>, Range<usize>> {
    let mut blocks = Vec::new();

    loop {
        let parsed_blocks = parse_code_blocks_inner(&mut lexer, left_to_right, assignments)?;
        if parsed_blocks.is_empty() {
            break;
        }
        blocks.extend_from_slice(&parsed_blocks);
    }

    if !left_to_right {
        blocks.reverse();
    }

    Ok(blocks)
}

fn parse_code_blocks_inner<'a>(
    lexer: &mut std::iter::Peekable<logos::SpannedIter<'a, Token<'a>>>,
    left_to_right: bool,
    assignments: &HashMap<&str, Vec<FunctionOrOp>>,
) -> Result<Vec<FunctionOrOp>, Range<usize>> {
    let (token, span) = match lexer.next() {
        Some((Ok(token), span)) => (token, span),
        Some((Err(()), span)) => return Err(span),
        None => return Ok(vec![]),
    };

    match parse(token) {
        Some(TokenType::AssignedOp(name)) => Ok(assignments.get(name).unwrap().clone()),
        Some(TokenType::Op(op)) => Ok(vec![FunctionOrOp::Op(op)]),
        Some(TokenType::Modifier(modifier)) => {
            if let Some(&(Ok(Token::OpenParen), _)) = lexer.peek() {
                let _ = lexer.next();
                let mut code = Vec::new();
                loop {
                    if let Some(&(Ok(Token::CloseParen), _)) = lexer.peek() {
                        let _ = lexer.next();
                        if !left_to_right {
                            code.reverse();
                        }
                        return Ok(vec![FunctionOrOp::Function { modifier, code }]);
                    }

                    code.extend_from_slice(&match parse_code_blocks_inner(
                        lexer,
                        left_to_right,
                        assignments,
                    ) {
                        Ok(ops) if ops.is_empty() => return Err(span),
                        Ok(ops) => ops,
                        Err(span) => return Err(span),
                    })
                }
            } else {
                Ok(vec![FunctionOrOp::Function {
                    modifier,
                    code: match parse_code_blocks_inner(lexer, left_to_right, assignments) {
                        Ok(ops) if ops.is_empty() => return Err(span),
                        Ok(ops) => ops,
                        Err(span) => return Err(span),
                    },
                }])
            }
        }
        None => Err(span),
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum NodeOp {
    Monadic(MonadicOp),
    Dyadic { is_table: bool, op: DyadicOp },
    Range,
    Rev,
    Rand,
    Len,
    Value(ordered_float::OrderedFloat<f32>),
    ReduceResult,
    Reduce(daggy::NodeIndex),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct Node {
    op: NodeOp,
    size: Size,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
enum Size {
    Scalar,
    RangeOf(daggy::NodeIndex),
    MaxOf(daggy::NodeIndex, daggy::NodeIndex),
    TransposeSizeOf(daggy::NodeIndex, daggy::NodeIndex),
}

#[derive(Default)]
struct Dag {
    inner: daggy::Dag<Node, ()>,
    stack: Vec<daggy::NodeIndex>,
    duplicate_map: HashMap<(Node, Vec<daggy::NodeIndex>), daggy::NodeIndex>,
}

impl Dag {
    fn insert_node(&mut self, node: Node, parent_indices: Vec<daggy::NodeIndex>) {
        let index = if node.op == NodeOp::Rand {
            self.inner.add_node(node)
        } else {
            *self
                .duplicate_map
                .entry((node.clone(), parent_indices.clone()))
                .or_insert_with(|| self.inner.add_node(node))
        };
        for parent_index in parent_indices {
            self.inner.update_edge(parent_index, index, ()).unwrap();
        }
        self.stack.push(index);
    }
}

fn handle_op(op: FunctionOrOp, dag: &mut Dag, mut table_of_size: Option<Size>) {
    match op {
        FunctionOrOp::Function { modifier, code } => {
            let mut dipped = None;
            let mut reducing = None;

            match modifier {
                Modifier::Back => {
                    let x = dag.stack.pop().unwrap();
                    let y = dag.stack.pop().unwrap();
                    dag.stack.push(x);
                    dag.stack.push(y);
                }
                Modifier::Dip => {
                    dipped = Some(dag.stack.pop().unwrap());
                }
                Modifier::By => {
                    let stack_delta = code.iter().map(|code| code.stack_delta()).sum::<i32>();
                    match stack_delta {
                        1..=i32::MAX => {}
                        other => {
                            let index = (dag.stack.len() as i32 - 1 + other) as usize;
                            dag.stack.insert(index, *dag.stack.get(index).unwrap());
                        }
                    }
                }
                Modifier::Gap => {
                    dag.stack.pop().unwrap();
                }
                Modifier::Table => {
                    let x = dag.stack.last().unwrap();
                    let y = dag.stack.get(dag.stack.len() - 2).unwrap();
                    table_of_size = Some(Size::TransposeSizeOf(*x, *y));
                }
                Modifier::Reduce => {
                    let stack_delta = code.iter().map(|code| code.stack_delta()).sum::<i32>();
                    assert_eq!(stack_delta, -1);
                    let reducing_array = *dag.stack.last().unwrap();
                    let reducing_array_size = dag.inner[reducing_array].size;
                    reducing = Some(*dag.stack.last().unwrap());
                    dag.insert_node(
                        Node {
                            op: NodeOp::ReduceResult,
                            size: Size::Scalar,
                        },
                        vec![],
                    );
                    table_of_size = Some(reducing_array_size);
                }
            }

            for op in code {
                handle_op(op, dag, table_of_size);
            }

            if let Some(reducing) = reducing {
                let parent = dag.stack.pop().unwrap();
                dag.insert_node(
                    Node {
                        op: NodeOp::Reduce(reducing),
                        size: Size::Scalar,
                    },
                    vec![parent],
                );
            }

            if let Some(value) = dipped {
                dag.stack.push(value);
            }
        }
        FunctionOrOp::Op(Op::Rand) => dag.insert_node(
            Node {
                op: NodeOp::Rand,
                size: table_of_size.unwrap_or(Size::Scalar),
            },
            vec![],
        ),
        FunctionOrOp::Op(Op::Value(value)) => dag.insert_node(
            Node {
                op: NodeOp::Value(value.into()),
                size: table_of_size.unwrap_or(Size::Scalar),
            },
            vec![],
        ),
        FunctionOrOp::Op(Op::Stack(StackOp::Dup)) => {
            dag.stack.push(*dag.stack.last().unwrap());
        }
        FunctionOrOp::Op(Op::Stack(StackOp::Pop)) => {
            dag.stack.pop();
        }
        FunctionOrOp::Op(Op::Stack(StackOp::Ident)) => {
            // Potentially change size of the node on the top of the stack.
            let index = dag.stack.pop().unwrap();
            let mut node = dag.inner[index].clone();
            node.size = table_of_size.unwrap_or(node.size);
            dag.insert_node(node, vec![index]);
        }
        FunctionOrOp::Op(Op::Len) => {
            let parent_index = dag.stack.pop().unwrap();
            dag.insert_node(
                Node {
                    op: NodeOp::Len,
                    size: Size::Scalar,
                },
                vec![parent_index],
            );
        }
        FunctionOrOp::Op(Op::Range) => {
            let parent_index = dag.stack.pop().unwrap();
            dag.insert_node(
                Node {
                    op: NodeOp::Range,
                    size: Size::RangeOf(parent_index),
                },
                vec![parent_index],
            );
        }
        FunctionOrOp::Op(Op::Rev) => {
            let index = dag.stack.pop().unwrap();
            let size = dag.inner[index].size;
            dag.insert_node(
                Node {
                    op: NodeOp::Rev,
                    size,
                },
                vec![index],
            );
        }
        FunctionOrOp::Op(Op::Monadic(op)) => {
            let index = dag.stack.pop().unwrap();
            let size = dag.inner[index].size;
            dag.insert_node(
                Node {
                    op: NodeOp::Monadic(op),
                    size: table_of_size.unwrap_or(size),
                },
                vec![index],
            );
        }
        FunctionOrOp::Op(Op::Dyadic(op)) => {
            let x = dag.stack.pop().unwrap();
            let y = dag.stack.pop().unwrap();
            let x_val = &dag.inner[x];
            let y_val = &dag.inner[y];
            let node = Node {
                op: NodeOp::Dyadic {
                    is_table: matches!(table_of_size, Some(Size::TransposeSizeOf(_, _))),
                    op,
                },
                size: if let Some(size) = table_of_size {
                    size
                } else {
                    match (x_val.size, y_val.size) {
                        (Size::Scalar, Size::Scalar) => Size::Scalar,
                        (Size::RangeOf(x), Size::RangeOf(y)) if x == y => Size::RangeOf(x),
                        (Size::MaxOf(a, b), Size::MaxOf(c, d))
                            if (a == c && b == d) || (a == d && b == c) =>
                        {
                            Size::MaxOf(a, b)
                        }
                        (Size::TransposeSizeOf(a, b), Size::TransposeSizeOf(c, d))
                            if (a == c && b == d) || (a == d && b == c) =>
                        {
                            Size::TransposeSizeOf(a, b)
                        }
                        (Size::Scalar, other) | (other, Size::Scalar) => other,
                        // give up
                        _ => Size::MaxOf(x, y),
                    }
                },
            };
            dag.insert_node(node, vec![x, y]);
        }
    }
}

fn parse_code_to_dag(code: Vec<FunctionOrOp>) -> (daggy::Dag<Node, ()>, Vec<daggy::NodeIndex>) {
    let mut dag = Dag::default();

    for op in code {
        handle_op(op, &mut dag, None);
    }

    (dag.inner, dag.stack)
}

use daggy::Walker;
use rand::SeedableRng;

struct EvaluationContext<'a, R> {
    dag: &'a daggy::Dag<Node, ()>,
    functions: &'a mut AuxFunctions,
    full_eval_to: &'a BTreeMap<daggy::NodeIndex, usize>,
    size_to_buffer: &'a HashMap<Size, usize>,
    rng: &'a mut R,
}

fn evaluate_scalar<R: Rng>(
    context: &mut EvaluationContext<R>,
    top_level_eval: bool,
    index: NodeIndex,
) -> String {
    if !top_level_eval {
        if let Some(buffer_index) = context.full_eval_to.get(&index) {
            return if context.dag[index].size == Size::Scalar {
                format!("read_buffer({}, Coord(0,0,0,0))", buffer_index)
            } else {
                format!("read_buffer({}, thread_coord)", buffer_index)
            };
        }
    }

    match &context.dag[index].op {
        NodeOp::Reduce(reducing) => {
            let function = format!(
                "fn reduce_{}() -> f32 {{
                    var thread_coord = Coord(0, 0, 0, 0);
                    let dispatch_size = {};
                    var thread = vec3(coord_to_index(thread_coord, dispatch_size), 0, 0);
                    var reduction = {};
                    for (thread_coord[0] = 1; thread_coord[0] < dispatch_size[0]; thread_coord[0]++) {{
                        thread = vec3(coord_to_index(thread_coord, dispatch_size), 0, 0);
                        var random_seed = u32({}) + thread.x;
                        reduction = {};
                    }}
                    return reduction;
                }}",
                index.index(),
                evaluate_size(context, false, *reducing),
                evaluate_scalar(context, false, *reducing),
                context.rng.random::<u32>(),
                evaluate_scalar(
                    context,
                    false,
                    context.dag.parents(index).walk_next(context.dag).unwrap().1
                )
            );
            context.functions.insert(index, function);
            format!("reduce_{}()", index.index())
        }
        NodeOp::ReduceResult => "reduction".to_string(),
        NodeOp::Len => {
            let size = evaluate_size(
                context,
                false,
                context.dag.parents(index).walk_next(context.dag).unwrap().1,
            );
            format!("f32({}[0])", size)
        }
        NodeOp::Monadic(op) => format!(
            "{}({})",
            format!("{:?}", op).to_lowercase(),
            evaluate_scalar(
                context,
                false,
                context.dag.parents(index).walk_next(context.dag).unwrap().1
            )
        ),
        NodeOp::Dyadic { op, is_table } => {
            let mut parents = context.dag.parents(index);
            let mut arg_0 =
                evaluate_scalar(context, false, parents.walk_next(context.dag).unwrap().1);
            let arg_1 = parents
                .walk_next(context.dag)
                .map(|parent| evaluate_scalar(context, false, parent.1))
                .unwrap_or_else(|| arg_0.clone());

            if *is_table {
                context.functions.insert(index, format!(
                    "fn table_{}(thread_coord: Coord, dispatch_size: Coord) -> f32 {{return {};}}",
                    index.index(),
                    arg_0
                ));
                arg_0 = format!(
                    "table_{}(coord_transpose(thread_coord), dispatch_size)",
                    index.index()
                );
            }

            format!(
                "{}({}, {})",
                format!("{:?}", op).to_lowercase(),
                arg_0,
                arg_1
            )
        }
        NodeOp::Value(value) => format!("f32({})", value),
        NodeOp::Range => "f32(thread_coord[0])".to_string(),
        NodeOp::Rand => "random_uniform(random_seed)".to_string(),
        NodeOp::Rev => {
            let function = format!(
                "fn rev_{}(thread_coord: Coord, dispatch_size: Coord, thread: vec3<u32>) -> f32 {{
                    var random_seed = u32({}) + thread.x;
                    return {};
                }}",
                index.index(),
                context.rng.random::<u32>(),
                evaluate_scalar(
                    context,
                    false,
                    context.dag.parents(index).walk_next(context.dag).unwrap().1
                )
            );
            context.functions.insert(index, function);
            format!(
                "rev_{}(coord_reverse(thread_coord, dispatch_size), dispatch_size, thread)",
                index.index(),
            )
        }
    }
}

type AuxFunctions = HashMap<daggy::NodeIndex, String>;

fn evaluate_size<R: Rng>(
    context: &mut EvaluationContext<R>,
    top_level_eval: bool,
    index: NodeIndex,
) -> String {
    dbg!(&context.full_eval_to);

    if !top_level_eval {
        if let Some(buffer_index) = context.size_to_buffer.get(&context.dag[index].size) {
            return format!("buffers[{}].size", buffer_index);
        }
    }

    match &context.dag[index].size {
        Size::RangeOf(range) => {
            format!(
                "Coord(u32({}), 1, 1, 1)",
                evaluate_scalar(context, false, *range)
            )
        }
        Size::Scalar => "Coord(1,1,1,1)".to_string(),
        Size::TransposeSizeOf(a, b) => format!(
            "coord_max({}, coord_transpose({}))",
            evaluate_size(context, false, *a),
            evaluate_size(context, false, *b)
        ),
        _ => panic!(),
    }
}

#[derive(Default, Debug)]
struct FunctionPair {
    admin_lines: Vec<String>,
    work_lines: Vec<String>,
    dispatching_on_buffer_index: Option<usize>,
}

struct CodeBuilder {
    aux_functions: AuxFunctions,
    functions: Vec<FunctionPair>,
    next_dispatch_index: usize,
    size_to_function_index: HashMap<Size, usize>,
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

#[cfg(test)]
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
        "
        ⇡5  # 0 through 4
        ⊞=. # Identity matrix
        ⊸⇌  # Reverse
        ↥   # Max
        ",
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
fn assignments() {
    assert_output(
        "
            xyz = +5
            xyz 5
        ",
        vec![ReadBackValue::scalar(10.0)],
    );
    assert_output(
        "
            div_new = div
            back div_new 10 1
        ",
        vec![ReadBackValue::scalar(10.0)],
    );
}

#[test]
fn scalar_values() {
    assert_output(
        "16.6 3",
        vec![ReadBackValue::scalar(3.0), ReadBackValue::scalar(16.6)],
    );
}

#[test]
fn regular_ident() {
    assert_output("ident 3", vec![ReadBackValue::scalar(3.0)]);
}

#[test]
fn table_ident() {
    assert_output(
        "⊞⋅⋅∘ . ⇡ . 3",
        vec![ReadBackValue {
            size: [3, 3, 1, 1],
            values: vec![3.0; 9],
        }],
    );
}

#[test]
fn identical_range_after_table() {
    assert_output(
        "range 3 table max . range 3",
        vec![
            ReadBackValue {
                size: [3, 3, 1, 1],
                #[rustfmt::skip]
                values: vec![
                        0.0, 1.0, 2.0,
                        1.0, 1.0, 2.0,
                        2.0, 2.0, 2.0
                ],
            },
            ReadBackValue {
                size: [3, 1, 1, 1],
                values: vec![0.0, 1.0, 2.0],
            },
        ],
    );
}

#[test]
fn table_double_gap() {
    assert_output(
        "table gap gap 5 . range 2",
        vec![ReadBackValue {
            size: [2, 2, 1, 1],
            values: vec![5.0, 5.0, 5.0, 5.0],
        }],
    );
}

#[test]
fn table_pop_in_parens() {
    assert_output(
        "table (5 pop pop) . range 2",
        vec![ReadBackValue {
            size: [2, 2, 1, 1],
            values: vec![5.0, 5.0, 5.0, 5.0],
        }],
    );
}

#[test]
fn reduce_mul() {
    assert_output("/*  + 1 range 3", vec![ReadBackValue::scalar(6.0)]);
}

#[test]
fn by_modifier() {
    assert_output(
        "⊸÷ 2 5",
        vec![ReadBackValue::scalar(5.0), ReadBackValue::scalar(2.5)],
    );
    assert_output(
        "
        tri ← ++
        ⊸tri 1 2 3
        ",
        vec![ReadBackValue::scalar(3.0), ReadBackValue::scalar(6.0)],
    );
    assert_output(
        "⊸5 1",
        vec![ReadBackValue::scalar(1.0), ReadBackValue::scalar(5.0)],
    );
}

#[test]
fn function_delta() {
    assert_eq!(
        FunctionOrOp::Function {
            modifier: Modifier::Table,
            code: vec![FunctionOrOp::Op(Op::Dyadic(DyadicOp::Eq))]
        }
        .stack_delta(),
        -1
    );
    assert_eq!(FunctionOrOp::Op(Op::Dyadic(DyadicOp::Eq)).stack_delta(), -1)
}

#[test]
fn deterministic_rng() {
    assert_output(
        "rand rand",
        vec![
            ReadBackValue::scalar(0.74225104),
            ReadBackValue::scalar(0.96570766),
        ],
    );
}

#[test]
fn rng_table() {
    assert_output(
        "table gap gap rand . range 3",
        vec![ReadBackValue {
            size: [3, 3, 1, 1],
            values: vec![
                0.4196651, 0.7859788, 0.84304845, 0.56487226, 0.15387845, 0.9303609, 0.51091456,
                0.705348, 0.974491,
            ],
        }],
    );
}

#[test]
fn rand_in_rev() {
    assert_output(
        "rev rev table gap gap rand . range 3",
        vec![ReadBackValue {
            size: [3, 3, 1, 1],
            values: vec![
                0.4196651, 0.7859788, 0.84304845, 0.56487226, 0.15387845, 0.9303609, 0.51091456,
                0.705348, 0.974491,
            ],
        }],
    );
}

#[test]
fn rand_average() {
    assert_output(
        "back div / (+ rand  dip pop)  range . 10",
        vec![ReadBackValue::scalar(0.5744712)],
    );
}
