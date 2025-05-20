fn generate_module(code: Vec<FunctionOrOp>) -> ShaderModule {
    let (dag, final_stack) = parse_code_to_dag(code);
    let full_eval_to: HashMap<daggy::NodeIndex, usize> = final_stack
        .iter()
        .enumerate()
        .map(|(i, &index)| (index, i))
        .collect();

    let mut code_builder = CodeBuilder {
        functions: vec![FunctionPair {
            admin_lines: Vec::new(),
            work_lines: Vec::new(),
        }],
        aux_functions: Default::default(),
        next_dispatch_index: 0,
        size_to_function_index: Default::default(),
    };

    let mut node_to_allocation = HashMap::new();

    for (i, item) in final_stack.iter().enumerate().rev() {
        if let Some(allocation) = node_to_allocation.get(item) {
            code_builder.functions[0]
                .admin_lines
                .push(format!("buffers[{}] = buffers[{}]", i, allocation));
        } else {
            let size = evaluate_size(&dag, &mut code_builder.aux_functions, &full_eval_to, *item);
            code_builder.functions[0]
                .admin_lines
                .push(format!("allocate({}, {})", i, size));
            node_to_allocation.insert(*item, i);

            match &dag[*item].size {
                Size::Scalar => {
                    code_builder.functions[0].admin_lines.push(format!(
                        "write_to_buffer({}, Coord(0,0,0,0), {})",
                        i,
                        evaluate_scalar(
                            &dag,
                            &mut code_builder.aux_functions,
                            &full_eval_to,
                            true,
                            *item
                        )
                    ));
                }
                _ => {
                    let dispatch_index = *code_builder
                        .size_to_function_index
                        .entry(dag[*item].size)
                        .or_insert_with(|| {
                            let dispatch_index = code_builder.next_dispatch_index;
                            code_builder.next_dispatch_index += 1;

                            code_builder.functions[dispatch_index]
                                .admin_lines
                                .push(format!("dispatch_for_buffer({})", i));
                            code_builder.functions[dispatch_index]
                                .work_lines
                                .push(format!("let dispatch_size = buffers[{}].size", i));
                            code_builder.functions[dispatch_index]
                                .work_lines
                                .push(format!(
                                    "let thread_coord = index_to_coord(thread.x, dispatch_size)"
                                ));

                            code_builder.functions.push(Default::default());
                            dispatch_index
                        });

                    code_builder.functions[dispatch_index]
                        .work_lines
                        .push(format!(
                            "write_to_buffer({}, thread_coord, {})",
                            i,
                            evaluate_scalar(
                                &dag,
                                &mut code_builder.aux_functions,
                                &full_eval_to,
                                true,
                                *item
                            )
                        ));
                }
            }
        }
    }

    let mut shader = include_str!("../out.wgsl").to_string();

    for function in code_builder.aux_functions {
        shader.push_str(&function);
        shader.push_str("\n");
    }

    let mut step_index = 0;

    for pair in &code_builder.functions {
        if pair.admin_lines.is_empty() && pair.work_lines.is_empty() {
            continue;
        }

        shader.push_str(&format!(
            "@compute @workgroup_size(1,1,1) fn step_{}() {{\n",
            step_index
        ));
        for line in &pair.admin_lines {
            shader.push_str(&format!("{};\n", line));
        }
        shader.push_str("}\n");
        step_index += 1;
        shader.push_str(&format!("@compute @workgroup_size(64,1,1) fn step_{}(@builtin(global_invocation_id) thread: vec3<u32>) {{\n", step_index));
        for line in &pair.work_lines {
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
            generate_module(parse_code(string, left_to_right).map_err(|err| format!("{:?}", err))?);
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
                    output.push_str("\n");
                }
                _ => panic!(),
            }
        }

        output
    }
}

use std::{
    collections::{HashMap, HashSet},
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DiadicOp {
    Add,
    Mul,
    Div,
    Eq,
    Sub,
    Max,
}

#[derive(Debug, Clone, Copy)]
enum StackOp {
    Dup,
    Pop,
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
    fn stack_delta(&self) -> i32 {
        match self {
            Self::Op(Op::Monadic(_)) => 0,
            Self::Op(Op::Diadic(_)) => -1,
            Self::Op(Op::Value(_)) => 1,
            Self::Op(Op::Stack(StackOp::Dup)) => 1,
            Self::Op(Op::Stack(StackOp::Pop)) => -1,
            Self::Op(Op::Rev) => 0,
            Self::Op(Op::Range) => 0,
            Self::Function { modifier, code } => {
                let modifier = match *modifier {
                    Modifier::Back => 0,
                    Modifier::Dip => 0,
                    Modifier::Table => 0,
                    Modifier::Gap => -1,
                };

                modifier + code.iter().map(|op| op.stack_delta()).sum::<i32>()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Monadic(MonadicOp),
    Diadic(DiadicOp),
    Stack(StackOp),
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
    #[regex(r"[0-9]+(\.[0-9]+)?", |lex| lex.slice().parse::<f32>().unwrap())]
    Value(f32),
    #[token("(")]
    OpenParen,
    #[token(")")]
    CloseParen,
}

fn parse(token: Token) -> Option<TokenType> {
    Some(match token {
        Token::Eq => TokenType::Op(Op::Diadic(DiadicOp::Eq)),
        Token::Abs => TokenType::Op(Op::Monadic(MonadicOp::Abs)),
        Token::Add => TokenType::Op(Op::Diadic(DiadicOp::Add)),
        Token::Mul => TokenType::Op(Op::Diadic(DiadicOp::Mul)),
        Token::Div => TokenType::Op(Op::Diadic(DiadicOp::Div)),
        Token::Sin => TokenType::Op(Op::Monadic(MonadicOp::Sin)),
        Token::Max => TokenType::Op(Op::Diadic(DiadicOp::Max)),
        Token::Sub => TokenType::Op(Op::Diadic(DiadicOp::Sub)),
        Token::Round => TokenType::Op(Op::Monadic(MonadicOp::Round)),
        Token::Floor => TokenType::Op(Op::Monadic(MonadicOp::Floor)),
        Token::Ceil => TokenType::Op(Op::Monadic(MonadicOp::Ceil)),
        Token::Dup => TokenType::Op(Op::Stack(StackOp::Dup)),
        Token::Pop => TokenType::Op(Op::Stack(StackOp::Pop)),
        Token::Table => TokenType::Modifier(Modifier::Table),
        Token::Back => TokenType::Modifier(Modifier::Back),
        Token::Gap => TokenType::Modifier(Modifier::Gap),
        Token::Dip => TokenType::Modifier(Modifier::Dip),
        Token::Range => TokenType::Op(Op::Range),
        Token::Rev => TokenType::Op(Op::Rev),
        Token::Value(value) => TokenType::Op(Op::Value(value)),
        Token::Comment | Token::OpenParen | Token::CloseParen => return None,
    })
}

fn parse_code(code: &str, left_to_right: bool) -> Result<Vec<FunctionOrOp>, Range<usize>> {
    let mut parsed_code = Vec::new();
    for line in code.lines() {
        let blocks = parse_code_blocks(Token::lexer(line).spanned().peekable(), left_to_right)?;
        parsed_code.extend_from_slice(&blocks);
    }

    Ok(parsed_code)
}

fn parse_code_blocks(
    mut lexer: std::iter::Peekable<logos::SpannedIter<Token>>,
    left_to_right: bool,
) -> Result<Vec<FunctionOrOp>, Range<usize>> {
    let mut blocks = Vec::new();

    while let Some(block) = parse_code_block(&mut lexer, left_to_right)? {
        blocks.push(block);
    }

    if !left_to_right {
        blocks.reverse();
    }

    Ok(blocks)
}

fn parse_code_block(
    lexer: &mut std::iter::Peekable<logos::SpannedIter<Token>>,
    left_to_right: bool,
) -> Result<Option<FunctionOrOp>, Range<usize>> {
    let (token, span) = match lexer.next() {
        Some((Ok(token), span)) => (token, span),
        Some((Err(()), span)) => return Err(span),
        None => return Ok(None),
    };

    match parse(token) {
        Some(TokenType::Op(op)) => return Ok(Some(FunctionOrOp::Op(op))),
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
                        return Ok(Some(FunctionOrOp::Function { modifier, code }));
                    }

                    code.push(match parse_code_block(lexer, left_to_right) {
                        Ok(None) => return Err(span),
                        Ok(Some(op)) => op,
                        Err(span) => return Err(span),
                    })
                }
            } else {
                return Ok(Some(FunctionOrOp::Function {
                    modifier,
                    code: vec![match parse_code_block(lexer, left_to_right) {
                        Ok(None) => return Err(span),
                        Ok(Some(op)) => op,
                        Err(span) => return Err(span),
                    }],
                }));
            }
        }
        None => return Err(span),
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum NodeOp {
    Monadic(MonadicOp),
    Diadic { is_table: bool, op: DiadicOp },
    Range,
    Rev,
    Value(ordered_float::OrderedFloat<f32>),
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

struct Dag {
    inner: daggy::Dag<Node, ()>,
    stack: Vec<daggy::NodeIndex>,
}

fn handle_op(
    op: FunctionOrOp,
    mut dag: &mut Dag,
    map: &mut HashMap<(Node, Vec<daggy::NodeIndex>), daggy::NodeIndex>,
    mut table_of_size: Option<Size>,
) {
    let mut insert_node = |dag: &mut Dag, parent_indices: Vec<daggy::NodeIndex>, node: Node| {
        let index = *map
            .entry((node.clone(), parent_indices.clone()))
            .or_insert_with(|| dag.inner.add_node(node));
        for parent_index in parent_indices {
            dag.inner.update_edge(parent_index, index, ()).unwrap();
        }
        dag.stack.push(index);
    };

    match op {
        FunctionOrOp::Function { modifier, code } => {
            let mut dipped = None;

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
                Modifier::Gap => {
                    dag.stack.pop().unwrap();
                }
                Modifier::Table => {
                    let x = dag.stack.get(dag.stack.len() - 1).unwrap();
                    let y = dag.stack.get(dag.stack.len() - 2).unwrap();
                    table_of_size = Some(Size::TransposeSizeOf(*x, *y));
                    assert_eq!(code.iter().map(|op| op.stack_delta()).sum::<i32>(), -1);
                }
            }

            for op in code {
                handle_op(op, dag, map, table_of_size);
            }

            if let Some(value) = dipped {
                dag.stack.push(value);
            }
        }
        FunctionOrOp::Op(Op::Value(value)) => insert_node(
            &mut dag,
            vec![],
            Node {
                op: NodeOp::Value(value.into()),
                size: table_of_size.unwrap_or(Size::Scalar),
            },
        ),
        FunctionOrOp::Op(Op::Stack(StackOp::Dup)) => {
            dag.stack.push(dag.stack.last().unwrap().clone());
        }
        FunctionOrOp::Op(Op::Stack(StackOp::Pop)) => {
            dag.stack.pop();
        }
        FunctionOrOp::Op(Op::Range) => {
            let parent_index = dag.stack.pop().unwrap();
            insert_node(
                &mut dag,
                vec![parent_index],
                Node {
                    op: NodeOp::Range,
                    size: Size::RangeOf(parent_index),
                },
            );
        }
        FunctionOrOp::Op(Op::Rev) => {
            let index = dag.stack.pop().unwrap();
            let size = dag.inner[index].size;
            insert_node(
                &mut dag,
                vec![index],
                Node {
                    op: NodeOp::Rev,
                    size,
                },
            );
        }
        FunctionOrOp::Op(Op::Monadic(op)) => {
            let index = dag.stack.pop().unwrap();
            let size = dag.inner[index].size;
            insert_node(
                &mut dag,
                vec![index],
                Node {
                    op: NodeOp::Monadic(op),
                    size: table_of_size.unwrap_or(size),
                },
            );
        }
        FunctionOrOp::Op(Op::Diadic(op)) => {
            let x = dag.stack.pop().unwrap();
            let y = dag.stack.pop().unwrap();
            let x_val = &dag.inner[x];
            let y_val = &dag.inner[y];
            let node = Node {
                op: NodeOp::Diadic {
                    is_table: table_of_size.is_some(),
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
            insert_node(&mut dag, vec![x, y], node);
        }
    }
}

fn parse_code_to_dag(code: Vec<FunctionOrOp>) -> (daggy::Dag<Node, ()>, Vec<daggy::NodeIndex>) {
    let mut dag = Dag {
        inner: daggy::Dag::<Node, ()>::new(),
        stack: Vec::new(),
    };
    let mut map = HashMap::new();

    for op in code {
        handle_op(op, &mut dag, &mut map, None);
    }

    (dag.inner, dag.stack)
}

use daggy::Walker;

fn evaluate_scalar(
    dag: &daggy::Dag<Node, ()>,
    functions: &mut HashSet<String>,
    full_eval_to: &HashMap<daggy::NodeIndex, usize>,
    top_level_eval: bool,
    index: NodeIndex,
) -> String {
    if !top_level_eval {
        if let Some(buffer_index) = full_eval_to.get(&index) {
            return if dag[index].size == Size::Scalar {
                format!("read_buffer({}, Coord(0,0,0,0))", buffer_index)
            } else {
                format!("read_buffer({}, thread_coord)", buffer_index)
            };
        }
    }

    match &dag[index].op {
        NodeOp::Monadic(op) => format!(
            "{}({})",
            format!("{:?}", op).to_lowercase(),
            evaluate_scalar(
                dag,
                functions,
                full_eval_to,
                false,
                dag.parents(index).walk_next(dag).unwrap().1
            )
        ),
        NodeOp::Diadic { op, is_table } => {
            let mut parents = dag.parents(index);
            let mut arg_0 = evaluate_scalar(
                dag,
                functions,
                full_eval_to,
                false,
                parents.walk_next(dag).unwrap().1,
            );
            let arg_1 = parents
                .walk_next(dag)
                .map(|parent| evaluate_scalar(dag, functions, full_eval_to, false, parent.1))
                .unwrap_or_else(|| arg_0.clone());

            if *is_table {
                functions.insert(format!(
                    "fn eval_{}(thread_coord: Coord, dispatch_size: Coord) -> f32 {{return {};}}",
                    index.index(),
                    arg_0
                ));
                arg_0 = format!(
                    "eval_{}(coord_transpose(thread_coord), dispatch_size)",
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
        NodeOp::Range => format!("f32(thread_coord[0])"),
        NodeOp::Rev => {
            let function = format!(
                "fn eval_{}(thread_coord: Coord, dispatch_size: Coord) -> f32 {{return {};}}",
                index.index(),
                evaluate_scalar(
                    dag,
                    functions,
                    full_eval_to,
                    false,
                    dag.parents(index).walk_next(dag).unwrap().1
                )
            );
            functions.insert(function);
            format!(
                "eval_{}(coord_reverse(thread_coord, dispatch_size), dispatch_size)",
                index.index(),
            )
        }
    }
}

fn evaluate_size(
    dag: &daggy::Dag<Node, ()>,
    functions: &mut HashSet<String>,
    full_eval_to: &HashMap<daggy::NodeIndex, usize>,
    index: NodeIndex,
) -> String {
    match &dag[index].size {
        Size::RangeOf(range) => {
            format!(
                "Coord(u32({}), 1, 1, 1)",
                evaluate_scalar(&dag, functions, full_eval_to, false, *range)
            )
        }
        Size::Scalar => format!("Coord(1,1,1,1)"),
        Size::TransposeSizeOf(a, b) => format!(
            "coord_max(Coord({0}[1], {0}[0], {0}[2], {0}[3]), {1})",
            evaluate_size(dag, functions, full_eval_to, *a),
            evaluate_size(dag, functions, full_eval_to, *b)
        ),
        _ => panic!(),
    }
}

#[derive(Default)]
struct FunctionPair {
    admin_lines: Vec<String>,
    work_lines: Vec<String>,
}

struct CodeBuilder {
    aux_functions: HashSet<String>,
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
fn function_delta() {
    assert_eq!(
        FunctionOrOp::Function {
            modifier: Modifier::Table,
            code: vec![FunctionOrOp::Op(Op::Diadic(DiadicOp::Eq))]
        }
        .stack_delta(),
        -1
    );
    assert_eq!(FunctionOrOp::Op(Op::Diadic(DiadicOp::Eq)).stack_delta(), -1)
}
