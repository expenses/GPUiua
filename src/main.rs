use rand::Rng;

mod graph_generation;
mod lexing;
mod parsing;
mod runner;
mod testing;

use graph_generation::{ArrayContents, Node, NodeOp, Size, parse_code_to_dag};
use lexing::FunctionOrOp;
use runner::Runner;

fn generate_module(code: Vec<FunctionOrOp>) -> ShaderModule {
    let mut rng = rand::rngs::StdRng::from_seed([0; 32]);

    let (dag, final_stack) = parse_code_to_dag(code);
    let mut full_eval_to: BTreeMap<usize, (usize, Vec<usize>)> = BTreeMap::new();
    for (i, &index) in final_stack.iter().enumerate() {
        match full_eval_to.entry(index) {
            Entry::Occupied(mut occupied) => occupied.get_mut().1.push(i),
            Entry::Vacant(vacancy) => {
                vacancy.insert((i, vec![]));
            }
        }
    }

    let mut dummy_buffer_index = final_stack.len();

    {
        let mut walk_stack = final_stack.clone();
        let mut touched = HashSet::new();
        while let Some(index) = walk_stack.pop() {
            if touched.contains(&index) {
                continue;
            }
            touched.insert(index);
            for &parent in &dag[index].1 {
                walk_stack.push(parent);
            }

            let node = &dag[index].0;
            match node {
                Node {
                    op: NodeOp::Dyadic { .. },
                    size: Size::Dyadic(_, _),
                    ..
                } => {
                    for &index in &dag[index].1 {
                        full_eval_to
                            .entry(index)
                            .or_insert_with(|| (dummy_buffer_index, vec![]));
                        dummy_buffer_index += 1;
                    }
                }
                Node {
                    op: NodeOp::Rand,
                    size: Size::Scalar,
                    ..
                }
                | Node {
                    op: NodeOp::Reduce(_),
                    ..
                }
                | Node { in_loop: true, .. } => {
                    full_eval_to
                        .entry(index)
                        .or_insert_with(|| (dummy_buffer_index, vec![]));
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

    for (&item, (i, copies)) in &full_eval_to {
        let size = evaluate_size(
            &mut EvaluationContext {
                dag: &dag,
                functions: &mut code_builder.aux_functions,
                full_eval_to: &full_eval_to,
                rng: &mut rng,
            },
            item,
        );
        code_builder.functions[0]
            .admin_lines
            .push(format!("allocate({}, {})", i, size));
        for copy in copies {
            code_builder.functions[0]
                .admin_lines
                .push(format!("buffers[{}] = buffers[{}]", copy, i));
        }
        node_to_allocation.insert(item, i);

        match &dag[item].0.size {
            Size::Scalar => {
                code_builder.functions[0]
                    .admin_lines
                    .push("random_seed += 1".to_string());
                code_builder.functions[0].admin_lines.push(format!(
                    "write_to_buffer({}, Coord(0,0,0,0), {})",
                    i,
                    evaluate(
                        &mut EvaluationContext {
                            dag: &dag,
                            functions: &mut code_builder.aux_functions,
                            full_eval_to: &full_eval_to,
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
                    .entry(dag[item].0.size)
                    .or_insert_with(|| {
                        let dispatch_index = code_builder.next_dispatch_index;
                        code_builder.next_dispatch_index += 1;

                        code_builder.functions[dispatch_index]
                            .admin_lines
                            .push(format!("dispatch_for_buffer({})", i));
                        code_builder.functions[dispatch_index].dispatching_on_buffer_index =
                            Some(*i);

                        code_builder.functions.push(Default::default());
                        dispatch_index
                    });

                code_builder.functions[dispatch_index]
                    .work_lines
                    .push(format!(
                        "write_to_buffer({}, thread_coord, {})",
                        i,
                        evaluate(
                            &mut EvaluationContext {
                                dag: &dag,
                                functions: &mut code_builder.aux_functions,
                                full_eval_to: &full_eval_to,
                                rng: &mut rng,
                            },
                            true,
                            item
                        )
                    ));
            }
        }
    }

    let mut shader = include_str!("shader.wgsl").to_string();

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
        shader.push_str("let thread_coord = Coord(0,0,0,0);\n");
        for line in &pair.admin_lines {
            shader.push_str(&format!("{};\n", line));
        }
        shader.push_str("}\n");
        step_index += 1;
        shader.push_str(&format!("@compute @workgroup_size(64,1,1) fn step_{}(@builtin(global_invocation_id) thread: vec3<u32>) {{\n", step_index));
        if let Some(index) = pair.dispatching_on_buffer_index {
            shader.push_str(&format!("let dispatch_size = buffers[{}].size;\n", index));
            shader.push_str("let thread_coord = index_to_coord(thread.x, dispatch_size);\n");
            shader.push_str("if (coord_any_gt(thread_coord, dispatch_size)) {return;}\n");
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
        final_stack_data: final_stack
            .iter()
            .map(|index| dag[*index].0.is_string)
            .collect(),
    }
}

struct ShaderModule {
    code: String,
    num_functions: usize,
    final_stack_data: Vec<bool>,
}

use std::collections::{BTreeMap, HashMap, HashSet, btree_map::Entry};

use rand::SeedableRng;

struct EvaluationContext<'a, R> {
    dag: &'a Vec<(Node, Vec<usize>)>,
    functions: &'a mut AuxFunctions,
    full_eval_to: &'a BTreeMap<usize, (usize, Vec<usize>)>,
    rng: &'a mut R,
}

fn evaluate<R: Rng>(
    context: &mut EvaluationContext<R>,
    top_level_eval: bool,
    index: usize,
) -> String {
    if !top_level_eval {
        if let Some((buffer_index, _)) = context.full_eval_to.get(&index) {
            return if context.dag[index].0.size == Size::Scalar {
                format!("read_buffer({}, Coord(0,0,0,0))", buffer_index)
            } else {
                format!("read_buffer({}, thread_coord)", buffer_index)
            };
        }
    }

    match &context.dag[index].0.op {
        NodeOp::Join => {
            let parents = &context.dag[index].1;
            let right = evaluate(context, false, parents[0]);
            let left_size = evaluate_size(context, parents[1]);
            let left = evaluate(context, false, parents[1]);
            insert_function(context.functions, format!("join_{}", index), format!("
                let left_size = {}[0];
                if (thread_coord[0] >= left_size) {{
                    thread_coord = coord_plus_x(thread_coord, -f32(left_size));
                    dispatch_size = coord_plus_x(dispatch_size, -f32(left_size));
                    return {};
                }} else {{
                    dispatch_size = Coord(left_size, dispatch_size[1], dispatch_size[2], dispatch_size[3]);
                    return {};
                }}
                ", left_size, right, left));
            format!("join_{}(thread_coord, dispatch_size, thread)", index)
        }
        NodeOp::Drop => {
            let parents = &context.dag[index].1;
            let array = evaluate(context, false, parents[0]);
            let num = evaluate(context, false, parents[1]);

            insert_function(
                context.functions,
                format!("drop_{}", index),
                format!("return {};", array),
            );

            format!(
                "drop_{}(coord_plus_x(thread_coord, {}), dispatch_size, thread)",
                index, num
            )
        }
        NodeOp::CreateArray(items) => {
            let size = match context.dag[index].0.size {
                Size::Known([x, 0, 0, 0]) => x,
                Size::Scalar => 1,
                _ => unreachable!(),
            };

            if size > 0 {
                let children = match items {
                    ArrayContents::Stack(indices) => indices
                        .iter()
                        .map(|child| evaluate(context, false, *child))
                        .collect::<Vec<_>>()
                        .join(", "),
                    ArrayContents::Values(values) => values
                        .iter()
                        .map(|&value| format!("{}", value))
                        .collect::<Vec<_>>()
                        .join(", "),
                };

                context.functions.insert(
                    format!("created_array_{0}", index),
                    format!(
                        "fn created_array_{0}(index: u32) -> f32 {{
                        let array_{0} = array<f32, {1}>({2});
                        return array_{0}[index];
                    }}",
                        index, size, children
                    ),
                );
                format!("created_array_{}(thread_coord[0])", index)
            } else {
                "0".to_string()
            }
        }
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
                index,
                evaluate_size(context, *reducing),
                evaluate(context, false, *reducing),
                context.rng.random::<u32>(),
                evaluate(
                    context,
                    false,
                    context.dag[index].1[0]
                )
            );
            context
                .functions
                .insert(format!("reduce_{}", index), function);
            format!("reduce_{}()", index)
        }
        NodeOp::ReduceResult => "reduction".to_string(),
        NodeOp::Len => {
            let size = evaluate_size(context, context.dag[index].1[0]);
            format!("f32({}[0])", size)
        }
        NodeOp::Monadic(op) => format!(
            "{}({})",
            format!("{:?}", op).to_lowercase(),
            evaluate(context, false, context.dag[index].1[0],)
        ),
        NodeOp::Dyadic { op, is_table } => {
            let parents = &context.dag[index].1;
            let mut arg_0 = evaluate(context, false, parents[0]);
            let arg_1 = evaluate(context, false, parents[1]);

            if *is_table {
                insert_function(
                    context.functions,
                    format!("table_{}", index),
                    format!("return {};", arg_0),
                );
                arg_0 = format!(
                    "table_{}(coord_transpose(thread_coord), dispatch_size, thread)",
                    index
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
                index,
                context.rng.random::<u32>(),
                evaluate(context, false, context.dag[index].1[0])
            );
            context.functions.insert(format!("rev_{}", index), function);
            format!(
                "rev_{}(coord_reverse(thread_coord, dispatch_size), dispatch_size, thread)",
                index,
            )
        }
    }
}

type AuxFunctions = HashMap<String, String>;

fn insert_function(functions: &mut AuxFunctions, name: String, body: String) {
    functions.insert(
        name.clone(),
        format!(
            "fn {}(thread_coord_: Coord, dispatch_size_: Coord, thread: vec3<u32>) -> f32 {{
                var thread_coord = thread_coord_;
                var dispatch_size = dispatch_size_;
            {}
        }}",
            name, body
        ),
    );
}

fn evaluate_size<R: Rng>(context: &mut EvaluationContext<R>, index: usize) -> String {
    match &context.dag[index].0.size {
        Size::Join(a, b) => {
            format!(
                "coord_plus_x({}, f32({}[0]))",
                evaluate_size(context, *a),
                evaluate_size(context, *b)
            )
        }
        Size::Drop { array, num } => {
            format!(
                "coord_plus_x({}, -{})",
                evaluate_size(context, *array),
                evaluate(context, false, *num)
            )
        }
        Size::Range(range) => {
            format!("Coord(u32({}), 0, 0, 0)", evaluate(context, false, *range))
        }
        Size::Scalar => "Coord(1,0,0,0)".to_string(),
        Size::Table(a, b) => format!(
            "coord_table({}, {})",
            evaluate_size(context, *a),
            evaluate_size(context, *b)
        ),
        Size::Dyadic(a, b) => format!(
            "coord_max({}, {})",
            evaluate_size(context, *a),
            evaluate_size(context, *b)
        ),
        Size::Known([x, y, z, w]) => format!("Coord({}, {}, {}, {})", x, y, z, w),
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
            let output = Runner::new()
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
            let context = std::rc::Rc::new(Runner::new().await);
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
            size: [1, 0, 0, 0],
            values: vec![value],
        }
    }
}
