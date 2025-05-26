use ordered_float::OrderedFloat;

use crate::lexing::{
    DyadicModifier, DyadicOp, FunctionOrOp, FunctionOrOpWithContext, MonadicModifier, MonadicOp,
    Op, StackOp,
};
use std::collections::HashMap;
use std::ops::Range;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ArrayContents {
    Stack(Vec<usize>),
    Values(Vec<OrderedFloat<f32>>),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum NodeOp {
    Monadic(MonadicOp),
    Dyadic { is_table: bool, op: DyadicOp },
    Range,
    Rev,
    Rand,
    Len,
    Drop,
    Join,
    Value(ordered_float::OrderedFloat<f32>),
    ReduceResult,
    Reduce(usize),
    CreateArray(ArrayContents),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Node {
    pub op: NodeOp,
    pub size: Size,
    pub is_string: bool,
    pub in_loop: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Size {
    Scalar,
    Known([u32; 4]),
    Range(usize),
    Dyadic(usize, usize),
    Table(usize, usize),
    Join(usize, usize),
    Drop { array: usize, num: usize },
}

#[derive(Default)]
struct Dag {
    dag: Vec<(Node, Vec<usize>)>,
    stack: Vec<usize>,
    duplicate_map: HashMap<(Node, Vec<usize>), usize>,
    pushes_since_array_began: Vec<usize>,
}

impl Dag {
    fn push(&mut self, index: usize) {
        if let Some(count) = self.pushes_since_array_began.last_mut() {
            *count += 1;
        }

        self.stack.push(index);
    }

    fn insert_node(&mut self, node: Node, mut parent_indices: Vec<usize>) {
        parent_indices.reverse();

        let index = if node.op == NodeOp::Rand {
            let index = self.dag.len();
            self.dag.push((node, parent_indices));
            index
        } else {
            *self
                .duplicate_map
                .entry((node.clone(), parent_indices.clone()))
                .or_insert_with(|| {
                    let index = self.dag.len();
                    self.dag.push((node, parent_indices));
                    index
                })
        };
        self.push(index);
    }
}

#[derive(Default, Clone)]
struct ActiveModifiers {
    override_size: Option<Size>,
    in_loop: bool,
}

#[derive(Debug)]
pub enum Error {
    EmptyStack,
    BadReduction,
    Unimplemented,
    LoopCountTooHigh,
}

fn handle_op(
    op: FunctionOrOpWithContext,
    dag: &mut Dag,
    mut modifiers: ActiveModifiers,
) -> Result<(), Error> {
    match op.inner {
        FunctionOrOp::DyadicModifierFunction {
            modifier,
            code_a,
            code_b,
        } => match modifier {
            DyadicModifier::Fork => {
                let stack_delta = code_b
                    .iter()
                    .map(|code| code.stack_delta())
                    .sum::<Result<i32, ()>>()
                    .map_err(|_| Error::Unimplemented)?;

                let mut copies = Vec::new();

                for i in (dag.stack.len() as i32 + stack_delta - 1) as usize..dag.stack.len() {
                    copies.push(*dag.stack.get(i).ok_or(Error::EmptyStack)?);
                }

                for op in &code_b {
                    handle_op(op.clone(), dag, modifiers.clone())?;
                }

                for copy in copies {
                    dag.stack.push(copy);
                }

                for op in &code_a {
                    handle_op(op.clone(), dag, modifiers.clone())?;
                }
            }
        },
        FunctionOrOp::MonadicModifierFunction { modifier, code } => {
            let mut dipped = vec![];
            let mut reducing = None;

            match modifier {
                MonadicModifier::Repeat => {
                    match &dag.dag[dag.stack.pop().ok_or(Error::EmptyStack)?].0.op {
                        NodeOp::Value(value) => {
                            let value = **value as u32;
                            let mut loops = value;
                            let stack_delta = code
                                .iter()
                                .map(|code| code.stack_delta())
                                .sum::<Result<i32, ()>>()
                                .map_err(|_| Error::Unimplemented)?;
                            let stack_usage = code
                                .iter()
                                .map(|code| code.stack_usage())
                                .sum::<Result<u32, ()>>()
                                .map_err(|_| Error::Unimplemented)?;

                            if stack_delta > 0 {
                                loops -= stack_usage;
                            }

                            if loops > 100_000 {
                                return Err(Error::LoopCountTooHigh);
                            }

                            for _ in 0..loops {
                                for (i, op) in code.iter().enumerate() {
                                    handle_op(
                                        op.clone(),
                                        dag,
                                        ActiveModifiers {
                                            // only fully evaluate the final loop result instead of every intermediate value.
                                            // so e.g. 'repeat (max max)' does write(max(.., max(.., ..)))
                                            // TODO: probably leaky?
                                            in_loop: i == code.len() - 1,
                                            ..modifiers
                                        },
                                    )?;
                                }
                            }

                            match stack_delta {
                                i32::MIN..=0 => {}
                                1 => {
                                    let mut items: Vec<_> = (0..value)
                                        .map(|_| dag.stack.pop())
                                        .collect::<Option<Vec<_>>>()
                                        .ok_or(Error::EmptyStack)?;
                                    items.reverse();
                                    dag.insert_node(
                                        Node {
                                            op: NodeOp::CreateArray(ArrayContents::Stack(
                                                items.clone(),
                                            )),
                                            size: Size::Known([items.len() as _, 0, 0, 0]),
                                            is_string: false,
                                            in_loop: modifiers.in_loop,
                                        },
                                        items,
                                    );
                                }
                                2..=i32::MAX => return dbg!(Err(Error::Unimplemented)),
                            }

                            return Ok(());
                        }
                        _ => return dbg!(Err(Error::Unimplemented)),
                    }
                }
                MonadicModifier::Both => {
                    for op in &code {
                        handle_op(op.clone(), dag, modifiers.clone())?;
                    }

                    let stack_delta = code
                        .iter()
                        .map(|code| code.stack_delta())
                        .sum::<Result<i32, ()>>()
                        .map_err(|_| Error::Unimplemented)?;

                    if stack_delta < 1 {
                        dipped.push(dag.stack.pop().ok_or(Error::EmptyStack)?);
                    }
                }
                MonadicModifier::Back => {
                    let x = dag.stack.pop().ok_or(Error::EmptyStack)?;
                    let y = dag.stack.pop().ok_or(Error::EmptyStack)?;
                    dag.stack.push(x);
                    dag.stack.push(y);
                }
                MonadicModifier::Dip => {
                    dipped.push(dag.stack.pop().ok_or(Error::EmptyStack)?);
                }
                MonadicModifier::On => {
                    dipped.push(*dag.stack.last().ok_or(Error::EmptyStack)?);
                }
                MonadicModifier::By => {
                    let stack_delta = code
                        .iter()
                        .map(|code| code.stack_delta())
                        .sum::<Result<i32, ()>>()
                        .map_err(|_| Error::Unimplemented)?;
                    match stack_delta {
                        1..=i32::MAX => {}
                        other => {
                            let index = (dag.stack.len() as i32 - 1 + other) as usize;
                            dag.stack
                                .insert(index, *dag.stack.get(index).ok_or(Error::EmptyStack)?);
                        }
                    }
                }
                MonadicModifier::Gap => {
                    dag.stack.pop().ok_or(Error::EmptyStack)?;
                }
                MonadicModifier::Table => {
                    let x = dag.stack.last().ok_or(Error::EmptyStack)?;
                    let y = dag
                        .stack
                        .get(dag.stack.len().checked_sub(2).ok_or(Error::EmptyStack)?)
                        .ok_or(Error::EmptyStack)?;
                    modifiers.override_size = Some(Size::Table(*x, *y));
                }
                MonadicModifier::Reduce => {
                    let stack_delta = code
                        .iter()
                        .map(|code| code.stack_delta())
                        .sum::<Result<i32, ()>>()
                        .map_err(|_| Error::Unimplemented)?;
                    if stack_delta != -1 {
                        return Err(Error::BadReduction);
                    }
                    let reducing_array = *dag.stack.last().ok_or(Error::EmptyStack)?;
                    let reducing_array_size = dag.dag[reducing_array].0.size;
                    reducing = Some(*dag.stack.last().ok_or(Error::EmptyStack)?);
                    dag.insert_node(
                        Node {
                            op: NodeOp::ReduceResult,
                            size: Size::Scalar,
                            is_string: false,
                            in_loop: modifiers.in_loop,
                        },
                        vec![],
                    );
                    modifiers.override_size = Some(reducing_array_size);
                }
                MonadicModifier::Rows => {
                    let x = *dag.stack.last().ok_or(Error::EmptyStack)?;
                    modifiers.override_size = Some(dag.dag[x].0.size);
                }
                MonadicModifier::Below => {
                    let stack_delta = code
                        .iter()
                        .map(|code| code.stack_delta())
                        .sum::<Result<i32, ()>>()
                        .map_err(|_| Error::Unimplemented)?;
                    match stack_delta {
                        0..=i32::MAX => return dbg!(Err(Error::Unimplemented)),
                        _ => {
                            for i in
                                (dag.stack.len() as i32 + stack_delta - 1) as usize..dag.stack.len()
                            {
                                dag.stack.push(*dag.stack.get(i).ok_or(Error::EmptyStack)?);
                            }
                        }
                    }
                }
            }

            for op in code {
                handle_op(op.clone(), dag, modifiers.clone())?;
            }

            if let Some(reducing) = reducing {
                let parent = dag.stack.pop().ok_or(Error::EmptyStack)?;
                dag.insert_node(
                    Node {
                        op: NodeOp::Reduce(reducing),
                        size: Size::Scalar,
                        is_string: false,
                        in_loop: modifiers.in_loop,
                    },
                    vec![parent],
                );
            }

            for value in dipped {
                dag.push(value);
            }
        }
        FunctionOrOp::Op(Op::Char(char)) => dag.insert_node(
            Node {
                op: NodeOp::Value((char as u32 as f32).into()),
                size: modifiers.override_size.unwrap_or(Size::Scalar),
                is_string: true,
                in_loop: modifiers.in_loop,
            },
            vec![],
        ),
        FunctionOrOp::Op(Op::String(string)) => {
            dag.insert_node(
                Node {
                    op: NodeOp::CreateArray(ArrayContents::Values(
                        string
                            .chars()
                            .map(|char| (char as u32 as f32).into())
                            .collect(),
                    )),
                    size: Size::Known([string.len() as _, 0, 0, 0]),
                    is_string: true,
                    in_loop: modifiers.in_loop,
                },
                vec![],
            );
        }
        FunctionOrOp::Op(Op::Array(values)) => {
            dag.insert_node(
                Node {
                    size: Size::Known([values.len() as _, 0, 0, 0]),
                    op: NodeOp::CreateArray(ArrayContents::Values(
                        values.into_iter().map(|value| value.into()).collect(),
                    )),
                    is_string: false,
                    in_loop: modifiers.in_loop,
                },
                vec![],
            );
        }
        FunctionOrOp::Op(Op::EndArray) => {
            let array_len = dag
                .pushes_since_array_began
                .pop()
                .ok_or_else(|| dbg!(Error::Unimplemented))?;
            let items = (0..array_len)
                .map(|_| dag.stack.pop())
                .collect::<Option<Vec<_>>>()
                .ok_or(Error::EmptyStack)?;
            dag.insert_node(
                Node {
                    op: NodeOp::CreateArray(ArrayContents::Stack(items.clone())),
                    size: Size::Known([items.len() as _, 0, 0, 0]),
                    is_string: false,
                    in_loop: modifiers.in_loop,
                },
                items,
            );
        }
        FunctionOrOp::Op(Op::StartArray) => {
            dag.pushes_since_array_began.push(0);
        }
        FunctionOrOp::Op(Op::Rand) => dag.insert_node(
            Node {
                op: NodeOp::Rand,
                size: modifiers.override_size.unwrap_or(Size::Scalar),
                is_string: false,
                in_loop: modifiers.in_loop,
            },
            vec![],
        ),
        FunctionOrOp::Op(Op::Value(value)) => dag.insert_node(
            Node {
                op: NodeOp::Value(value.into()),
                size: modifiers.override_size.unwrap_or(Size::Scalar),
                is_string: false,
                in_loop: modifiers.in_loop,
            },
            vec![],
        ),
        FunctionOrOp::Op(Op::Stack(StackOp::Dup)) => {
            // slight hack to make sure that uiua behaviour is copied.
            if dag.pushes_since_array_began.last() == Some(&0) {
                let item = dag.stack.pop().ok_or(Error::EmptyStack)?;
                dag.push(item);
                dag.push(item);
            } else {
                let item = *dag.stack.last().ok_or(Error::EmptyStack)?;
                dag.push(item);
            }
        }
        FunctionOrOp::Op(Op::Stack(StackOp::Pop)) => {
            dag.stack.pop().ok_or(Error::EmptyStack)?;
        }
        FunctionOrOp::Op(Op::Stack(StackOp::Ident)) => {
            // Potentially change size of the node on the top of the stack.
            let index = dag.stack.pop().ok_or(Error::EmptyStack)?;
            let parents: Vec<_> = dag.dag[index].1.clone();
            let mut node = dag.dag[index].0.clone();
            node.size = modifiers.override_size.unwrap_or(node.size);
            dag.insert_node(node, parents);
        }
        FunctionOrOp::Op(Op::Len) => {
            let parent_index = dag.stack.pop().ok_or(Error::EmptyStack)?;
            dag.insert_node(
                Node {
                    op: NodeOp::Len,
                    size: Size::Scalar,
                    is_string: false,
                    in_loop: modifiers.in_loop,
                },
                vec![parent_index],
            );
        }
        FunctionOrOp::Op(Op::Range) => {
            let parent_index = dag.stack.pop().ok_or(Error::EmptyStack)?;
            dag.insert_node(
                Node {
                    op: NodeOp::Range,
                    size: Size::Range(parent_index),
                    is_string: false,
                    in_loop: modifiers.in_loop,
                },
                vec![parent_index],
            );
        }
        FunctionOrOp::Op(Op::Drop) => {
            let num = dag.stack.pop().ok_or(Error::EmptyStack)?;
            let array = dag.stack.pop().ok_or(Error::EmptyStack)?;
            dag.insert_node(
                Node {
                    op: NodeOp::Drop,
                    size: Size::Drop { array, num },
                    is_string: dag.dag[array].0.is_string,
                    in_loop: modifiers.in_loop,
                },
                vec![num, array],
            );
        }
        FunctionOrOp::Op(Op::Join) => {
            let a = dag.stack.pop().ok_or(Error::EmptyStack)?;
            let b = dag.stack.pop().ok_or(Error::EmptyStack)?;
            dag.insert_node(
                Node {
                    op: NodeOp::Join,
                    size: match (dag.dag[a].0.size, dag.dag[b].0.size) {
                        (Size::Scalar, Size::Scalar) => Size::Known([2, 0, 0, 0]),
                        (Size::Scalar, Size::Known(known)) | (Size::Known(known), Size::Scalar) => {
                            Size::Known([known[0] + 1, known[1], known[2], known[3]])
                        }
                        (Size::Known(a), Size::Known(b)) => {
                            Size::Known([a[0] + b[0], a[1], a[2], a[3]])
                        }
                        _ => Size::Join(a, b),
                    },
                    is_string: dag.dag[a].0.is_string && dag.dag[b].0.is_string,
                    in_loop: modifiers.in_loop,
                },
                vec![a, b],
            );
        }
        FunctionOrOp::Op(Op::Rev) => {
            let index = dag.stack.pop().ok_or(Error::EmptyStack)?;
            let size = dag.dag[index].0.size;
            dag.insert_node(
                Node {
                    op: NodeOp::Rev,
                    size,
                    is_string: dag.dag[index].0.is_string,
                    in_loop: modifiers.in_loop,
                },
                vec![index],
            );
        }
        FunctionOrOp::Op(Op::Monadic(op)) => {
            let index = dag.stack.pop().ok_or(Error::EmptyStack)?;
            let size = dag.dag[index].0.size;
            dag.insert_node(
                Node {
                    op: NodeOp::Monadic(op),
                    size: modifiers.override_size.unwrap_or(size),
                    is_string: false,
                    in_loop: modifiers.in_loop,
                },
                vec![index],
            );
        }
        FunctionOrOp::Op(Op::Dyadic(op)) => {
            let x = dag.stack.pop().ok_or(Error::EmptyStack)?;
            let y = dag.stack.pop().ok_or(Error::EmptyStack)?;
            let x_val = &dag.dag[x].0;
            let y_val = &dag.dag[y].0;
            let node = Node {
                op: NodeOp::Dyadic {
                    is_table: matches!(modifiers.override_size, Some(Size::Table(_, _))),
                    op,
                },
                is_string: x_val.is_string && y_val.is_string,
                in_loop: modifiers.in_loop,
                size: if let Some(size) = modifiers.override_size {
                    size
                } else {
                    match (x_val.size, y_val.size) {
                        (Size::Known(x), Size::Known(y)) => Size::Known([
                            x[0].max(y[0]),
                            x[1].max(y[1]),
                            x[2].max(y[2]),
                            x[3].max(y[3]),
                        ]),
                        (Size::Range(x), Size::Range(y)) if x == y => Size::Range(x),
                        (Size::Dyadic(a, b), Size::Dyadic(c, d))
                            if (a == c && b == d) || (a == d && b == c) =>
                        {
                            Size::Dyadic(a, b)
                        }
                        (Size::Table(a, b), Size::Table(c, d))
                            if (a == c && b == d) || (a == d && b == c) =>
                        {
                            Size::Table(a, b)
                        }
                        (Size::Scalar, other) | (other, Size::Scalar) => other,
                        // give up
                        _ => Size::Dyadic(x, y),
                    }
                },
            };
            dag.insert_node(node, vec![x, y]);
        }
    }

    Ok(())
}

pub fn parse_code_to_dag(
    code: Vec<FunctionOrOpWithContext>,
) -> Result<(Vec<(Node, Vec<usize>)>, Vec<usize>), (Error, Range<usize>, &str)> {
    let mut dag = Dag::default();

    for op in code {
        handle_op(op.clone(), &mut dag, Default::default())
            .map_err(|err| (err, op.span, op.line))?;
    }

    Ok((dag.dag, dag.stack))
}
