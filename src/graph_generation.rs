use crate::lexing::{DyadicOp, FunctionOrOp, Modifier, MonadicOp, Op, StackOp};
use daggy::Walker;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ArrayContents {
    Stack(Vec<daggy::NodeIndex>),
    Chars(Vec<char>),
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
    Value(ordered_float::OrderedFloat<f32>),
    ReduceResult,
    Reduce(daggy::NodeIndex),
    CreateArray(ArrayContents),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Node {
    pub op: NodeOp,
    pub size: Size,
    pub is_string: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Size {
    Scalar,
    Known([u32; 4]),
    RangeOf(daggy::NodeIndex),
    MaxOf(daggy::NodeIndex, daggy::NodeIndex),
    TransposeSizeOf(daggy::NodeIndex, daggy::NodeIndex),
    Drop {
        array: daggy::NodeIndex,
        num: daggy::NodeIndex,
    },
}

#[derive(Default)]
struct Dag {
    inner: daggy::Dag<Node, ()>,
    stack: Vec<daggy::NodeIndex>,
    duplicate_map: HashMap<(Node, Vec<daggy::NodeIndex>), daggy::NodeIndex>,
    pushes_since_array_began: Vec<usize>,
}

impl Dag {
    fn push(&mut self, index: daggy::NodeIndex) {
        if let Some(count) = self.pushes_since_array_began.last_mut() {
            *count += 1;
        }

        self.stack.push(index);
    }

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
        self.push(index);
    }
}

fn handle_op(op: FunctionOrOp, dag: &mut Dag, mut override_size: Option<Size>) {
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
                Modifier::On => {
                    dipped = Some(*dag.stack.last().unwrap());
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
                    override_size = Some(Size::TransposeSizeOf(*x, *y));
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
                            is_string: false,
                        },
                        vec![],
                    );
                    override_size = Some(reducing_array_size);
                }
                Modifier::Rows => {
                    let x = *dag.stack.last().unwrap();
                    override_size = Some(dag.inner[x].size);
                }
            }

            for op in code {
                handle_op(op, dag, override_size);
            }

            if let Some(reducing) = reducing {
                let parent = dag.stack.pop().unwrap();
                dag.insert_node(
                    Node {
                        op: NodeOp::Reduce(reducing),
                        size: Size::Scalar,
                        is_string: false,
                    },
                    vec![parent],
                );
            }

            if let Some(value) = dipped {
                dag.push(value);
            }
        }
        FunctionOrOp::Op(Op::Char(char)) => dag.insert_node(
            Node {
                op: NodeOp::Value((char as u32 as f32).into()),
                size: override_size.unwrap_or(Size::Scalar),
                is_string: true,
            },
            vec![],
        ),
        FunctionOrOp::Op(Op::String(string)) => {
            dag.insert_node(
                Node {
                    op: NodeOp::CreateArray(ArrayContents::Chars(string.chars().collect())),
                    size: Size::Known([string.len() as _, 1, 1, 1]),
                    is_string: true,
                },
                vec![],
            );
        }
        FunctionOrOp::Op(Op::EndArray) => {
            let array_len = dag.pushes_since_array_began.pop().unwrap();
            let items: Vec<_> = (0..array_len).map(|_| dag.stack.pop().unwrap()).collect();
            dag.insert_node(
                Node {
                    op: NodeOp::CreateArray(ArrayContents::Stack(items.clone())),
                    size: Size::Known([items.len() as _, 1, 1, 1]),
                    is_string: false,
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
                size: override_size.unwrap_or(Size::Scalar),
                is_string: false,
            },
            vec![],
        ),
        FunctionOrOp::Op(Op::Value(value)) => dag.insert_node(
            Node {
                op: NodeOp::Value(value.into()),
                size: override_size.unwrap_or(Size::Scalar),
                is_string: false,
            },
            vec![],
        ),
        FunctionOrOp::Op(Op::Stack(StackOp::Dup)) => {
            // slight hack to make sure that uiua behaviour is copied.
            if dag.pushes_since_array_began.last() == Some(&0) {
                let item = dag.stack.pop().unwrap();
                dag.push(item);
                dag.push(item);
            } else {
                let item = *dag.stack.last().unwrap();
                dag.push(item);
            }
        }
        FunctionOrOp::Op(Op::Stack(StackOp::Pop)) => {
            dag.stack.pop().unwrap();
        }
        FunctionOrOp::Op(Op::Stack(StackOp::Ident)) => {
            // Potentially change size of the node on the top of the stack.
            let index = dag.stack.pop().unwrap();
            let parents: Vec<_> = dag
                .inner
                .parents(index)
                .iter(&dag.inner)
                .map(|(_, index)| index)
                .collect();
            let mut node = dag.inner[index].clone();
            node.size = override_size.unwrap_or(node.size);
            dag.insert_node(node, parents);
        }
        FunctionOrOp::Op(Op::Len) => {
            let parent_index = dag.stack.pop().unwrap();
            dag.insert_node(
                Node {
                    op: NodeOp::Len,
                    size: Size::Scalar,
                    is_string: false,
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
                    is_string: false,
                },
                vec![parent_index],
            );
        }
        FunctionOrOp::Op(Op::Drop) => {
            let num = dag.stack.pop().unwrap();
            let array = dag.stack.pop().unwrap();
            dag.insert_node(
                Node {
                    op: NodeOp::Drop,
                    size: Size::Drop { array, num },
                    is_string: dag.inner[array].is_string,
                },
                vec![num, array],
            );
        }
        FunctionOrOp::Op(Op::Rev) => {
            let index = dag.stack.pop().unwrap();
            let size = dag.inner[index].size;
            dag.insert_node(
                Node {
                    op: NodeOp::Rev,
                    size,
                    is_string: dag.inner[index].is_string,
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
                    size: override_size.unwrap_or(size),
                    is_string: false,
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
                    is_table: matches!(override_size, Some(Size::TransposeSizeOf(_, _))),
                    op,
                },
                is_string: x_val.is_string && y_val.is_string,
                size: if let Some(size) = override_size {
                    size
                } else {
                    match (x_val.size, y_val.size) {
                        (Size::Known(x), Size::Known(y)) => Size::Known([
                            x[0].max(y[0]),
                            x[1].max(y[1]),
                            x[2].max(y[2]),
                            x[3].max(y[3]),
                        ]),
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

pub fn parse_code_to_dag(code: Vec<FunctionOrOp>) -> (daggy::Dag<Node, ()>, Vec<daggy::NodeIndex>) {
    let mut dag = Dag::default();

    for op in code {
        handle_op(op, &mut dag, None);
    }

    (dag.inner, dag.stack)
}
