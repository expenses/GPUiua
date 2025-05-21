use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"[ \t]+")]
pub enum Token<'source> {
    #[regex("@.", |lex| lex.slice().chars().nth(1).unwrap())]
    Char(char),
    #[regex("\"[^\"]*\"", |lex| &lex.slice()[1..lex.slice().len()-1])]
    String(&'source str),
    #[regex(r"[a-zA-Z_]+[a-zA-Z_0-9]*", priority = 0)]
    AssignedName(&'source str),
    #[regex(r"rows|≡")]
    Rows,
    #[regex(r"add|\+")]
    Add,
    #[regex(r"mul|\*|×")]
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
    #[regex(r"on|⟜")]
    On,
    #[regex(r"sqrt|√")]
    Sqrt,
    #[regex(r"len|⧻")]
    Len,
    #[regex(r"/")]
    Reduce,
    #[regex(r"[0-9]+(\.[0-9]+)?", |lex| lex.slice().parse::<f32>().unwrap())]
    Value(f32),
    #[token("(")]
    OpenParen,
    #[token(")")]
    CloseParen,
    #[token("[")]
    ArrayLeft,
    #[token("]")]
    ArrayRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonadicOp {
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
pub enum DyadicOp {
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
pub enum StackOp {
    Dup,
    Pop,
    Ident,
}

#[derive(Clone, Debug)]
pub enum FunctionOrOp<'a> {
    Op(Op<'a>),
    Function {
        modifier: Modifier,
        code: Vec<FunctionOrOp<'a>>,
    },
}

impl<'a> FunctionOrOp<'a> {
    #[allow(unused)]
    pub fn stack_delta(&self) -> i32 {
        match self {
            Self::Op(Op::Monadic(_)) => 0,
            Self::Op(Op::Dyadic(_)) => -1,
            Self::Op(Op::Value(_)) | Self::Op(Op::String(_)) | Self::Op(Op::Char(_)) => 1,
            Self::Op(Op::Rand) => 1,
            Self::Op(Op::Stack(StackOp::Dup)) => 1,
            Self::Op(Op::Stack(StackOp::Ident)) => 0,
            Self::Op(Op::Stack(StackOp::Pop)) => -1,
            Self::Op(Op::Len | Op::Rev | Op::Range) => 0,
            Self::Op(Op::EndArray) => 0,
            // Tricky
            Self::Op(Op::StartArray) => 0,
            Self::Function { modifier, code } => {
                let modifier = match *modifier {
                    Modifier::Back => 0,
                    Modifier::Dip => 0,
                    Modifier::Table => 0,
                    Modifier::Gap => -1,
                    Modifier::By => 1,
                    Modifier::Reduce => 1,
                    Modifier::On => 1,
                    Modifier::Rows => 0,
                };

                modifier + code.iter().map(|op| op.stack_delta()).sum::<i32>()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Op<'a> {
    Monadic(MonadicOp),
    Dyadic(DyadicOp),
    Stack(StackOp),
    Value(f32),
    Range,
    Rev,
    Rand,
    Len,
    StartArray,
    EndArray,
    String(&'a str),
    Char(char),
}

#[derive(Debug, Clone, Copy)]
pub enum Modifier {
    Table,
    Back,
    Gap,
    Dip,
    By,
    Reduce,
    On,
    Rows,
}
