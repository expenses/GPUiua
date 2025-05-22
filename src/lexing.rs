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
    #[regex(r"drop|↘")]
    Drop,
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
            Self::Op(Op::Drop) => -1,
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
    Drop,
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

pub enum TokenType<'a> {
    Modifier(Modifier),
    Op(Op<'a>),
    AssignedOp(&'a str),
}

pub fn parse(token: Token) -> Option<TokenType> {
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
        Token::On => TokenType::Modifier(Modifier::On),
        Token::Gap => TokenType::Modifier(Modifier::Gap),
        Token::Dip => TokenType::Modifier(Modifier::Dip),
        Token::Back => TokenType::Modifier(Modifier::Back),
        Token::Table => TokenType::Modifier(Modifier::Table),
        Token::Reduce => TokenType::Modifier(Modifier::Reduce),
        Token::Rows => TokenType::Modifier(Modifier::Rows),
        Token::Rev => TokenType::Op(Op::Rev),
        Token::Rand => TokenType::Op(Op::Rand),
        Token::Range => TokenType::Op(Op::Range),
        Token::Len => TokenType::Op(Op::Len),
        Token::Drop => TokenType::Op(Op::Drop),
        Token::Value(value) => TokenType::Op(Op::Value(value)),
        Token::String(string) => TokenType::Op(Op::String(string)),
        Token::Char(char) => TokenType::Op(Op::Char(char)),
        Token::AssignedName(string) => TokenType::AssignedOp(string),
        Token::OpenParen
        | Token::CloseParen
        | Token::ArrayLeft
        | Token::ArrayRight
        | Token::Assignment => return None,
    })
}
