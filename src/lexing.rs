use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t]+")]
pub enum Token<'source> {
    #[regex(r"[0-9]+(\.[0-9]+)?(_[0-9]+(\.[0-9]+)?)+", |lex| {
        lex.slice().split('_').map(|value| value.parse::<f32>()).collect::<Result<Vec<_>, _>>().unwrap()
    })]
    UnderscoreArray(Vec<f32>),
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
    #[regex(r"min|↧")]
    Min,
    #[regex(r"round|⁅")]
    Round,
    #[regex(r"not|¬")]
    Not,
    #[regex("sub|-")]
    Sub,
    #[regex("mod|◿")]
    Mod,
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
    #[regex(r"repeat|⍥")]
    Repeat,
    #[regex(r"below|◡")]
    Below,
    #[regex(r"both|∩")]
    Both,
    #[regex(r"fork|⊃")]
    Fork,
    #[regex(r"pow|ⁿ")]
    Pow,
    #[regex(r"tau|τ")]
    Tau,
    #[regex(r"join")]
    Join,
    #[token("&asr")]
    AudioSampleRate,
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
    Pow,
    Min,
    Modulus,
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
    MonadicModifierFunction {
        modifier: MonadicModifier,
        code: Vec<FunctionOrOp<'a>>,
    },
    DyadicModifierFunction {
        modifier: DyadicModifier,
        code_a: Vec<FunctionOrOp<'a>>,
        code_b: Vec<FunctionOrOp<'a>>,
    },
}

impl<'a> FunctionOrOp<'a> {
    pub fn stack_delta(&self) -> i32 {
        match self {
            Self::Op(Op::Monadic(_)) => 0,
            Self::Op(Op::Drop) => -1,
            Self::Op(Op::Dyadic(_) | Op::Join) => -1,
            Self::Op(Op::Value(_) | Op::String(_) | Op::Char(_) | Op::Array(_)) => 1,
            Self::Op(Op::Rand) => 1,
            Self::Op(Op::Stack(StackOp::Dup)) => 1,
            Self::Op(Op::Stack(StackOp::Ident)) => 0,
            Self::Op(Op::Stack(StackOp::Pop)) => -1,
            Self::Op(Op::Len | Op::Rev | Op::Range) => 0,
            Self::Op(Op::EndArray) => todo!(),
            Self::Op(Op::StartArray) => todo!(),
            Self::MonadicModifierFunction { modifier, code } => {
                let stack_delta = code.iter().map(|op| op.stack_delta()).sum::<i32>();

                let modifier = match *modifier {
                    MonadicModifier::Back => 0,
                    MonadicModifier::Dip => 0,
                    MonadicModifier::Table => 0,
                    MonadicModifier::Gap => -1,
                    MonadicModifier::By => 1,
                    MonadicModifier::Reduce => 1,
                    MonadicModifier::On => 1,
                    MonadicModifier::Rows => 0,
                    MonadicModifier::Below => return 1,
                    MonadicModifier::Repeat => todo!(),
                    MonadicModifier::Both => todo!(),
                };

                modifier + stack_delta
            }
            Self::DyadicModifierFunction {
                modifier,
                code_a,
                code_b,
            } => {
                todo!()
            }
        }
    }

    pub fn stack_usage(&self) -> u32 {
        match self {
            Self::Op(Op::Monadic(_)) => 1,
            Self::Op(Op::Drop) => 2,
            Self::Op(Op::Dyadic(_) | Op::Join) => 2,
            Self::Op(Op::Value(_) | Op::String(_) | Op::Char(_) | Op::Array(_)) => 0,
            Self::Op(Op::Rand) => 0,
            Self::Op(Op::Stack(StackOp::Dup)) => 0,
            Self::Op(Op::Stack(StackOp::Ident)) => 0,
            Self::Op(Op::Stack(StackOp::Pop)) => 1,
            Self::Op(Op::Len | Op::Rev | Op::Range) => 1,
            Self::Op(Op::EndArray) => todo!(),
            // Tricky
            Self::Op(Op::StartArray) => todo!(),
            Self::MonadicModifierFunction { modifier, code } => {
                let stack_usage = code.iter().map(|op| op.stack_usage()).sum::<u32>();

                let modifier = match *modifier {
                    MonadicModifier::Back => 0,
                    MonadicModifier::Dip => 0,
                    MonadicModifier::Table => 0,
                    MonadicModifier::Gap => 1,
                    MonadicModifier::By => 0,
                    MonadicModifier::Reduce => 0,
                    MonadicModifier::On => 0,
                    MonadicModifier::Rows => 0,
                    MonadicModifier::Below => 0,
                    MonadicModifier::Repeat => todo!(),
                    MonadicModifier::Both => todo!(),
                };

                modifier + stack_usage
            }
            Self::DyadicModifierFunction {
                modifier,
                code_a,
                code_b,
            } => {
                todo!()
            }
        }
    }
}

#[derive(Debug, Clone)]
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
    Join,
    StartArray,
    EndArray,
    String(&'a str),
    Char(char),
    Array(Vec<f32>),
}

#[derive(Debug, Clone, Copy)]
pub enum DyadicModifier {
    Fork,
}

#[derive(Debug, Clone, Copy)]
pub enum MonadicModifier {
    Table,
    Back,
    Gap,
    Dip,
    By,
    Reduce,
    On,
    Rows,
    Below,
    Repeat,
    Both,
}

pub enum TokenType<'a> {
    MonadicModifier(MonadicModifier),
    DyadicModifier(DyadicModifier),
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
        Token::Pow => TokenType::Op(Op::Dyadic(DyadicOp::Pow)),
        Token::Min => TokenType::Op(Op::Dyadic(DyadicOp::Min)),
        Token::Mod => TokenType::Op(Op::Dyadic(DyadicOp::Modulus)),
        Token::Sqrt => TokenType::Op(Op::Monadic(MonadicOp::Sqrt)),
        Token::Dup => TokenType::Op(Op::Stack(StackOp::Dup)),
        Token::Pop => TokenType::Op(Op::Stack(StackOp::Pop)),
        Token::Ident => TokenType::Op(Op::Stack(StackOp::Ident)),
        Token::By => TokenType::MonadicModifier(MonadicModifier::By),
        Token::On => TokenType::MonadicModifier(MonadicModifier::On),
        Token::Gap => TokenType::MonadicModifier(MonadicModifier::Gap),
        Token::Dip => TokenType::MonadicModifier(MonadicModifier::Dip),
        Token::Back => TokenType::MonadicModifier(MonadicModifier::Back),
        Token::Table => TokenType::MonadicModifier(MonadicModifier::Table),
        Token::Reduce => TokenType::MonadicModifier(MonadicModifier::Reduce),
        Token::Rows => TokenType::MonadicModifier(MonadicModifier::Rows),
        Token::Below => TokenType::MonadicModifier(MonadicModifier::Below),
        Token::Repeat => TokenType::MonadicModifier(MonadicModifier::Repeat),
        Token::Both => TokenType::MonadicModifier(MonadicModifier::Both),
        Token::Fork => TokenType::DyadicModifier(DyadicModifier::Fork),
        Token::Rev => TokenType::Op(Op::Rev),
        Token::Rand => TokenType::Op(Op::Rand),
        Token::Range => TokenType::Op(Op::Range),
        Token::Len => TokenType::Op(Op::Len),
        Token::Drop => TokenType::Op(Op::Drop),
        Token::Join => TokenType::Op(Op::Join),
        Token::AudioSampleRate => TokenType::Op(Op::Value(44100.0)),
        Token::Tau => TokenType::Op(Op::Value(std::f32::consts::TAU)),
        Token::Value(value) => TokenType::Op(Op::Value(value)),
        Token::String(string) => TokenType::Op(Op::String(string)),
        Token::Char(char) => TokenType::Op(Op::Char(char)),
        Token::AssignedName(string) => TokenType::AssignedOp(string),
        Token::UnderscoreArray(array) => TokenType::Op(Op::Array(array)),
        Token::OpenParen
        | Token::CloseParen
        | Token::ArrayLeft
        | Token::ArrayRight
        | Token::Assignment => return None,
    })
}
