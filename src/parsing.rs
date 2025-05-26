use crate::lexing::{FunctionOrOp, FunctionOrOpWithContext, Op, Token, TokenType, parse};
use logos::Logos;
use std::collections::HashMap;
use std::ops::Range;

pub fn parse_code(
    code: &str,
    left_to_right: bool,
) -> Result<Vec<FunctionOrOpWithContext>, (&str, Range<usize>)> {
    let mut parsed_code = Vec::new();
    let mut assignments: HashMap<&str, Vec<FunctionOrOpWithContext>> = HashMap::new();
    for line in code.lines() {
        let line = line
            .split_once('#')
            .map(|(before_comment, _)| before_comment)
            .unwrap_or(line);
        let mut lexer = Token::lexer(line).spanned().peekable();

        if let Some((Ok(Token::AssignedName(name)), span)) = lexer.peek().cloned() {
            if assignments.contains_key(name) {
                let line = parse_code_line(lexer, left_to_right, left_to_right, &assignments, line)
                    .map_err(|span| (line, span))?;
                parsed_code.extend_from_slice(&line);
            } else {
                let _ = lexer.next().unwrap();
                match lexer.next() {
                    Some((Ok(Token::EqualSign | Token::Assignment), _)) => {}
                    Some((_, span)) => return Err((line, span)),
                    None => {
                        return Err((line, span));
                    }
                }
                let line = parse_code_line(lexer, true, left_to_right, &assignments, line)
                    .map_err(|span| (line, span))?;
                assignments.insert(name, line);
            }
        } else {
            let line = parse_code_line(lexer, left_to_right, left_to_right, &assignments, line)
                .map_err(|span| (line, span))?;
            parsed_code.extend_from_slice(&line);
        }
    }

    Ok(parsed_code)
}

pub fn parse_code_line<'a>(
    mut lexer: std::iter::Peekable<logos::SpannedIter<'a, Token<'a>>>,
    left_to_right: bool,
    children_are_left_to_right: bool,
    assignments: &HashMap<&str, Vec<FunctionOrOpWithContext<'a>>>,
    line: &'a str,
) -> Result<Vec<FunctionOrOpWithContext<'a>>, Range<usize>> {
    let mut code = Vec::new();

    loop {
        let section =
            parse_code_section(&mut lexer, children_are_left_to_right, assignments, line)?;
        if section.is_empty() {
            break;
        }
        code.extend_from_slice(&section);
    }

    if !left_to_right {
        code.reverse();
    }

    Ok(code)
}

fn parse_code_section<'a>(
    lexer: &mut std::iter::Peekable<logos::SpannedIter<'a, Token<'a>>>,
    left_to_right: bool,
    assignments: &HashMap<&str, Vec<FunctionOrOpWithContext<'a>>>,
    line: &'a str,
) -> Result<Vec<FunctionOrOpWithContext<'a>>, Range<usize>> {
    let (mut token, mut span) = match lexer.next() {
        Some((Ok(token), span)) => (token, span),
        Some((Err(()), span)) => return Err(span),
        None => return Ok(vec![]),
    };
    while let Token::OpenParen | Token::CloseParen = token {
        match lexer.next() {
            Some((Ok(token2), span2)) => {
                token = token2;
                span = span2;
            }
            Some((Err(()), span)) => return Err(span),
            None => return Ok(vec![]),
        };
    }

    let mut get_code = |span| {
        if let Some(&(Ok(Token::OpenParen), _)) = lexer.peek() {
            let _ = lexer.next();
            let mut code = Vec::new();
            loop {
                if let Some(&(Ok(Token::CloseParen), _)) = lexer.peek() {
                    let _ = lexer.next();
                    if !left_to_right {
                        code.reverse();
                    }
                    return Ok(code);
                }

                code.extend_from_slice(&match parse_code_section(
                    lexer,
                    left_to_right,
                    assignments,
                    line,
                ) {
                    Ok(ops) if ops.is_empty() => return Err(span),
                    Ok(ops) => ops,
                    Err(span) => return Err(span),
                })
            }
        } else {
            match parse_code_section(lexer, left_to_right, assignments, line) {
                Ok(ops) if ops.is_empty() => Err(span),
                Ok(ops) => Ok(ops),
                Err(span) => Err(span),
            }
        }
    };

    match token {
        Token::ArrayLeft => Ok(vec![FunctionOrOpWithContext::new(
            FunctionOrOp::Op(if left_to_right {
                Op::StartArray
            } else {
                Op::EndArray
            }),
            span,
            line,
        )]),
        Token::ArrayRight => Ok(vec![FunctionOrOpWithContext::new(
            FunctionOrOp::Op(if left_to_right {
                Op::EndArray
            } else {
                Op::StartArray
            }),
            span,
            line,
        )]),
        _ => match parse(token) {
            Some(TokenType::AssignedOp(name)) => match assignments.get(name).cloned() {
                Some(code) => Ok(code),
                None => Err(span),
            },
            Some(TokenType::Op(op)) => Ok(vec![FunctionOrOpWithContext::new(
                FunctionOrOp::Op(op),
                span,
                line,
            )]),
            Some(TokenType::MonadicModifier(modifier)) => Ok(vec![FunctionOrOpWithContext::new(
                FunctionOrOp::MonadicModifierFunction {
                    modifier,
                    code: get_code(span.clone())?,
                },
                span,
                line,
            )]),
            Some(TokenType::DyadicModifier(modifier)) => Ok(vec![FunctionOrOpWithContext::new(
                FunctionOrOp::DyadicModifierFunction {
                    modifier,
                    code_a: get_code(span.clone())?,
                    code_b: get_code(span.clone())?,
                },
                span,
                line,
            )]),
            None => Err(span),
        },
    }
}
