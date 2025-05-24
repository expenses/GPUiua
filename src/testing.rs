#[cfg(test)]
use crate::{
    ReadBackValue, Runner,
    lexing::{DyadicOp, FunctionOrOp, MonadicModifier, Op},
};

#[cfg(test)]
fn assert_output(string: &str, output: Vec<ReadBackValue>) {
    pollster::block_on(async {
        let context = Runner::new().await;
        let (values, _module) = context.run_string(string, false).await.unwrap();
        assert_eq!(values, output);
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
            size: [5, 5, 0, 0],
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
            size: [3, 3, 0, 0],
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
                size: [3, 3, 0, 0],
                #[rustfmt::skip]
                values: vec![
                        0.0, 1.0, 2.0,
                        1.0, 1.0, 2.0,
                        2.0, 2.0, 2.0
                ],
            },
            ReadBackValue {
                size: [3, 0, 0, 0],
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
            size: [2, 2, 0, 0],
            values: vec![5.0, 5.0, 5.0, 5.0],
        }],
    );
}

#[test]
fn table_pop_in_parens() {
    assert_output(
        "table (5 pop pop) . range 2",
        vec![ReadBackValue {
            size: [2, 2, 0, 0],
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
        FunctionOrOp::MonadicModifierFunction {
            modifier: MonadicModifier::Table,
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
            size: [3, 3, 0, 0],
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
            size: [3, 3, 0, 0],
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

#[test]
fn sum_div_len() {
    assert_output(
        "
        [1 5 8 2]
        ⟜/+ # Sum
        ⧻   # Length
        ÷   # Divide
        ",
        vec![ReadBackValue::scalar(4.0)],
    )
}

#[test]
fn max_different_sized_arrays() {
    assert_output(
        "+ range 5 range 4",
        vec![ReadBackValue {
            size: [5, 0, 0, 0],
            // 7.0 because the read from range 4 is clamped.
            values: vec![0.0, 2.0, 4.0, 6.0, 7.0],
        }],
    )
}

#[test]
fn array_creation() {
    assert_output(
        "[. rand rand] [1]",
        vec![
            ReadBackValue::scalar(1.0),
            ReadBackValue {
                size: [3, 0, 0, 0],
                values: vec![0.96570766, 0.96570766, 0.74225104],
            },
        ],
    )
}

#[test]
fn array_creation_mirrors_uiua() {
    assert_output(
        "[ident] len range 5 [.]1",
        vec![
            ReadBackValue {
                size: [2, 0, 0, 0],
                values: vec![1.0; 2],
            },
            ReadBackValue::scalar(5.0),
        ],
    )
}

#[test]
fn rows_rand() {
    assert_output(
        "rows gap rand range 5",
        vec![ReadBackValue {
            size: [5, 0, 0, 0],
            values: vec![0.4196651, 0.7859788, 0.84304845, 0.56487226, 0.15387845],
        }],
    );
}

#[test]
fn strings_and_chars() {
    assert_output(
        "ne@ \"hi :)\"",
        vec![ReadBackValue {
            size: [5, 0, 0, 0],
            values: vec![1.0, 1.0, 0.0, 1.0, 1.0],
        }],
    );
}

#[test]
fn string_example() {
    assert_output(
        "
            \"Unabashedly I utilize arrays\"
            ⊸≠@  # Mask of non-spaces
            # ⊜⊢   # All first letters
            ⊙◌
        ",
        vec![ReadBackValue {
            size: [28, 0, 0, 0],
            values: vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        }],
    );
}

#[test]
fn drop_1d() {
    assert_output("drop 2 [1.5 2.5 3.5]", vec![ReadBackValue::scalar(3.5)]);
}

#[test]
fn drop_2d() {
    assert_output(
        "drop 2 table max . range 5",
        vec![ReadBackValue {
            size: [3, 5, 0, 0],
            #[rustfmt::skip]
            values: vec![
                2.0, 3.0, 4.0,
                2.0, 3.0, 4.0,
                2.0, 3.0, 4.0,
                3.0, 3.0, 4.0,
                4.0, 4.0, 4.0
            ],
        }],
    );
}

#[test]
fn double_drop_add() {
    assert_output(
        "+drop 1 drop 1 .. table max .. range 3",
        vec![
            ReadBackValue {
                size: [3, 0, 0, 0],
                values: vec![0.0, 1.0, 2.0],
            },
            ReadBackValue {
                size: [3, 3, 0, 0],
                #[rustfmt::skip]
                values: vec![
                    0.0, 1.0, 2.0,
                    1.0, 1.0, 2.0,
                    2.0, 2.0, 2.0
                ],
            },
            ReadBackValue {
                size: [3, 3, 0, 0],
                values: vec![2.0, 3.0, 4.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            },
        ],
    );
}

#[test]
fn dip_rev() {
    assert_output(
        "dip rev drop 2 . range 3",
        vec![
            ReadBackValue {
                size: [3, 0, 0, 0],
                values: vec![2.0, 1.0, 0.0],
            },
            ReadBackValue {
                size: [1, 0, 0, 0],
                values: vec![2.0],
            },
        ],
    );
}

#[test]
fn below_modifier() {
    assert_output(
        "below+ 1 2",
        vec![
            ReadBackValue::scalar(2.0),
            ReadBackValue::scalar(1.0),
            ReadBackValue::scalar(3.0),
        ],
    );
}

#[test]
fn fibonacci() {
    assert_output(
        "⍥◡+9 .1",
        vec![ReadBackValue {
            size: [9, 0, 0, 0],
            values: vec![1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0],
        }],
    );
}

#[test]
fn floor_repeat_rand() {
    assert_output(
        "floor*10 repeat rand 5",
        vec![ReadBackValue {
            size: [5, 0, 0, 0],
            values: vec![7.0, 9.0, 6.0, 7.0, 0.0],
        }],
    );
}

#[test]
fn repeat_table() {
    assert_output(
        "dip repeat(mul 2).8 table gap gap 1.range 3",
        vec![
            ReadBackValue {
                size: [3, 3, 0, 0],
                values: vec![256.0; 9],
            },
            ReadBackValue::scalar(8.0),
        ],
    );
}

#[test]
fn both_dyadic() {
    assert_output(
        "both + 1 2 3 4",
        vec![ReadBackValue::scalar(7.0), ReadBackValue::scalar(3.0)],
    );
}

#[test]
fn both_monadic() {
    assert_output(
        "both (+1) 1 2 3 4",
        vec![
            ReadBackValue::scalar(4.0),
            ReadBackValue::scalar(3.0),
            ReadBackValue::scalar(3.0),
            ReadBackValue::scalar(2.0),
        ],
    );
}

#[test]
fn both_stack_pushes() {
    assert_output(
        "both (4 rand) 1",
        vec![
            ReadBackValue::scalar(1.0),
            ReadBackValue::scalar(0.96570766),
            ReadBackValue::scalar(4.0),
            ReadBackValue::scalar(0.7708467),
            ReadBackValue::scalar(4.0),
        ],
    );

    assert_output(
        "both (rand) 1",
        vec![
            ReadBackValue::scalar(1.0),
            ReadBackValue::scalar(0.96570766),
            ReadBackValue::scalar(0.6287222),
        ],
    );
}

#[test]
fn fork_dyadic() {
    assert_output(
        "⊃+ - 4 5",
        vec![ReadBackValue::scalar(1.0), ReadBackValue::scalar(9.0)],
    );
}

#[test]
fn fork_monadic() {
    assert_output(
        "fork ceil floor 4.5",
        vec![ReadBackValue::scalar(4.0), ReadBackValue::scalar(5.0)],
    );
}

#[test]
fn fork_pushes() {
    assert_output(
        "⊃(5)(4) 3",
        vec![
            ReadBackValue::scalar(3.0),
            ReadBackValue::scalar(4.0),
            ReadBackValue::scalar(5.0),
        ],
    )
}

#[test]
fn fork_idents() {
    assert_output(
        "⊃(∘)(∘) 5",
        vec![ReadBackValue::scalar(5.0), ReadBackValue::scalar(5.0)],
    );
}

#[test]
fn fork_dupes() {
    assert_output(
        "⊃(.4)(.5)",
        vec![
            ReadBackValue::scalar(5.0),
            ReadBackValue::scalar(5.0),
            ReadBackValue::scalar(4.0),
            ReadBackValue::scalar(4.0),
        ],
    );
}

#[test]
fn audio_example_almost() {
    assert_output(
        "
        [0 4 7 10]     # Notes
        ×220 ˜ⁿ2÷12    # Freqs
        ∿×τ ⊞× ÷⟜⇡&asr # Generate
        len #÷⧻⟜/+⍉         # Mix
        ",
        vec![ReadBackValue::scalar(44100.0)],
    )
}

#[test]
fn sine_waves_almost() {
    assert_output(
        "
        ⊞+⇡3∿∩(÷25)⇡240⇡80
        ⊙⧻⧻ # ⍉⊞<
        ",
        vec![ReadBackValue::scalar(80.0), ReadBackValue::scalar(3.0)],
    );
}

#[test]
fn reverse_example() {
    assert_output(
        "⇌ 1_2_3_4 # Reverse",
        vec![ReadBackValue {
            size: [4, 0, 0, 0],
            values: vec![4.0, 3.0, 2.0, 1.0],
        }],
    );
}

#[test]
fn mod_op() {
    assert_output(
        "◿4 ¯1 ◿5 51",
        vec![ReadBackValue::scalar(1.0), ReadBackValue::scalar(3.0)],
    );
}

#[test]
fn underscore_array() {
    assert_output(
        "div 5 32_23",
        vec![ReadBackValue {
            size: [2, 0, 0, 0],
            values: vec![6.4, 4.6],
        }],
    );
}

#[test]
fn uiua_logo_parsing_sorta() {
    assert_output(
        "
        U ← /=⊞<0.2_0.7 /+×⟜ⁿ1_2
        I ← <⊙(⌵)#/ℂ) # Circle
        u ← +0.1↧#¤ ⊃(I0.95|⊂⊙0.5⇌°√)
        A ← ×⊃U(I 1) # Alpha
        #⍜°⍉(⊂⊃u A) ⊞⊟.-1×2÷⟜⇡200
        ",
        vec![],
    );
}

#[test]
fn basic_joins() {
    assert_output(
        "join 3 join 1 2",
        vec![ReadBackValue {
            size: [3, 0, 0, 0],
            values: vec![3.0, 1.0, 2.0],
        }],
    );

    assert_output(
        "join 3_4 1_2",
        vec![ReadBackValue {
            size: [4, 0, 0, 0],
            values: vec![3.0, 4.0, 1.0, 2.0],
        }],
    );
}

#[test]
fn join_rev() {
    assert_output(
        "join 3_4 rev 1_2",
        vec![ReadBackValue {
            size: [4, 0, 0, 0],
            values: vec![3.0, 4.0, 2.0, 1.0],
        }],
    );
}

#[test]
fn spiral() {
    assert_output(
        "⟜(×20-⊸¬÷⟜⇡)200",
        vec![
            ReadBackValue {
                size: [200, 0, 0, 0],
                values: vec![
                    -20.0,
                    -19.8,
                    -19.6,
                    -19.400002,
                    -19.2,
                    -19.0,
                    -18.800001,
                    -18.599998,
                    -18.4,
                    -18.199999,
                    -18.0,
                    -17.8,
                    -17.6,
                    -17.4,
                    -17.2,
                    -17.0,
                    -16.800001,
                    -16.6,
                    -16.400002,
                    -16.2,
                    -15.999999,
                    -15.799999,
                    -15.599999,
                    -15.4,
                    -15.2,
                    -15.0,
                    -14.8,
                    -14.6,
                    -14.400001,
                    -14.200001,
                    -14.000001,
                    -13.800001,
                    -13.600001,
                    -13.4,
                    -13.199999,
                    -13.0,
                    -12.799999,
                    -12.6,
                    -12.4,
                    -12.200001,
                    -12.0,
                    -11.800001,
                    -11.6,
                    -11.400002,
                    -11.200001,
                    -10.999999,
                    -10.799999,
                    -10.599999,
                    -10.4,
                    -10.2,
                    -10.0,
                    -9.8,
                    -9.6,
                    -9.400001,
                    -9.200001,
                    -9.0,
                    -8.8,
                    -8.6,
                    -8.4,
                    -8.2,
                    -8.0,
                    -7.7999997,
                    -7.6,
                    -7.4,
                    -7.2000003,
                    -7.0000005,
                    -6.8000007,
                    -6.600001,
                    -6.4000006,
                    -6.200001,
                    -6.000001,
                    -5.7999997,
                    -5.6,
                    -5.4,
                    -5.2,
                    -5.0,
                    -4.8,
                    -4.6000004,
                    -4.4000006,
                    -4.2000003,
                    -4.0000005,
                    -3.800001,
                    -3.6000009,
                    -3.400001,
                    -3.1999998,
                    -2.9999998,
                    -2.8,
                    -2.6000001,
                    -2.4000003,
                    -2.2000003,
                    -2.0000005,
                    -1.8000005,
                    -1.6000006,
                    -1.4000008,
                    -1.2000008,
                    -1.000001,
                    -0.79999983,
                    -0.59999996,
                    -0.40000004,
                    -0.20000012,
                    -2.2351742e-7,
                    0.19999968,
                    0.3999996,
                    0.5999995,
                    0.79999936,
                    0.9999993,
                    1.1999998,
                    1.3999997,
                    1.5999997,
                    1.7999995,
                    1.9999994,
                    2.1999993,
                    2.3999999,
                    2.5999997,
                    2.7999997,
                    2.9999995,
                    3.1999993,
                    3.3999994,
                    3.5999992,
                    3.7999997,
                    3.9999998,
                    4.2,
                    4.3999996,
                    4.5999994,
                    4.799999,
                    4.9999995,
                    5.2,
                    5.3999996,
                    5.5999994,
                    5.799999,
                    5.999999,
                    6.199999,
                    6.3999996,
                    6.5999994,
                    6.799999,
                    6.999999,
                    7.199999,
                    7.3999987,
                    7.6,
                    7.7999997,
                    7.9999995,
                    8.199999,
                    8.4,
                    8.599998,
                    8.799999,
                    9.0,
                    9.2,
                    9.4,
                    9.599999,
                    9.799999,
                    9.999999,
                    10.2,
                    10.4,
                    10.599999,
                    10.799999,
                    10.999999,
                    11.199999,
                    11.4,
                    11.599999,
                    11.799999,
                    11.999999,
                    12.199999,
                    12.399999,
                    12.599998,
                    12.799999,
                    13.0,
                    13.199999,
                    13.4,
                    13.599998,
                    13.799999,
                    14.0,
                    14.2,
                    14.4,
                    14.599999,
                    14.799999,
                    14.999999,
                    15.2,
                    15.4,
                    15.599999,
                    15.799999,
                    15.999999,
                    16.199999,
                    16.399998,
                    16.6,
                    16.8,
                    17.0,
                    17.199999,
                    17.4,
                    17.599998,
                    17.8,
                    18.0,
                    18.199999,
                    18.4,
                    18.599998,
                    18.8,
                    18.999998,
                    19.199999,
                    19.4,
                    19.599998,
                    19.8,
                ],
            },
            ReadBackValue::scalar(200.0),
        ],
    );
}
