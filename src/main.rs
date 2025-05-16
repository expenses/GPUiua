#[derive(Debug)]
enum Command {
    AllocateAndWrite {
        allocation_size_code: &'static str,
        write_code: String,
    },
}

use nbn::vk;

use Command::*;
fn main() {
    env_logger::init();
    let device = nbn::Device::new(None, true);
    let shader = device.load_shader("out.spv");
    let allocate_scalar = device.create_compute_pipeline(&shader, c"allocate_scalar");
    let range_allocation = device.create_compute_pipeline(&shader, c"range_allocation");
    let range = device.create_compute_pipeline(&shader, c"range");
    let allocate_copy = device.create_compute_pipeline(&shader, c"allocate_copy");
    let allocate_max = device.create_compute_pipeline(&shader, c"allocate_max");
    let add = device.create_compute_pipeline(&shader, c"add");
    let copy = device.create_compute_pipeline(&shader, c"copy");
    let allocate_transpose = device.create_compute_pipeline(&shader, c"allocate_transpose");
    let table_eq = device.create_compute_pipeline(&shader, c"table_eq");
    let rev = device.create_compute_pipeline(&shader, c"rev");
    let max = device.create_compute_pipeline(&shader, c"max_vals");
    let allocate_fold = device.create_compute_pipeline(&shader, c"allocate_fold");
    let fold_add = device.create_compute_pipeline(&shader, c"fold_add");
    let clear = device.create_compute_pipeline(&shader, c"clear");
    let length = device.create_compute_pipeline(&shader, c"length");
    let div = device.create_compute_pipeline(&shader, c"div");
    
    let command_buffer = device.create_command_buffer(nbn::QueueType::Compute);
    let fence = device.create_fence();

    let dispatches = device
        .create_buffer(nbn::BufferDescriptor {
            name: "dispatch",
            size: 4 * 3 * 2,
            ty: nbn::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let stack_size = device
        .create_buffer(nbn::BufferDescriptor {
            name: "stack_size",
            size: 4,
            ty: nbn::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let offset = device
        .create_buffer(nbn::BufferDescriptor {
            name: "offset",
            size: 4,
            ty: nbn::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let buffers = device
        .create_buffer(nbn::BufferDescriptor {
            name: "buffers",
            size: (4 * 4 + 8) * 100,
            ty: nbn::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let values = device
        .create_buffer(nbn::BufferDescriptor {
            name: "values",
            size: 4024 * 4,
            ty: nbn::MemoryLocation::GpuToCpu,
        })
        .unwrap();

    unsafe {
        device
            .begin_command_buffer(*command_buffer, &Default::default())
            .unwrap();
        device.push_constants::<(u64, u64, u64, u64, u64, u32)>(
            &command_buffer,
            (*dispatches, *stack_size, *buffers, *values, *offset, 0),
        );

        let dispatch = |allocation_pipeline, write_pipeline| {
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                allocation_pipeline,
            );
            device.cmd_dispatch(*command_buffer, 1, 1, 1);
            device.insert_global_barrier(
                &command_buffer,
                &[nbn::AccessType::ComputeShaderReadWrite],
                &[nbn::AccessType::ComputeShaderReadWrite],
            );
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                write_pipeline,
            );
            device.cmd_dispatch_indirect(*command_buffer, *dispatches.buffer, 0);
            device.insert_global_barrier(
                &command_buffer,
                &[nbn::AccessType::ComputeShaderReadWrite],
                &[nbn::AccessType::ComputeShaderReadWrite],
            );
        };
        let push_scalar = |value| {
            device.push_constants::<(u64, u64, u64, u64, u64, u32)>(
                &command_buffer,
                (*dispatches, *stack_size, *buffers, *values, *offset, value),
            );
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                *allocate_scalar,
            );
            device.cmd_dispatch(*command_buffer, 1, 1, 1);
            device.insert_global_barrier(
                &command_buffer,
                &[nbn::AccessType::ComputeShaderReadWrite],
                &[nbn::AccessType::ComputeShaderReadWrite],
            );
        };
        let dispatch_second_size = |pipeline| {
            device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_dispatch_indirect(*command_buffer, *dispatches.buffer, 4 * 3);
            device.insert_global_barrier(
                &command_buffer,
                &[nbn::AccessType::ComputeShaderReadWrite],
                &[nbn::AccessType::ComputeShaderReadWrite],
            );
        };
        let fold = |write_pipeline| {
            dispatch(*allocate_fold, *clear);
            dispatch_second_size(write_pipeline);
        };
        push_scalar(10);
        dispatch(*range_allocation, *range);
        dispatch(*allocate_max, *div);
        //dispatch(*allocate_copy, *copy);
        //dispatch(*allocate_transpose, *table_eq);
        //dispatch(*allocate_copy, *rev);
        //dispatch(*allocate_copy, *max);
        device.end_command_buffer(*command_buffer).unwrap();
        device
            .queue_submit(
                *device.compute_queue,
                &[vk::SubmitInfo::default().command_buffers(&[*command_buffer])],
                *fence,
            )
            .unwrap();

        device.wait_for_fences(&[*fence], true, !0).unwrap();
    }

    let lines = &values.try_as_slice::<f32>().unwrap()[10..25];

    for line in lines {
        println!("{:?}", line);
    }

    /*let _input = "
    +1⇡5  # 1 through 5
    ⊞=. # Identity matrix
    ⊸⇌  # Reverse
    ↥   # Max
    ";

    let _output = [
        AllocateAndWrite {
            allocation_size_code: "uint3(1, 0, 0)",
            write_code: "5".into(),
        },
        AllocateAndWrite {
            allocation_size_code: "uint3(load_buffer_at_stack_depth(0, 0), 0, 0)",
            write_code: "thread_index".into(),
        },
        AllocateAndWrite {
            allocation_size_code: "uint3(1, 0, 0)",
            write_code: "1".into(),
        },
        AllocateAndWrite {
            allocation_size_code: "buffer_size_at_stack_depth(0)",
            write_code: "load_buffer_at_stack_depth(0, thread_index) + load_buffer_at_stack_depth(1, thread_index)".into(),
        },
        AllocateAndWrite {
            allocation_size_code: "buffer_size_at_stack_depth(0)",
            write_code: "load_buffer_at_stack_depth(0, thread_index)".into(),
        },
        AllocateAndWrite {
            allocation_size_code: "buffer_size_at_stack_depth(0) + buffer_size_at_stack_depth(1).yzx",
            write_code: "load_buffer_at_stack_depth(0, thread_index) == load_buffer_at_stack_depth(1, thread_index.yxz)".into()
        }
    ];

    let _shader_code = "
        struct Buffer {
            float* ptr;
            uint3 size;
        };

        Buffer* BUFFERS;

        float load_buffer_1d(uint buffer_index, uint index) {
            return BUFFERS[buffer_index].ptr[index]
        }
    ";

    dbg!(
        _input
            .lines()
            .flat_map(|line| {
                let a = line.split_once('#').map(|(a, b)| a).unwrap_or(line);

                a.chars().rev().filter(|&c| c != ' ')
            })
            .collect::<Vec<_>>()
    );

    let chars = _input.lines().flat_map(|line| {
        let a = line.split_once('#').map(|(a, b)| a).unwrap_or(line);

        a.chars().rev().filter(|&c| c != ' ')
    });

    /*let mut output = Vec::new();

    for char in chars {
        match char {
            '0'..'9' => {
                output.push(AllocateAndWrite {
                    allocation_size_code: "uint3(1, 0, 0)",
                    write_code: char.to_string(),
                });
            }
            '⇡' => {
                output.push(AllocateAndWrite {
                    allocation_size_code: "uint3(load_buffer_at_stack_depth(0, 0), 0, 0)",
                    write_code: format!("thread_index"),
                });
            }
            '+' => {
                output.push(AllocateAndWrite {
                    allocation_size_code: "buffer_size_at_stack_depth(0)",
                    write_code: format!("load_buffer_at_stack_depth(0, thread_index) + load_buffer_at_stack_depth(0, thread_index)"),
                });
            }
            '.' => {
                output.push(AllocateAndWrite {
                    allocation_size_code: "buffer_size_at_stack_depth(0)",
                    write_code: format!("load_buffer_at_stack_depth(0, thread_index)"),
                });
            }
            _ => panic!("{}: {:#?}", char, output),
        }
    }*/

    /*let _output = [
        //
        Allocate {
            code_to_determine_size_and_type: "uint3(1, 0, 0)",
        },
        WriteInShaderTobuffer {
            code: "5",
            buffer_index: 0,
            dispatch: (1, 1, 1),
        },
        Allocate {
            code_to_determine_size_and_type: "uint3(load_buffer_1d(0, 0), 0, 0)",
        },
        WriteInShaderTobuffer {
            code: "thread_index + 1",
            buffer_index: 0,
            dispatch: (5, 1, 1),
        },
        Allocate {
            code_to_determine_size_and_type: "buffer_size(1)",
        },
        WriteInShaderTobuffer {
            code: "load_buffer_1d(1, thread_index)",
            buffer_index: 1,
            dispatch: (5, 1, 1),
        },
        Allocate {
            // Funky swizzle here on buffer_size(2). Only arrays sizes up to 3D are supported
            code_to_determine_size_and_type: "buffer_size(1) + buffer_size(2).yzx",
        },
        WriteInShaderTobuffer {
            code: "load_buffer_1d(1, thread_index.x) == load_buffer_1d(2, thread_index.y)",
            buffer_index: 2,
            dispatch: (5, 5, 1),
        },
        Allocate {
            code_to_determine_size_and_type: "buffer_size(3)",
        },
        WriteInShaderTobuffer {
            code: "load_buffer_2d(3, uint2(4-thread_index.x, thread_index.y))",
            buffer_index: 3,
            dispatch: (5, 5, 1),
        },
        Allocate {
            code_to_determine_size_and_type: "buffer_size(4)",
        },
        WriteInShaderTobuffer {
            code: "max(load_buffer_2d(4, thread_index), load_buffer_2d(3, thread_index))",
            buffer_index: 4,
            dispatch: (5, 5, 1),
        },
    ];*/*/
}
