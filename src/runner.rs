use crate::parsing::parse_code;
use crate::{ReadBackValue, ShaderModule, generate_module};

pub struct Runner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

impl Runner {
    pub async fn new() -> Self {
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

    pub async fn run(&self, module: &ShaderModule) -> Vec<ReadBackValue> {
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
            size: 4 * 2,
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
            .take(module.final_stack_data.len())
            .map(|&[x, y, z, w, offset, ..]| ReadBackValue {
                size: [x, y, z, w],
                values: values[offset as usize..offset as usize + (x * y * z * w) as usize]
                    .to_vec(),
            })
            .collect()
    }

    pub async fn run_string(
        &self,
        string: &str,
        left_to_right: bool,
    ) -> Result<(Vec<ReadBackValue>, ShaderModule), String> {
        let module = generate_module(
            parse_code(string, left_to_right)
                .map_err(|(str, span)| format!("'{}' {:?} '{}'", str, span.clone(), &str[span]))?,
        );
        log::debug!("{}", module.code);
        Ok((self.run(&module).await, module))
    }

    pub async fn run_string_and_get_string_output(
        &self,
        string: &str,
        left_to_right: bool,
    ) -> String {
        let (values, module) = match self.run_string(string, left_to_right).await {
            Ok(values) => values,
            Err(error) => return format!("{:?}", error),
        };

        let mut output = String::new();

        for (value, &is_string) in values.iter().zip(&module.final_stack_data) {
            let print = |output: &mut String, slice: &[f32]| {
                if is_string {
                    let chars: String = slice.iter().map(|&val| val as u8 as char).collect();
                    output.push_str(&format!("{:?}\n", chars));
                } else {
                    output.push_str(&format!("{:?}\n", slice))
                }
            };

            match value.size {
                [0, 0, 0, 0] => {
                    println!("empty")
                }
                [1, 1, 1, 1] => output.push_str(&format!("{}\n", value.values[0])),
                [_, 1, 1, 1] => {
                    print(&mut output, &value.values);
                }
                [x, _, 1, 1] => {
                    output.push('\n');
                    for chunk in value.values.chunks(x as usize) {
                        print(&mut output, chunk);
                    }
                    output.push('\n');
                }
                other => panic!("{:?}", other),
            }
        }

        output
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
