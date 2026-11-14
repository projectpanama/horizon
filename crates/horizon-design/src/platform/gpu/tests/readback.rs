use std::{sync::mpsc::TryRecvError, time::Duration};

use vello::wgpu;

use super::super::validate_dimensions;

const BYTES_PER_PIXEL: u32 = 4;

/// The padded GPU copy layout and the tightly packed CPU output size.
#[derive(Debug, Clone, Copy)]
pub(super) struct ReadbackLayout {
    width: u32,
    height: u32,
    bytes_per_row: u32,
    padded_bytes_per_row: u32,
    buffer_size: u64,
    output_size: usize,
}

impl ReadbackLayout {
    /// Computes the GPU row stride, mapped buffer size, and packed output size.
    ///
    /// GPU copy rows are padded to [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`], while the returned
    /// CPU buffer contains only tightly packed RGBA8 rows. Dimensions and allocation sizes are
    /// checked against the device limits and the addressable CPU slice size.
    pub(super) fn new(limits: &wgpu::Limits, width: u32, height: u32) -> Result<Self, String> {
        validate_dimensions(limits, width, height)?;

        let bytes_per_row = width
            .checked_mul(BYTES_PER_PIXEL)
            .ok_or("test readback row size overflow")?;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = bytes_per_row
            .div_ceil(alignment)
            .checked_mul(alignment)
            .ok_or("aligned test readback row size overflow")?;
        let buffer_size = u64::from(padded_bytes_per_row)
            .checked_mul(u64::from(height))
            .ok_or("test readback buffer size overflow")?;

        if buffer_size > limits.max_buffer_size {
            return Err(format!(
                "test readback needs {buffer_size} bytes; GPU buffer limit is {}",
                limits.max_buffer_size
            ));
        }

        // A slice cannot contain more than isize::MAX bytes, including padding.
        let addressable_size = usize::try_from(buffer_size)
            .map_err(|_| "test readback exceeds the CPU address space")?;
        if addressable_size > isize::MAX as usize {
            return Err("test readback exceeds the CPU slice size limit".into());
        }

        let output_size = usize::try_from(u64::from(bytes_per_row) * u64::from(height))
            .map_err(|_| "test readback output size overflow")?;

        Ok(Self {
            width,
            height,
            bytes_per_row,
            padded_bytes_per_row,
            buffer_size,
            output_size,
        })
    }

    /// Removes GPU row padding from a mapped readback buffer.
    ///
    /// Returns an error when the mapped range does not match the layout's
    /// expected padded size.
    fn unpack(&self, mapped_bytes: &[u8]) -> Result<Vec<u8>, String> {
        if mapped_bytes.len() as u64 != self.buffer_size {
            return Err("test readback mapping has an unexpected length".into());
        }

        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(self.output_size)
            .map_err(|error| format!("can't allocate test readback pixels: {error}"))?;

        if self.bytes_per_row == self.padded_bytes_per_row {
            pixels.extend_from_slice(mapped_bytes);
        } else {
            for padded_row in mapped_bytes.chunks_exact(self.padded_bytes_per_row as usize) {
                pixels.extend_from_slice(&padded_row[..self.bytes_per_row as usize]);
            }
        }

        Ok(pixels)
    }
}

/// An in-flight test readback copy, independent of the renderer after submission.
///
/// Call [`Self::try_read`] from the event loop or [`Self::wait`] on a worker.
/// Dropping the readback cancels interest in its result and releases its buffer.
#[must_use]
pub(super) struct PixelReadback {
    buffer: wgpu::Buffer,
    layout: ReadbackLayout,
    mapping_result: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    submission: wgpu::SubmissionIndex,
    finished: bool,
}

impl PixelReadback {
    /// Returns pixels when ready without waiting for GPU execution.
    ///
    /// The device must be the device that created this readback.
    /// Returns `None` while the copy or mapping callback is pending. A completed
    /// readback is consumed logically: calling again after a result or mapping
    /// failure is an error.
    pub(super) fn try_read(&mut self, device: &wgpu::Device) -> Result<Option<Vec<u8>>, String> {
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| format!("GPU readback poll: {error}"))?;
        self.finish_if_ready()
    }

    /// Waits up to `timeout` for this copy and returns tightly packed pixels.
    ///
    /// The device must be the same device that created this readback. A timeout
    /// or mapping failure is returned as an error.
    pub(super) fn wait(
        mut self,
        device: &wgpu::Device,
        timeout: Duration,
    ) -> Result<Vec<u8>, String> {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(self.submission.clone()),
                timeout: Some(timeout),
            })
            .map_err(|error| format!("GPU readback wait: {error}"))?;
        self.finish_if_ready()?
            .ok_or_else(|| "GPU copy completed without a mapping result".into())
    }

    /// Queues a texture-to-buffer copy and starts mapping the destination.
    ///
    /// The returned readback is pending until the submitted GPU work completes.
    /// The texture must use a four-byte-per-pixel RGBA8 format matching `layout`.
    pub(super) fn copy(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        layout: ReadbackLayout,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback readback"),
            size: layout.buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test readback copy"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(layout.padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: layout.width,
                height: layout.height,
                depth_or_array_layers: 1,
            },
        );

        let (sender, mapping_result) = std::sync::mpsc::channel();

        // Register the map on this submission so waiting for its index also
        // drives the map callback; later submissions need not finish first.
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });

        let submission = queue.submit([encoder.finish()]);

        Self {
            buffer,
            layout,
            mapping_result,
            submission,
            finished: false,
        }
    }

    fn finish_if_ready(&mut self) -> Result<Option<Vec<u8>>, String> {
        if self.finished {
            return Err("test readback readback has already finished".into());
        }

        match self.mapping_result.try_recv() {
            Err(TryRecvError::Empty) => return Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.finished = true;
                return Err("test readback mapping callback disconnected".into());
            }
            Ok(mapping_result) => {
                self.finished = true;
                mapping_result.map_err(|error| format!("test readback mapping: {error}"))?;
            }
        }

        let pixels = {
            let mapped_bytes = self
                .buffer
                .get_mapped_range(..)
                .map_err(|error| format!("test readback mapped range: {error}"))?;
            self.layout.unpack(&mapped_bytes)
        };

        self.buffer.unmap();
        pixels.map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_padding_at_row_alignment_boundaries() {
        for width in [1, 63, 64, 65] {
            let layout = ReadbackLayout::new(&wgpu::Limits::default(), width, 3).unwrap();
            let mut mapped_bytes = vec![255; layout.buffer_size as usize];
            let mut expected = Vec::new();
            for (row_index, row) in mapped_bytes
                .chunks_exact_mut(layout.padded_bytes_per_row as usize)
                .enumerate()
            {
                row[..layout.bytes_per_row as usize].fill(row_index as u8);
                expected.extend(std::iter::repeat_n(
                    row_index as u8,
                    layout.bytes_per_row as usize,
                ));
            }
            assert_eq!(layout.unpack(&mapped_bytes).unwrap(), expected);
            assert_eq!(
                layout.padded_bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
                0
            );
        }
    }

    #[test]
    fn rejects_invalid_dimensions_and_buffer_limits() {
        let limits = wgpu::Limits::default();
        for (width, height) in [(0, 1), (1, 0), (limits.max_texture_dimension_2d + 1, 1)] {
            assert!(ReadbackLayout::new(&limits, width, height).is_err());
        }
        let limited = wgpu::Limits {
            max_buffer_size: 255,
            ..limits
        };
        assert!(ReadbackLayout::new(&limited, 1, 1).is_err());
    }

    #[test]
    fn calculates_large_sizes_without_u32_overflow() {
        let limits = wgpu::Limits {
            max_texture_dimension_2d: u32::MAX,
            max_buffer_size: u64::MAX,
            ..Default::default()
        };
        assert!(ReadbackLayout::new(&limits, u32::MAX, 1).is_err());
        assert!(ReadbackLayout::new(&limits, u32::MAX / 4, 1).is_err());
        if usize::BITS >= 64 {
            let layout = ReadbackLayout::new(&limits, 65_536, 65_536).unwrap();
            assert_eq!(layout.buffer_size, 1_u64 << 34);
            assert_eq!(layout.output_size as u64, 1_u64 << 34);
        }
    }

    #[test]
    fn rejects_incomplete_mapped_data() {
        let layout = ReadbackLayout::new(&wgpu::Limits::default(), 1, 2).unwrap();
        assert!(layout.unpack(&[0; 4]).is_err());
    }
}
