//! `_native` extension module: a uniform replay buffer that stores single
//! grayscale frames as `u8` and reconstructs `frame_stack`-deep stacks at
//! sample time. With capacity 100k and 84x84 frames this costs ~700MB
//! instead of the ~2.7GB a pre-stacked buffer would need.

use ndarray::{Array1, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray4, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<ReplayBuffer>()?;
	Ok(())
}

#[pyclass]
pub struct ReplayBuffer {
	capacity: usize,
	frame_h: usize,
	frame_w: usize,
	frame_stack: usize,
	frames: Vec<u8>,
	actions: Vec<i64>,
	rewards: Vec<f32>,
	dones: Vec<bool>,
	total_pushed: u64,
	rng: StdRng,
}

#[pymethods]
impl ReplayBuffer {
	#[new]
	#[pyo3(signature = (capacity, frame_h=84, frame_w=84, frame_stack=4, seed=42))]
	fn py_new(
		capacity: usize,
		frame_h: usize,
		frame_w: usize,
		frame_stack: usize,
		seed: u64,
	) -> Self {
		Self::new(capacity, frame_h, frame_w, frame_stack, seed)
	}

	fn __len__(&self) -> usize {
		(self.total_pushed as usize).min(self.capacity)
	}

	#[getter]
	fn capacity(&self) -> usize {
		self.capacity
	}

	fn push(
		&mut self,
		frame: PyReadonlyArray2<'_, u8>,
		action: i64,
		reward: f32,
		done: bool,
	) -> PyResult<()> {
		let view = frame.as_array();
		let shape = view.shape();
		if shape != [self.frame_h, self.frame_w] {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"expected frame shape ({}, {}), got {:?}",
				self.frame_h, self.frame_w, shape
			)));
		}
		let idx = (self.total_pushed % self.capacity as u64) as usize;
		let frame_size = self.frame_h * self.frame_w;
		let dst = &mut self.frames[idx * frame_size..(idx + 1) * frame_size];
		// `view.iter()` yields elements in row-major (C) order regardless of
		// underlying strides, so this is correct for non-contiguous inputs too.
		for (d, s) in dst.iter_mut().zip(view.iter()) {
			*d = *s;
		}
		self.actions[idx] = action;
		self.rewards[idx] = reward;
		self.dones[idx] = done;
		self.total_pushed += 1;
		Ok(())
	}

	/// Returns `(states, actions, rewards, next_states, dones)`.
	///
	/// - `states`, `next_states`: `(batch, frame_stack, H, W)` `uint8`
	/// - `actions`: `(batch,)` `int64`
	/// - `rewards`: `(batch,)` `float32`
	/// - `dones`: `(batch,)` `bool`
	#[allow(clippy::type_complexity)]
	fn sample<'py>(
		&mut self,
		py: Python<'py>,
		batch_size: usize,
	) -> PyResult<(
		Bound<'py, PyArray4<u8>>,
		Bound<'py, PyArray1<i64>>,
		Bound<'py, PyArray1<f32>>,
		Bound<'py, PyArray4<u8>>,
		Bound<'py, PyArray1<bool>>,
	)> {
		let (low, high) = self.sample_range()?;
		let stack = self.frame_stack;
		let h = self.frame_h;
		let w = self.frame_w;
		let frame_size = h * w;
		let stack_size = stack * frame_size;

		let mut states = vec![0u8; batch_size * stack_size];
		let mut next_states = vec![0u8; batch_size * stack_size];
		let mut actions = vec![0i64; batch_size];
		let mut rewards = vec![0f32; batch_size];
		let mut dones = vec![false; batch_size];

		for b in 0..batch_size {
			let i = self.rng.random_range(low..=high);
			self.fill_stack(i, &mut states[b * stack_size..(b + 1) * stack_size]);
			self.fill_stack(i + 1, &mut next_states[b * stack_size..(b + 1) * stack_size]);
			let buf_i = (i % self.capacity as u64) as usize;
			actions[b] = self.actions[buf_i];
			rewards[b] = self.rewards[buf_i];
			dones[b] = self.dones[buf_i];
		}

		let states_arr = Array4::from_shape_vec((batch_size, stack, h, w), states)
			.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
		let next_states_arr = Array4::from_shape_vec((batch_size, stack, h, w), next_states)
			.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
		let actions_arr = Array1::from_vec(actions);
		let rewards_arr = Array1::from_vec(rewards);
		let dones_arr = Array1::from_vec(dones);

		Ok((
			states_arr.into_pyarray(py),
			actions_arr.into_pyarray(py),
			rewards_arr.into_pyarray(py),
			next_states_arr.into_pyarray(py),
			dones_arr.into_pyarray(py),
		))
	}
}

impl ReplayBuffer {
	fn new(capacity: usize, frame_h: usize, frame_w: usize, frame_stack: usize, seed: u64) -> Self {
		let frame_size = frame_h * frame_w;
		Self {
			capacity,
			frame_h,
			frame_w,
			frame_stack,
			frames: vec![0u8; capacity * frame_size],
			actions: vec![0i64; capacity],
			rewards: vec![0f32; capacity],
			dones: vec![false; capacity],
			total_pushed: 0,
			rng: StdRng::seed_from_u64(seed),
		}
	}

	fn sample_range(&self) -> PyResult<(u64, u64)> {
		// Transition i needs frame i+1 to exist for next_state, so i must be
		// strictly less than total_pushed - 1.
		if self.total_pushed < 2 {
			return Err(pyo3::exceptions::PyRuntimeError::new_err(
				"replay buffer needs at least 2 entries before sampling",
			));
		}
		let low = self.total_pushed.saturating_sub(self.capacity as u64);
		let high = self.total_pushed - 2;
		if high < low {
			return Err(pyo3::exceptions::PyRuntimeError::new_err("no valid sample indices yet"));
		}
		Ok((low, high))
	}

	/// Fill `out` (length `frame_stack * frame_h * frame_w`) with the stack
	/// of frames whose newest element is the frame at logical index `idx`.
	/// Walks backward from `idx`; if a `done=True` is hit on the way, all
	/// older slots are zero-padded so the stack never crosses an episode
	/// boundary.
	fn fill_stack(&self, idx: u64, out: &mut [u8]) {
		let stack = self.frame_stack;
		let frame_size = self.frame_h * self.frame_w;
		let buffer_low = self.total_pushed.saturating_sub(self.capacity as u64);

		let mut in_episode = true;
		for k in (0..stack).rev() {
			let dst = &mut out[k * frame_size..(k + 1) * frame_size];
			let offset = (stack - 1 - k) as u64;
			if !in_episode || offset > idx {
				dst.fill(0);
				continue;
			}
			let logical = idx - offset;
			let buf_idx = (logical % self.capacity as u64) as usize;
			let src = &self.frames[buf_idx * frame_size..(buf_idx + 1) * frame_size];
			dst.copy_from_slice(src);

			// If the frame *before* this one was terminal, everything older
			// belongs to a prior episode — flip the flag for the next iteration.
			if k > 0 && logical > 0 {
				let prev = logical - 1;
				if prev >= buffer_low {
					let prev_buf = (prev % self.capacity as u64) as usize;
					if self.dones[prev_buf] {
						in_episode = false;
					}
				}
			}
		}
	}
}
