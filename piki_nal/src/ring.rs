//! [`RustNalRingBuffer`] — zero-copy rolling pre-buffer of H.264 NAL units.
//!
//! This is a drop-in replacement for the Python `H264RollingBuffer`.  The key
//! difference is that NAL data is **never copied** after the initial
//! `slices_from_views()` call: the ring stores `Arc` handles, and
//! `snapshot()` returns `Arc` clones, not `bytes` objects.
//!
//! Python API
//! ----------
//! ```python
//! # Construction: same args as H264RollingBuffer (channel handled in Python)
//! ring = RustNalRingBuffer()
//!
//! # push: accepts list[memoryview] returned by HwH264Encoder.encode_nv12()
//! nals: list[NalSlice] = ring.push(nals_mv, capture_ns, max_seconds)
//!
//! # snapshot: returns IDR-aligned window, zero copy
//! frames: list[tuple[int, list[NalSlice]]] = ring.snapshot(seconds)
//! ```

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use pyo3::prelude::*;

use crate::slab::{NalSlice, slices_from_views};

const NOMINAL_FPS: usize = 30;

// One entry in the ring deque.
struct RingEntry {
    ts_ns: i64,
    nals: Vec<NalSlice>,
    is_idr: bool,
}

// We need RingEntry to be Send — NalSlice contains Arc<Vec<u8>> which is Send.
// SAFETY: NalSlice only holds an Arc<Vec<u8>> (immutable after creation) and
// two u32 offsets.
unsafe impl Send for RingEntry {}

struct Inner {
    ring: VecDeque<RingEntry>,
}

/// Rust replacement for Python `H264RollingBuffer`.
///
/// Stores one `Arc<Vec<u8>>` slab per frame (the single copy from VPU
/// output), then hands out `NalSlice` handles that reference that slab.
/// `snapshot()` clones Arc handles — no data is copied.
#[pyclass(name = "RustNalRingBuffer")]
pub struct RustNalRingBuffer {
    inner: Arc<Mutex<Inner>>,
}

#[pymethods]
impl RustNalRingBuffer {
    #[new]
    fn new() -> Self {
        RustNalRingBuffer {
            inner: Arc::new(Mutex::new(Inner {
                ring: VecDeque::new(),
            })),
        }
    }

    /// Push one group of NAL units into the ring.
    ///
    /// `nals_mv` — `list[memoryview]` returned by `HwH264Encoder.encode_nv12()`.
    /// `ts_ns`   — capture timestamp in nanoseconds (monotonic).
    /// `max_seconds` — rolling window size; older frames are pruned.
    ///
    /// Returns the same NALs as `list[NalSlice]` (zero-copy after the initial
    /// copy from the VPU output buffer).
    fn push(
        &self,
        py: Python<'_>,
        nals_mv: Vec<Bound<'_, PyAny>>,
        ts_ns: i64,
        max_seconds: f64,
    ) -> PyResult<Vec<NalSlice>> {
        if nals_mv.is_empty() {
            return Ok(vec![]);
        }

        let nals = slices_from_views(py, nals_mv)?;

        let is_idr = nals.iter().any(|s| s.is_idr_bool());

        let cutoff_ns = ts_ns - (max_seconds * 1_000_000_000.0) as i64;
        let max_frames = (max_seconds * NOMINAL_FPS as f64) as usize + NOMINAL_FPS;

        // Clone slices for storage (Arc clone, not data copy).
        let stored: Vec<NalSlice> = nals
            .iter()
            .map(|s| NalSlice {
                slab: Arc::clone(&s.slab),
                start: s.start,
                end: s.end,
            })
            .collect();

        let mut inner = self.inner.lock();
        inner.ring.push_back(RingEntry { ts_ns, nals: stored, is_idr });

        // Prune by time.
        while inner.ring.front().map(|e| e.ts_ns < cutoff_ns).unwrap_or(false) {
            inner.ring.pop_front();
        }
        // Hard frame cap.
        while inner.ring.len() > max_frames {
            inner.ring.pop_front();
        }

        Ok(nals)
    }

    /// Return an IDR-aligned window covering the last `seconds` seconds.
    ///
    /// Mirrors `H264RollingBuffer.snapshot()` exactly.  Returns
    /// `list[tuple[int, list[NalSlice]]]` — all via Arc clones, zero copy.
    fn snapshot(&self, seconds: f64) -> Vec<(i64, Vec<NalSlice>)> {
        // Use the most recent entry's timestamp as a proxy for "now" so we
        // don't need a syscall.  All timestamps are time.monotonic_ns() values
        // passed in from Python.
        let inner = self.inner.lock();
        if inner.ring.is_empty() {
            return vec![];
        }

        // Approximate now_ns as the timestamp of the most recent entry.
        let now_ns = inner.ring.back().map(|e| e.ts_ns).unwrap_or(0);
        let cutoff_ns = now_ns - (seconds * 1_000_000_000.0) as i64;

        // Find first frame at or after cutoff.
        let window_start = inner
            .ring
            .iter()
            .position(|e| e.ts_ns >= cutoff_ns)
            .unwrap_or(inner.ring.len());

        // Walk backwards to find last IDR at or before window_start.
        let idr_idx = (0..window_start.min(inner.ring.len()))
            .rev()
            .find(|&i| inner.ring[i].is_idr)
            .or_else(|| inner.ring.iter().position(|e| e.is_idr));

        let Some(idr_idx) = idr_idx else {
            return vec![];
        };

        inner.ring.iter().skip(idr_idx).map(|entry| {
            let cloned: Vec<NalSlice> = entry.nals.iter().map(|s| NalSlice {
                slab: Arc::clone(&s.slab),
                start: s.start,
                end: s.end,
            }).collect();
            (entry.ts_ns, cloned)
        }).collect()
    }

    /// Number of frames currently in the ring.
    fn __len__(&self) -> usize {
        self.inner.lock().ring.len()
    }

    fn __repr__(&self) -> String {
        format!("RustNalRingBuffer(frames={})", self.inner.lock().ring.len())
    }
}
