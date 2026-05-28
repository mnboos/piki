//! [`NalSlab`] and [`NalSlice`] — the core zero-copy buffer primitives.
//!
//! The VPU encoder (`libsrcampy`) reuses its output buffer on the next
//! `send_frame()` call, so we must copy the annex-B data out exactly once.
//! [`NalSlab`] owns that single allocation inside an `Arc<Vec<u8>>`.
//! [`NalSlice`] is a lightweight, reference-counted window into a slab and
//! can be passed around — including across Python — without any further
//! copying.
//!
//! Python API
//! ----------
//! ```python
//! slab = NalSlab(raw_bytes)           # one memcpy from Python bytes/bytearray/memoryview
//! slices = slab.split_annex_b()       # list[NalSlice], zero-copy views
//! for s in slices:
//!     print(s.nal_type, s.is_idr)
//!     raw = bytes(s)                  # copies from Arc slab to Python bytes
//!
//! # Hot-path helper: list[memoryview] → list[NalSlice] in one allocation
//! slices = piki_nal.slices_from_views(enc.encode_nv12(nv12))
//! ```

use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Find annex-B start code positions in `data`.
/// Returns `(offset_of_start_code, prefix_len)` pairs.
fn annex_b_starts(data: &[u8]) -> Vec<(usize, usize)> {
    let n = data.len();
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 2 < n {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                starts.push((i, 3));
                i += 3;
                continue;
            }
            if i + 3 < n && data[i + 2] == 0 && data[i + 3] == 1 {
                starts.push((i, 4));
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    starts
}

// ---------------------------------------------------------------------------
// NalSlab
// ---------------------------------------------------------------------------

/// Owns the raw encoded bytes for one VPU output frame.
///
/// Construct via `NalSlab(buf)` from Python where `buf` is any buffer-protocol
/// object (bytes, bytearray, memoryview).  The data is copied once into an
/// `Arc<Vec<u8>>` so it survives after the originating VPU buffer is reused.
#[pyclass(name = "NalSlab")]
pub struct NalSlab {
    pub(crate) data: Arc<Vec<u8>>,
}

#[pymethods]
impl NalSlab {
    /// Create a slab from a Python buffer (bytes / bytearray / memoryview).
    /// This is the **one permitted copy** in the zero-copy pipeline.
    #[new]
    fn new(py: Python<'_>, buf: &Bound<'_, PyAny>) -> PyResult<Self> {
        let data = if let Ok(bytes) = buf.cast_exact::<PyBytes>() {
            bytes.as_bytes().to_vec()
        } else {
            let pybuf = PyBuffer::<u8>::get(buf)?;
            let mut v = vec![0u8; pybuf.len_bytes()];
            pybuf.copy_to_slice(py, &mut v)?;
            v
        };
        Ok(NalSlab { data: Arc::new(data) })
    }

    /// Split the slab's annex-B bitstream into individual NAL unit slices.
    ///
    /// Returns a `list[NalSlice]` whose elements share ownership of this
    /// slab's buffer — zero additional copies.
    pub fn split_annex_b(&self) -> Vec<NalSlice> {
        let data = &self.data;
        let n = data.len();
        let starts = annex_b_starts(data);
        let mut slices = Vec::with_capacity(starts.len());
        for (k, &(off, prefix_len)) in starts.iter().enumerate() {
            let body_start = off + prefix_len;
            let body_end = if k + 1 < starts.len() { starts[k + 1].0 } else { n };
            if body_end > body_start {
                slices.push(NalSlice {
                    slab: Arc::clone(data),
                    start: body_start as u32,
                    end: body_end as u32,
                });
            }
        }
        slices
    }

    /// Create a single NalSlice covering the entire slab (no start-code split).
    pub fn as_slice(&self) -> NalSlice {
        NalSlice {
            slab: Arc::clone(&self.data),
            start: 0,
            end: self.data.len() as u32,
        }
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    fn __repr__(&self) -> String {
        format!("NalSlab(len={})", self.data.len())
    }
}

// ---------------------------------------------------------------------------
// NalSlice
// ---------------------------------------------------------------------------

/// A zero-copy, reference-counted window into a [`NalSlab`].
///
/// Common properties:
/// - `.nal_type: int`   — lower 5 bits of the first NAL byte
/// - `.is_idr: bool`    — True when nal_type == 5
/// - `.is_sps: bool`    — True when nal_type == 7
/// - `.is_pps: bool`    — True when nal_type == 8
/// - `.nbytes: int`     — length of the payload (no start code)
///
/// Use ``bytes(s)`` or ``s.tobytes()`` to materialise a Python ``bytes``
/// object (one copy from the Arc slab).
#[pyclass(name = "NalSlice")]
pub struct NalSlice {
    pub(crate) slab: Arc<Vec<u8>>,
    pub(crate) start: u32,
    pub(crate) end: u32,
}

impl NalSlice {
    /// Raw byte slice into the slab (Rust-internal, no copy).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.slab[self.start as usize..self.end as usize]
    }

    #[inline]
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    // Rust-side helpers with non-conflicting names (PyO3 getters use the
    // plain names `nal_type`, `is_idr`, etc. for Python).
    #[inline]
    pub fn nal_type_byte(&self) -> u8 {
        self.as_bytes().first().map(|b| b & 0x1F).unwrap_or(0)
    }

    #[inline]
    pub fn is_idr_bool(&self) -> bool {
        self.nal_type_byte() == 5
    }

    #[inline]
    pub fn is_sps_bool(&self) -> bool {
        self.nal_type_byte() == 7
    }

    #[inline]
    pub fn is_pps_bool(&self) -> bool {
        self.nal_type_byte() == 8
    }
}

#[pymethods]
impl NalSlice {
    /// NAL unit type (lower 5 bits of first byte), or 0 for empty slice.
    #[getter]
    fn nal_type(&self) -> u8 {
        self.nal_type_byte()
    }

    /// True if this is an IDR slice (nal_type == 5).
    #[getter]
    fn is_idr(&self) -> bool {
        self.is_idr_bool()
    }

    /// True if this is an SPS NAL (nal_type == 7).
    #[getter]
    fn is_sps(&self) -> bool {
        self.is_sps_bool()
    }

    /// True if this is a PPS NAL (nal_type == 8).
    #[getter]
    fn is_pps(&self) -> bool {
        self.is_pps_bool()
    }

    /// Length of the NAL payload in bytes (no start code).
    #[getter]
    fn nbytes(&self) -> usize {
        self.len()
    }

    /// Return the NAL payload as a Python `bytes` object (one copy from slab).
    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.as_bytes())
    }

    /// `bytes(nal_slice)` support.
    fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.as_bytes())
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        format!("NalSlice(nal_type={}, len={})", self.nal_type_byte(), self.len())
    }

    /// Integer index access: `nal_slice[0]` returns the first byte as `int`.
    fn __getitem__(&self, idx: isize) -> PyResult<u8> {
        let bytes = self.as_bytes();
        let len = bytes.len() as isize;
        let i = if idx < 0 { len + idx } else { idx };
        if i < 0 || i >= len {
            Err(pyo3::exceptions::PyIndexError::new_err("index out of range"))
        } else {
            Ok(bytes[i as usize])
        }
    }
}

// ---------------------------------------------------------------------------
// Module-level hot-path helper
// ---------------------------------------------------------------------------

/// Convert a `list[memoryview]` (from `HwH264Encoder.encode_nv12()`) into a
/// `list[NalSlice]` using a **single allocation**.
///
/// All views are concatenated into one `Arc<Vec<u8>>`, then zero-copy
/// `NalSlice` handles are returned.
///
/// ```python
/// nals_mv = enc.encode_nv12(nv12)              # list[memoryview]
/// slices  = piki_nal.slices_from_views(nals_mv) # list[NalSlice], one copy
/// ```
#[pyfunction]
pub fn slices_from_views(py: Python<'_>, views: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<NalSlice>> {
    if views.is_empty() {
        return Ok(vec![]);
    }

    // First pass: read all views into temporary buffers (can't avoid this
    // without unsafe pointer tricks on the Python buffer — one copy total).
    let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(views.len());
    let mut total = 0usize;
    for v in &views {
        let pybuf = PyBuffer::<u8>::get(v)?;
        let mut tmp = vec![0u8; pybuf.len_bytes()];
        pybuf.copy_to_slice(py, &mut tmp)?;
        total += tmp.len();
        bufs.push(tmp);
    }

    // Single Arc allocation for the whole frame.
    let mut combined = Vec::with_capacity(total);
    let mut offsets: Vec<(u32, u32)> = Vec::with_capacity(bufs.len());
    for chunk in bufs {
        let start = combined.len() as u32;
        combined.extend_from_slice(&chunk);
        offsets.push((start, combined.len() as u32));
    }
    let slab = Arc::new(combined);

    Ok(offsets
        .into_iter()
        .map(|(start, end)| NalSlice { slab: Arc::clone(&slab), start, end })
        .collect())
}
