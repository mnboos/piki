mod mp4;
mod ring;
mod slab;

use pyo3::prelude::*;

use crate::mp4::RustMp4Writer;
use crate::ring::RustNalRingBuffer;
use crate::slab::{NalSlab, NalSlice, slices_from_views};

/// piki_nal — zero-copy NAL buffer primitives for the piki pipeline.
///
/// Key types
/// ---------
/// ``NalSlab``            — owns one VPU output frame (one memcpy from Python).
/// ``NalSlice``           — zero-copy, reference-counted view into a NalSlab.
///                          Implements the buffer protocol so ``bytes(s)`` and
///                          ``memoryview(s)`` work without a second copy.
/// ``RustNalRingBuffer``  — drop-in for ``H264RollingBuffer``; stores Arc
///                          handles instead of ``list[bytes]``.
/// ``RustMp4Writer``      — drop-in for ``H264DirectMP4Writer``; writes AVCC
///                          via scatter-gather I/O on a background thread.
///
/// Module-level helpers
/// --------------------
/// ``slices_from_views(views)``  — convert ``list[memoryview]`` from
///                                 ``HwH264Encoder.encode_nv12()`` into a
///                                 ``list[NalSlice]`` with a single copy.
#[pymodule]
fn piki_nal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NalSlab>()?;
    m.add_class::<NalSlice>()?;
    m.add_class::<RustNalRingBuffer>()?;
    m.add_class::<RustMp4Writer>()?;
    m.add_function(wrap_pyfunction!(slices_from_views, m)?)?;
    Ok(())
}
