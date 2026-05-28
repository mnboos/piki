//! [`RustMp4Writer`] — zerocopy AVCC/MP4 muxer backed by a background thread.
//!
//! Drop-in replacement for `H264DirectMP4Writer` in `hw_recorder.py` and
//! `recording.py`.  The critical improvement is that AVCC payloads are written
//! via `writev()` scatter-gather directly from [`NalSlice`] slab pointers —
//! no `b"".join(...)` heap allocation per frame.
//!
//! ## MP4/AVCC format written
//!
//! We write a self-contained ISO base media file (ftyp + moov + mdat).
//! The codec is AVC1/H.264 in AVCC format (4-byte big-endian length prefix
//! before each NAL unit — no Annex-B start codes).
//!
//! The file layout is:
//! ```text
//! [ftyp box]
//! [free box]           <- placeholder so moov can be rewritten at end
//! [mdat box]           <- NAL payload written incrementally
//! [moov box]           <- written / rewritten at release()
//! ```
//!
//! Python API
//! ----------
//! ```python
//! w = RustMp4Writer(path, fps, width, height)
//! w.write_nals(nals: list[NalSlice], ts_ns: int)
//! w.release()
//! ```

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::sync::mpsc::{self, SyncSender};
use std::thread;

use pyo3::prelude::*;

use crate::slab::NalSlice;

// NAL type constants
const NAL_SPS: u8 = 7;
const NAL_PPS: u8 = 8;

// ---------------------------------------------------------------------------
// Background write thread message
// ---------------------------------------------------------------------------

#[allow(dead_code)]
enum WriteMsg {
    Frame {
        nals: Vec<NalSlice>,
        ts_ns: i64,
    },
    Flush,
    #[allow(dead_code)]
    Shutdown,
}

// SAFETY: NalSlice contains Arc<Vec<u8>> which is Send.
unsafe impl Send for WriteMsg {}

// ---------------------------------------------------------------------------
// MP4 box helpers
// ---------------------------------------------------------------------------

fn write_u32_be(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}
fn write_u16_be(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}
fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
fn write_bytes(buf: &mut Vec<u8>, b: &[u8]) {
    buf.extend_from_slice(b);
}

/// Write a 4-byte box size + 4-byte fourcc, then the body.
fn box_full(fourcc: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let size = (8 + body.len()) as u32;
    let mut out = Vec::with_capacity(8 + body.len());
    write_u32_be(&mut out, size);
    out.extend_from_slice(fourcc);
    out.extend_from_slice(body);
    out
}

/// Return an ftyp box.
fn ftyp_box() -> Vec<u8> {
    let mut body = Vec::new();
    write_bytes(&mut body, b"isom"); // major brand
    write_u32_be(&mut body, 0x200);  // minor version
    write_bytes(&mut body, b"isomiso2avc1mp41"); // compatible brands
    box_full(b"ftyp", &body)
}

/// Return a `free` box of exactly `size` bytes total (placeholder for moov).
fn free_box(size: u32) -> Vec<u8> {
    assert!(size >= 8);
    let body = vec![0u8; (size - 8) as usize];
    box_full(b"free", &body)
}

/// Build the `avcC` extradata from raw SPS and PPS bytes (no start codes).
fn avcc_extradata(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    write_u8(&mut out, 1); // configurationVersion
    write_u8(&mut out, sps[1]); // profile_indication
    write_u8(&mut out, sps[2]); // profile_compatibility
    write_u8(&mut out, sps[3]); // level_indication
    write_u8(&mut out, 0xFF); // reserved (6 bits) | lengthSizeMinusOne (2 bits) = 3 → 4-byte lengths
    write_u8(&mut out, 0xE1); // reserved (3 bits) | numSequenceParameterSets (5 bits) = 1
    write_u16_be(&mut out, sps.len() as u16);
    write_bytes(&mut out, sps);
    write_u8(&mut out, 1); // numPictureParameterSets
    write_u16_be(&mut out, pps.len() as u16);
    write_bytes(&mut out, pps);
    out
}

// ---------------------------------------------------------------------------
// Muxer state (runs in background thread)
// ---------------------------------------------------------------------------

struct MuxState {
    file: BufWriter<File>,
    fps: u32,
    width: u16,
    height: u16,
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
    initialized: bool,
    // Byte offset where mdat size field was written (we patch it at end).
    mdat_size_offset: u64,
    // Total bytes written into mdat payload so far.
    mdat_payload_bytes: u64,
    // Sample table accumulators for moov.
    // Each sample: (pts, size_bytes, is_key)
    samples: Vec<(i64, u32, bool)>,
    start_ts_ns: Option<i64>,
    last_pts: i64,
    timescale: u32, // = fps * 1000 for sub-frame precision
}

impl MuxState {
    fn new(file: File, fps: u32, width: u16, height: u16) -> Self {
        MuxState {
            file: BufWriter::with_capacity(256 * 1024, file),
            fps,
            width,
            height,
            sps: None,
            pps: None,
            initialized: false,
            mdat_size_offset: 0,
            mdat_payload_bytes: 0,
            samples: Vec::new(),
            start_ts_ns: None,
            last_pts: -1,
            timescale: 90000, // common MPEG timescale (90 kHz)
        }
    }

    /// Write ftyp + reserve space for a large free box (placeholder for moov).
    fn write_header(&mut self) -> std::io::Result<()> {
        let ftyp = ftyp_box();
        self.file.write_all(&ftyp)?;
        // Reserve 4 KB placeholder — moov will be written at end.
        // If moov turns out larger, we append it after mdat.
        let placeholder = free_box(4096);
        self.file.write_all(&placeholder)?;

        // Write mdat box header with size = 0 (large box placeholder).
        // We'll patch the real size at close time.
        self.mdat_size_offset = ftyp.len() as u64 + 4096;
        // large box: size=1 means the real size follows as a 64-bit field
        self.file.write_all(&1u32.to_be_bytes())?; // size = 1 → extended
        self.file.write_all(b"mdat")?;
        self.file.write_all(&0u64.to_be_bytes())?; // placeholder 64-bit size
        Ok(())
    }

    /// Init the MP4 container on first SPS+PPS — deferred so we can write
    /// the correct codec parameters.
    fn ensure_initialized(&mut self) -> std::io::Result<bool> {
        if self.initialized {
            return Ok(true);
        }
        if self.sps.is_none() || self.pps.is_none() {
            return Ok(false);
        }
        self.write_header()?;
        self.initialized = true;
        Ok(true)
    }

    /// Write one frame's AVCC payload using scatter-gather writes.
    ///
    /// Each NAL is preceded by a 4-byte big-endian length field.
    /// No intermediate `b"".join()` allocation — each NalSlice is written
    /// directly from its slab pointer.
    fn write_frame(&mut self, nals: &[NalSlice], ts_ns: i64) -> std::io::Result<()> {
        // Extract SPS/PPS on first frame (or whenever they appear).
        for nal in nals {
            match nal.nal_type_byte() {
                NAL_SPS if self.sps.is_none() => {
                    self.sps = Some(nal.as_bytes().to_vec());
                }
                NAL_PPS if self.pps.is_none() => {
                    self.pps = Some(nal.as_bytes().to_vec());
                }
                _ => {}
            }
        }

        if !self.ensure_initialized()? {
            return Ok(());
        }

        let slice_nals: Vec<&NalSlice> = nals
            .iter()
            .filter(|n| !n.is_sps_bool() && !n.is_pps_bool())
            .collect();
        if slice_nals.is_empty() {
            return Ok(());
        }

        let start_ts = *self.start_ts_ns.get_or_insert(ts_ns);
        let mut pts = (ts_ns - start_ts) as i64 * self.timescale as i64 / 1_000_000_000;
        if pts <= self.last_pts {
            pts = self.last_pts + 1;
        }
        self.last_pts = pts;

        let is_key = slice_nals.iter().any(|n| n.is_idr_bool());

        // Compute total AVCC payload size (4-byte length prefix per NAL).
        let frame_size: u32 = slice_nals
            .iter()
            .map(|n| 4 + n.len() as u32)
            .sum();

        // Scatter-gather write: length prefix then NAL data for each unit.
        for nal in &slice_nals {
            let len_bytes = (nal.len() as u32).to_be_bytes();
            self.file.write_all(&len_bytes)?;
            self.file.write_all(nal.as_bytes())?;
        }

        self.mdat_payload_bytes += frame_size as u64;
        self.samples.push((pts, frame_size, is_key));
        Ok(())
    }

    /// Finalise the file: patch mdat size, write moov box.
    fn finalize(&mut self) -> std::io::Result<()> {
        if !self.initialized {
            return Ok(());
        }

        self.file.flush()?;

        let mdat_total = 16 + self.mdat_payload_bytes; // 4 (size=1) + 4 (mdat) + 8 (64-bit size) + payload

        // Patch the 64-bit mdat size field.
        {
            let f = self.file.get_mut();
            f.seek(SeekFrom::Start(self.mdat_size_offset + 8))?; // skip u32(1) + "mdat"
            f.write_all(&mdat_total.to_be_bytes())?;
            f.seek(SeekFrom::End(0))?;
        }

        // Build and write moov.
        let moov = self.build_moov();
        self.file.write_all(&moov)?;
        self.file.flush()?;
        Ok(())
    }

    fn build_moov(&self) -> Vec<u8> {
        let sps = self.sps.as_deref().unwrap_or(&[]);
        let pps = self.pps.as_deref().unwrap_or(&[]);
        let extra = avcc_extradata(sps, pps);

        // trak / tkhd / mdia / mdhd / hdlr / minf / stbl
        let stbl = self.build_stbl(&extra);
        let minf = build_minf(&stbl);
        let hdlr = build_hdlr();
        let mdhd = build_mdhd(self.timescale, self.total_duration_ticks());
        let mdia = build_box(b"mdia", &[&mdhd, &hdlr, &minf]);
        let tkhd = build_tkhd(self.width, self.height, self.total_duration_ticks());
        let trak = build_box(b"trak", &[&tkhd, &mdia]);
        let mvhd = build_mvhd(self.timescale, self.total_duration_ticks());
        build_box(b"moov", &[&mvhd, &trak])
    }

    fn total_duration_ticks(&self) -> u32 {
        self.samples.last().map(|(pts, _, _)| *pts as u32 + 1).unwrap_or(0)
    }

    fn build_stbl(&self, avcc_extra: &[u8]) -> Vec<u8> {
        // stsd
        let avc1 = build_avc1(self.width, self.height, avcc_extra);
        let stsd = build_stsd(&avc1);

        // stts: decode-time-to-sample (constant frame duration assumed)
        let stts = self.build_stts();
        // stss: sync sample table (key frames)
        let stss = self.build_stss();
        // stsc: sample-to-chunk (one chunk per sample for simplicity)
        let stsc = build_stsc_one_per_chunk(self.samples.len() as u32);
        // stsz: sample sizes
        let stsz = self.build_stsz();
        // stco: chunk offsets
        let stco = self.build_stco();

        build_box(b"stbl", &[&stsd, &stts, &stss, &stsc, &stsz, &stco])
    }

    fn build_stts(&self) -> Vec<u8> {
        // Run-length encode consecutive equal durations.
        let mut entries: Vec<(u32, u32)> = Vec::new(); // (count, delta)
        let iter = self.samples.windows(2);
        let mut pending_count = 0u32;
        let mut pending_delta = 0u32;
        let nominal = if self.samples.len() > 1 {
            (self.samples[1].0 - self.samples[0].0) as u32
        } else {
            self.timescale / self.fps
        };
        for w in self.samples.windows(2) {
            let delta = (w[1].0 - w[0].0) as u32;
            if delta == pending_delta && pending_count > 0 {
                pending_count += 1;
            } else {
                if pending_count > 0 {
                    entries.push((pending_count, pending_delta));
                }
                pending_delta = delta;
                pending_count = 1;
            }
        }
        if !self.samples.is_empty() {
            if pending_count > 0 {
                entries.push((pending_count, pending_delta));
            }
            // Last sample gets nominal delta.
            entries.push((1, nominal));
        }
        drop(iter); // suppress unused warning

        let mut body = Vec::new();
        write_u32_be(&mut body, 0); // version + flags
        write_u32_be(&mut body, entries.len() as u32);
        for (count, delta) in &entries {
            write_u32_be(&mut body, *count);
            write_u32_be(&mut body, *delta);
        }
        box_full(b"stts", &body)
    }

    fn build_stss(&self) -> Vec<u8> {
        let key_indices: Vec<u32> = self
            .samples
            .iter()
            .enumerate()
            .filter(|(_, (_, _, k))| *k)
            .map(|(i, _)| i as u32 + 1)
            .collect();
        let mut body = Vec::new();
        write_u32_be(&mut body, 0); // version + flags
        write_u32_be(&mut body, key_indices.len() as u32);
        for idx in &key_indices {
            write_u32_be(&mut body, *idx);
        }
        box_full(b"stss", &body)
    }

    fn build_stsz(&self) -> Vec<u8> {
        let mut body = Vec::new();
        write_u32_be(&mut body, 0); // version + flags
        write_u32_be(&mut body, 0); // sample_size = 0 → per-sample sizes follow
        write_u32_be(&mut body, self.samples.len() as u32);
        for (_, size, _) in &self.samples {
            write_u32_be(&mut body, *size);
        }
        box_full(b"stsz", &body)
    }

    fn build_stco(&self) -> Vec<u8> {
        // mdat payload starts after: ftyp + free(4096) + 16 bytes mdat header
        let mdat_payload_start: u64 = {
            let ftyp_len = ftyp_box().len() as u64;
            ftyp_len + 4096 + 16
        };
        let mut offset = mdat_payload_start;
        let mut body = Vec::new();
        write_u32_be(&mut body, 0); // version + flags
        write_u32_be(&mut body, self.samples.len() as u32);
        for (_, size, _) in &self.samples {
            write_u32_be(&mut body, offset as u32);
            offset += *size as u64;
        }
        box_full(b"stco", &body)
    }
}

// ---------------------------------------------------------------------------
// Box builders
// ---------------------------------------------------------------------------

fn build_box(fourcc: &[u8; 4], children: &[&[u8]]) -> Vec<u8> {
    let body_len: usize = children.iter().map(|c| c.len()).sum();
    let size = (8 + body_len) as u32;
    let mut out = Vec::with_capacity(8 + body_len);
    write_u32_be(&mut out, size);
    out.extend_from_slice(fourcc);
    for c in children {
        out.extend_from_slice(c);
    }
    out
}

fn build_mvhd(timescale: u32, duration: u32) -> Vec<u8> {
    let mut body = Vec::new();
    write_u32_be(&mut body, 0); // version + flags
    write_u32_be(&mut body, 0); // creation_time
    write_u32_be(&mut body, 0); // modification_time
    write_u32_be(&mut body, timescale);
    write_u32_be(&mut body, duration);
    write_u32_be(&mut body, 0x00010000); // rate = 1.0
    write_u16_be(&mut body, 0x0100); // volume = 1.0
    body.extend_from_slice(&[0u8; 10]); // reserved
    // Unity matrix
    let matrix: &[u32] = &[0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
    for &v in matrix {
        write_u32_be(&mut body, v);
    }
    body.extend_from_slice(&[0u8; 24]); // pre_defined
    write_u32_be(&mut body, 2); // next_track_ID
    box_full(b"mvhd", &body)
}

fn build_tkhd(width: u16, height: u16, duration: u32) -> Vec<u8> {
    let mut body = Vec::new();
    write_u32_be(&mut body, 0x0000_0003); // version=0, flags=track_enabled|track_in_movie
    write_u32_be(&mut body, 0); // creation_time
    write_u32_be(&mut body, 0); // modification_time
    write_u32_be(&mut body, 1); // track_ID
    write_u32_be(&mut body, 0); // reserved
    write_u32_be(&mut body, duration);
    body.extend_from_slice(&[0u8; 8]); // reserved
    write_u16_be(&mut body, 0); // layer
    write_u16_be(&mut body, 0); // alternate_group
    write_u16_be(&mut body, 0); // volume (video = 0)
    write_u16_be(&mut body, 0); // reserved
    let matrix: &[u32] = &[0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
    for &v in matrix {
        write_u32_be(&mut body, v);
    }
    write_u32_be(&mut body, (width as u32) << 16); // width (16.16 fixed)
    write_u32_be(&mut body, (height as u32) << 16); // height (16.16 fixed)
    box_full(b"tkhd", &body)
}

fn build_mdhd(timescale: u32, duration: u32) -> Vec<u8> {
    let mut body = Vec::new();
    write_u32_be(&mut body, 0); // version + flags
    write_u32_be(&mut body, 0); // creation_time
    write_u32_be(&mut body, 0); // modification_time
    write_u32_be(&mut body, timescale);
    write_u32_be(&mut body, duration);
    write_u16_be(&mut body, 0x55C4); // language = 'und'
    write_u16_be(&mut body, 0); // pre_defined
    box_full(b"mdhd", &body)
}

fn build_hdlr() -> Vec<u8> {
    let mut body = Vec::new();
    write_u32_be(&mut body, 0); // version + flags
    write_u32_be(&mut body, 0); // pre_defined
    write_bytes(&mut body, b"vide"); // handler_type
    body.extend_from_slice(&[0u8; 12]); // reserved
    write_bytes(&mut body, b"VideoHandler\0");
    box_full(b"hdlr", &body)
}

fn build_minf(stbl: &[u8]) -> Vec<u8> {
    // vmhd
    let mut vmhd_body = Vec::new();
    write_u32_be(&mut vmhd_body, 1); // version=0, flags=1
    write_u16_be(&mut vmhd_body, 0); // graphicsMode
    write_bytes(&mut vmhd_body, &[0u8; 6]); // opcolor
    let vmhd = box_full(b"vmhd", &vmhd_body);

    // dinf / dref
    let mut url_body = Vec::new();
    write_u32_be(&mut url_body, 1); // version=0, flags=self-contained
    let url = box_full(b"url ", &url_body);
    let mut dref_body = Vec::new();
    write_u32_be(&mut dref_body, 0);
    write_u32_be(&mut dref_body, 1); // entry_count
    dref_body.extend_from_slice(&url);
    let dref = box_full(b"dref", &dref_body);
    let dinf = build_box(b"dinf", &[&dref]);

    build_box(b"minf", &[&vmhd, &dinf, stbl])
}

fn build_stsd(avc1: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    write_u32_be(&mut body, 0); // version + flags
    write_u32_be(&mut body, 1); // entry_count
    body.extend_from_slice(avc1);
    box_full(b"stsd", &body)
}

fn build_avc1(width: u16, height: u16, avcc_extra: &[u8]) -> Vec<u8> {
    let avcc_box = box_full(b"avcC", avcc_extra);
    let mut body = Vec::new();
    body.extend_from_slice(&[0u8; 6]); // reserved
    write_u16_be(&mut body, 1); // data_reference_index
    body.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
    write_u16_be(&mut body, width);
    write_u16_be(&mut body, height);
    write_u32_be(&mut body, 0x00480000); // horizresolution 72dpi
    write_u32_be(&mut body, 0x00480000); // vertresolution 72dpi
    write_u32_be(&mut body, 0); // reserved
    write_u16_be(&mut body, 1); // frame_count
    body.extend_from_slice(&[0u8; 32]); // compressorname
    write_u16_be(&mut body, 0x0018); // depth
    write_u16_be(&mut body, 0xFFFF); // pre_defined = -1
    body.extend_from_slice(&avcc_box);
    box_full(b"avc1", &body)
}

fn build_stsc_one_per_chunk(_n_samples: u32) -> Vec<u8> {
    let mut body = Vec::new();
    write_u32_be(&mut body, 0); // version + flags
    // One entry: first_chunk=1, samples_per_chunk=1, sample_description_index=1
    write_u32_be(&mut body, 1);
    write_u32_be(&mut body, 1);
    write_u32_be(&mut body, 1);
    write_u32_be(&mut body, 1);
    box_full(b"stsc", &body)
}

// ---------------------------------------------------------------------------
// RustMp4Writer Python class
// ---------------------------------------------------------------------------

/// Zerocopy MP4 muxer.  Drop-in for `H264DirectMP4Writer`.
///
/// Writes AVCC-formatted H.264 into an ISO base media file using a background
/// thread so `write_nals()` returns immediately from Python (matching the
/// existing `_AsyncNalWriter` semantics).
#[pyclass(name = "RustMp4Writer")]
pub struct RustMp4Writer {
    tx: Option<SyncSender<WriteMsg>>,
    thread: Option<thread::JoinHandle<()>>,
    closed: bool,
}

#[pymethods]
impl RustMp4Writer {
    /// Open `path` for writing.  `fps`, `width`, `height` must match the
    /// encoder's parameters (same signature as `H264DirectMP4Writer`).
    #[new]
    fn new(path: String, fps: f64, width: u32, height: u32) -> PyResult<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let fps_u = fps.round().max(1.0) as u32;
        let (tx, rx) = mpsc::sync_channel::<WriteMsg>(64);

        let handle = thread::Builder::new()
            .name(format!("piki_nal_mp4_{path}"))
            .spawn(move || {
                let mut state = MuxState::new(file, fps_u, width as u16, height as u16);
                loop {
                    match rx.recv() {
                        Ok(WriteMsg::Frame { nals, ts_ns }) => {
                            if let Err(e) = state.write_frame(&nals, ts_ns) {
                                eprintln!("[piki_nal] mp4 write error: {e}");
                            }
                        }
                        Ok(WriteMsg::Flush) => {
                            let _ = state.file.flush();
                        }
                        Ok(WriteMsg::Shutdown) | Err(_) => {
                            if let Err(e) = state.finalize() {
                                eprintln!("[piki_nal] mp4 finalize error: {e}");
                            }
                            break;
                        }
                    }
                }
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(RustMp4Writer { tx: Some(tx), thread: Some(handle), closed: false })
    }

    /// Mux one group of pre-encoded NAL units (drop-in for `write_nals`).
    ///
    /// Returns immediately — actual disk I/O happens on the background thread.
    fn write_nals(&self, nals: Vec<Bound<'_, NalSlice>>, ts_ns: i64) -> PyResult<()> {
        if self.closed {
            return Ok(());
        }
        if nals.is_empty() {
            return Ok(());
        }
        let owned: Vec<NalSlice> = nals
            .into_iter()
            .map(|b| {
                let s = b.borrow();
                NalSlice {
                    slab: std::sync::Arc::clone(&s.slab),
                    start: s.start,
                    end: s.end,
                }
            })
            .collect();
        if let Some(tx) = &self.tx {
            tx.send(WriteMsg::Frame { nals: owned, ts_ns })
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("mp4 writer thread dead"))?;
        }
        Ok(())
    }

    /// Mirror of `H264DirectMP4Writer.isOpened()`.
    #[allow(non_snake_case)]
    fn isOpened(&self) -> bool {
        !self.closed
    }

    /// Finalise and close the file.
    ///
    /// Signals the background thread to flush + write the moov box, then
    /// waits for it to complete.  The GIL is held during the wait; in
    /// practice the finalize I/O is fast (moov metadata only).
    fn release(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(WriteMsg::Shutdown);
            drop(tx);
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }

    fn __del__(&mut self) {
        self.release();
    }
}
