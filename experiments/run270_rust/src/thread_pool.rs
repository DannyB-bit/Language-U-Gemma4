//! Persistent fast-core worker pool & ARM64 NEON SIMD hot primitives (`V11`).
//!
//! Eliminates per-projection OS `pthread_create`/`clone` spawn storms (`std::thread::scope`)
//! and implements big.LITTLE performance-core affinity filtering (`fast_core_count`)
//! alongside explicit AArch64 NEON (`vfmaq_f32`, `vmovl_s8`, `vaddq_f32`) vector kernels.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Parse Linux cpuset range syntax (e.g., `"0-3,6,7"`) into sorted unique CPU IDs.
pub fn parse_cpu_list_str(raw: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in raw.trim().split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Some((a, b)) = p.split_once('-') {
            if let (Ok(s), Ok(e)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                if s <= e && e.saturating_sub(s) <= 1024 {
                    for cpu in s..=e {
                        out.push(cpu);
                    }
                }
            }
        } else if let Ok(c) = p.parse::<usize>() {
            out.push(c);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Filter `(cpu_id, max_freq_khz)` pairs to performance cores (`>= 80%` of peak clock).
/// Matches the mobile big.LITTLE scheduler fix verified in `ClusterNative.fastCoreCount`.
pub fn filter_fast_cores_by_freq(cpu_freqs: &[(usize, u64)]) -> Vec<usize> {
    if cpu_freqs.is_empty() {
        return Vec::new();
    }
    let max_khz = cpu_freqs.iter().map(|&(_, f)| f).max().unwrap_or(0);
    if max_khz == 0 {
        return cpu_freqs.iter().map(|&(c, _)| c).collect();
    }
    let threshold = (max_khz as u128 * 80 / 100) as u64;
    let fast: Vec<usize> = cpu_freqs
        .iter()
        .filter(|&&(_, f)| f >= threshold)
        .map(|&(c, _)| c)
        .collect();
    if fast.is_empty() {
        cpu_freqs.iter().map(|&(c, _)| c).collect()
    } else {
        fast
    }
}

/// Detect allowed performance CPU IDs on the current host.
pub fn detect_fast_cpus() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        let allowed: Vec<usize> = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|txt| {
                txt.lines()
                    .find(|l| l.starts_with("Cpus_allowed_list:"))
                    .and_then(|l| l.split_once(':'))
                    .map(|(_, rhs)| parse_cpu_list_str(rhs))
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| {
                let n = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
                (0..n).collect()
            });

        let mut freqs = Vec::with_capacity(allowed.len());
        for &cpu in &allowed {
            let p1 = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
            let p2 = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq");
            let khz = std::fs::read_to_string(&p1)
                .or_else(|_| std::fs::read_to_string(&p2))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            freqs.push((cpu, khz));
        }
        let fast = filter_fast_cores_by_freq(&freqs);
        if !fast.is_empty() {
            return fast;
        }
    }
    let n = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    (0..n).collect()
}

/// Best-effort thread affinity pin to `cpu_id` on Linux (`sched_setaffinity`).
#[inline]
pub fn pin_current_thread_to_cpu(cpu_id: usize) {
    #[cfg(target_os = "linux")]
    {
        if cpu_id < 1024 {
            #[repr(C)]
            struct CpuSet {
                bits: [usize; 1024 / (8 * std::mem::size_of::<usize>())],
            }
            extern "C" {
                fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> i32;
            }
            let word_bits = 8 * std::mem::size_of::<usize>();
            let mut set = CpuSet {
                bits: [0usize; 1024 / (8 * std::mem::size_of::<usize>())],
            };
            set.bits[cpu_id / word_bits] |= 1usize << (cpu_id % word_bits);
            unsafe {
                let _ = sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &set as *const CpuSet);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu_id;
    }
}

struct JobState {
    generation: u64,
    active_workers: usize,
    n_chunks: usize,
    chunk_size: usize,
    total_items: usize,
    fn_ptr: usize,
    ctx_ptr: usize,
    completed: usize,
    panicked: bool,
}

/// Persistent zero-spawn barrier worker pool for inner-loop tensor projections.
pub struct PersistentPool {
    max_workers: usize,
    next_chunk: AtomicUsize,
    state: Mutex<JobState>,
    cv_start: Condvar,
    cv_done: Condvar,
}

impl PersistentPool {
    pub fn new(max_workers: usize) -> std::sync::Arc<Self> {
        let fast_cpus = detect_fast_cpus();
        let extra = max_workers.saturating_sub(1);
        let pool = std::sync::Arc::new(Self {
            max_workers: max_workers.max(1),
            next_chunk: AtomicUsize::new(0),
            state: Mutex::new(JobState {
                generation: 0,
                active_workers: 0,
                n_chunks: 0,
                chunk_size: 1,
                total_items: 0,
                fn_ptr: 0,
                ctx_ptr: 0,
                completed: 0,
                panicked: false,
            }),
            cv_start: Condvar::new(),
            cv_done: Condvar::new(),
        });

        for wid in 0..extra {
            let p = std::sync::Arc::clone(&pool);
            let cpu_opt = if !fast_cpus.is_empty() {
                Some(fast_cpus[(wid + 1) % fast_cpus.len()])
            } else {
                None
            };
            std::thread::Builder::new()
                .name(format!("lu-v11-w{wid}"))
                .spawn(move || {
                    if let Some(cpu) = cpu_opt {
                        pin_current_thread_to_cpu(cpu);
                    }
                    p.worker_loop(wid);
                })
                .expect("spawn PersistentPool worker");
        }
        pool
    }

    fn worker_loop(&self, wid: usize) {
        let mut seen_gen = 0u64;
        loop {
            let (n_chunks, chunk_size, total_items, fn_ptr, ctx_ptr) = {
                let mut st = self.state.lock().unwrap();
                while st.generation == seen_gen {
                    st = self.cv_start.wait(st).unwrap();
                }
                seen_gen = st.generation;
                if wid >= st.active_workers {
                    continue;
                }
                (st.n_chunks, st.chunk_size, st.total_items, st.fn_ptr, st.ctx_ptr)
            };

            let tramp: unsafe fn(usize, usize, usize, usize) = unsafe { std::mem::transmute(fn_ptr) };
            let mut local_panic = false;
            loop {
                let cid = self.next_chunk.fetch_add(1, Ordering::Relaxed);
                if cid >= n_chunks {
                    break;
                }
                let start = cid * chunk_size;
                let end = (start + chunk_size).min(total_items);
                if start < end {
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                        tramp(ctx_ptr, cid, start, end);
                    }));
                    if res.is_err() {
                        local_panic = true;
                        break;
                    }
                }
            }

            let mut st = self.state.lock().unwrap();
            if local_panic {
                st.panicked = true;
            }
            st.completed += 1;
            if st.completed >= st.active_workers {
                self.cv_done.notify_one();
            }
        }
    }

    /// Execute `body(chunk_idx, start, end)` across `[0..total_items)` partitioned into `chunk_size` ranges
    /// using up to `threads` threads (caller + `threads - 1` persistent workers).
    pub fn par_for_chunks<F>(&self, total_items: usize, chunk_size: usize, threads: usize, body: F)
    where
        F: Fn(usize, usize, usize) + Sync,
    {
        if total_items == 0 {
            return;
        }
        let csz = chunk_size.max(1);
        let n_chunks = total_items.div_ceil(csz);
        let eff_threads = threads.clamp(1, self.max_workers).min(n_chunks);
        if eff_threads <= 1 {
            for cid in 0..n_chunks {
                let s = cid * csz;
                let e = (s + csz).min(total_items);
                body(cid, s, e);
            }
            return;
        }

        unsafe fn trampoline<F: Fn(usize, usize, usize) + Sync>(
            ctx: usize,
            cid: usize,
            start: usize,
            end: usize,
        ) {
            let f = &*(ctx as *const F);
            f(cid, start, end);
        }

        // Serialize concurrent outer callers cleanly via a dispatch lock so nested calls fall back safely.
        static DISPATCH_LOCK: Mutex<()> = Mutex::new(());
        let Ok(_outer_guard) = DISPATCH_LOCK.try_lock() else {
            for cid in 0..n_chunks {
                let s = cid * csz;
                let e = (s + csz).min(total_items);
                body(cid, s, e);
            }
            return;
        };

        let extra_workers = eff_threads - 1;
        self.next_chunk.store(0, Ordering::Release);
        {
            let mut st = self.state.lock().unwrap();
            st.active_workers = extra_workers;
            st.n_chunks = n_chunks;
            st.chunk_size = csz;
            st.total_items = total_items;
            st.fn_ptr = (trampoline::<F> as unsafe fn(usize, usize, usize, usize)) as usize;
            st.ctx_ptr = (&body as *const F) as usize;
            st.completed = 0;
            st.panicked = false;
            st.generation = st.generation.wrapping_add(1);
        }
        self.cv_start.notify_all();

        let mut caller_panic = false;
        loop {
            let cid = self.next_chunk.fetch_add(1, Ordering::Relaxed);
            if cid >= n_chunks {
                break;
            }
            let s = cid * csz;
            let e = (s + csz).min(total_items);
            if s < e {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(cid, s, e))).is_err() {
                    caller_panic = true;
                    break;
                }
            }
        }

        let panicked = {
            let mut st = self.state.lock().unwrap();
            while st.completed < extra_workers {
                st = self.cv_done.wait(st).unwrap();
            }
            st.fn_ptr = 0;
            st.ctx_ptr = 0;
            st.panicked || caller_panic
        };
        assert!(!panicked, "PersistentPool worker panicked during par_for_chunks");
    }
}

static GLOBAL_POOL: OnceLock<std::sync::Arc<PersistentPool>> = OnceLock::new();

#[inline]
pub fn global_pool() -> &'static PersistentPool {
    GLOBAL_POOL.get_or_init(|| {
        let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        PersistentPool::new(hw.clamp(2, 16))
    })
}

/// Convenience helper partitioning `[0..total_items)` across `threads` using the global persistent pool.
#[inline]
pub fn par_for_chunks<F>(total_items: usize, chunk_size: usize, threads: usize, body: F)
where
    F: Fn(usize, usize, usize) + Sync,
{
    if threads <= 1 || total_items <= chunk_size {
        let csz = chunk_size.max(1);
        let n_chunks = total_items.div_ceil(csz);
        for cid in 0..n_chunks {
            let s = cid * csz;
            let e = (s + csz).min(total_items);
            body(cid, s, e);
        }
    } else {
        global_pool().par_for_chunks(total_items, chunk_size, threads, body);
    }
}

// ============================================================================
// Explicit ARM64 NEON + Portable SIMD Hot-Path Kernels
// ============================================================================

/// Vectorized FP32 dot product (`4x float32x4_t` = 16 floats/iter on AArch64 NEON).
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut i = 0usize;
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < n {
            sum += *ap.add(i) * *bp.add(i);
            i += 1;
        }
        return sum;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut i = 0usize;
        while i + 4 <= n {
            s0 += a[i] * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
            i += 4;
        }
        let mut s = (s0 + s1) + (s2 + s3);
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }
}

/// Vectorized `dst[i] += alpha * src[i]` (`vfmaq_n_f32` 16 floats/iter on AArch64 NEON).
#[inline]
pub fn simd_axpy_f32(dst: &mut [f32], alpha: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let va = vdupq_n_f32(alpha);
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let mut i = 0usize;
        while i + 16 <= n {
            vst1q_f32(dp.add(i), vfmaq_f32(vld1q_f32(dp.add(i)), vld1q_f32(sp.add(i)), va));
            vst1q_f32(dp.add(i + 4), vfmaq_f32(vld1q_f32(dp.add(i + 4)), vld1q_f32(sp.add(i + 4)), va));
            vst1q_f32(dp.add(i + 8), vfmaq_f32(vld1q_f32(dp.add(i + 8)), vld1q_f32(sp.add(i + 8)), va));
            vst1q_f32(dp.add(i + 12), vfmaq_f32(vld1q_f32(dp.add(i + 12)), vld1q_f32(sp.add(i + 12)), va));
            i += 16;
        }
        while i + 4 <= n {
            vst1q_f32(dp.add(i), vfmaq_f32(vld1q_f32(dp.add(i)), vld1q_f32(sp.add(i)), va));
            i += 4;
        }
        while i < n {
            *dp.add(i) += alpha * *sp.add(i);
            i += 1;
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = n;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d += alpha * s;
        }
    }
}

/// Vectorized signed INT8 x FP32 dot product (`vmovl_s8` + `vcvtq_f32_s32` + `vfmaq_f32` on AArch64 NEON).
#[inline]
pub fn simd_dot_i8_f32(w_i8: &[i8], x_f32: &[f32]) -> f32 {
    debug_assert_eq!(w_i8.len(), x_f32.len());
    let n = w_i8.len();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let wp = w_i8.as_ptr();
        let xp = x_f32.as_ptr();
        let mut i = 0usize;
        while i + 16 <= n {
            let v8 = vld1q_s8(wp.add(i));
            let lo16 = vmovl_s8(vget_low_s8(v8));
            let hi16 = vmovl_s8(vget_high_s8(v8));
            let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
            let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
            let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
            let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
            acc0 = vfmaq_f32(acc0, f0, vld1q_f32(xp.add(i)));
            acc1 = vfmaq_f32(acc1, f1, vld1q_f32(xp.add(i + 4)));
            acc2 = vfmaq_f32(acc2, f2, vld1q_f32(xp.add(i + 8)));
            acc3 = vfmaq_f32(acc3, f3, vld1q_f32(xp.add(i + 12)));
            i += 16;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < n {
            sum += (*wp.add(i) as f32) * *xp.add(i);
            i += 1;
        }
        return sum;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut i = 0usize;
        while i + 4 <= n {
            s0 += (w_i8[i] as f32) * x_f32[i];
            s1 += (w_i8[i + 1] as f32) * x_f32[i + 1];
            s2 += (w_i8[i + 2] as f32) * x_f32[i + 2];
            s3 += (w_i8[i + 3] as f32) * x_f32[i + 3];
            i += 4;
        }
        let mut s = (s0 + s1) + (s2 + s3);
        while i < n {
            s += (w_i8[i] as f32) * x_f32[i];
            i += 1;
        }
        s
    }
}

/// Vectorized orthonormal Fast Walsh-Hadamard Transform (`bsize >= 8`, power of 2).
#[inline]
pub fn simd_fwht_block_inplace(x: &mut [f32], bsize: usize) -> Result<(), String> {
    if !bsize.is_power_of_two() || bsize == 0 || x.len() % bsize != 0 {
        return Err(format!("simd_fwht_block_inplace invalid bsize={bsize} len={}", x.len()));
    }
    let inv_sqrt = 1.0f32 / (bsize as f32).sqrt();
    #[cfg(target_arch = "aarch64")]
    if bsize >= 8 {
        unsafe {
            use std::arch::aarch64::*;
            let vscale = vdupq_n_f32(inv_sqrt);
            for chunk in x.chunks_exact_mut(bsize) {
                // Stages len=1 and len=2 fused per 4-float register (`H_4`)
                let cp = chunk.as_mut_ptr();
                let mut base = 0usize;
                while base < bsize {
                    let a0 = *cp.add(base);
                    let a1 = *cp.add(base + 1);
                    let a2 = *cp.add(base + 2);
                    let a3 = *cp.add(base + 3);
                    let b0 = a0 + a1;
                    let b1 = a0 - a1;
                    let b2 = a2 + a3;
                    let b3 = a2 - a3;
                    *cp.add(base) = b0 + b2;
                    *cp.add(base + 1) = b1 + b3;
                    *cp.add(base + 2) = b0 - b2;
                    *cp.add(base + 3) = b1 - b3;
                    base += 4;
                }
                // Stages len >= 4 in 128-bit NEON registers
                let mut len = 4usize;
                while len < bsize {
                    let step = len * 2;
                    let mut i = 0usize;
                    while i < bsize {
                        let mut j = 0usize;
                        while j < len {
                            let u = vld1q_f32(cp.add(i + j));
                            let v = vld1q_f32(cp.add(i + len + j));
                            vst1q_f32(cp.add(i + j), vaddq_f32(u, v));
                            vst1q_f32(cp.add(i + len + j), vsubq_f32(u, v));
                            j += 4;
                        }
                        i += step;
                    }
                    len = step;
                }
                let mut k = 0usize;
                while k < bsize {
                    vst1q_f32(cp.add(k), vmulq_f32(vld1q_f32(cp.add(k)), vscale));
                    k += 4;
                }
            }
        }
        return Ok(());
    }

    for chunk in x.chunks_exact_mut(bsize) {
        let mut len = 1usize;
        while len < bsize {
            let step = len * 2;
            let mut i = 0usize;
            while i < bsize {
                let (left, right) = chunk[i..i + step].split_at_mut(len);
                for j in 0..len {
                    let u = left[j];
                    let v = right[j];
                    left[j] = u + v;
                    right[j] = u - v;
                }
                i += step;
            }
            len = step;
        }
        for v in chunk.iter_mut() {
            *v *= inv_sqrt;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_list_and_fast_core_filter_handle_big_little_and_homogeneous() {
        assert_eq!(parse_cpu_list_str("0-3, 6, 7-8\n"), vec![0, 1, 2, 3, 6, 7, 8]);
        // Poco F1 / Snapdragon 845 big.LITTLE: 4 LITTLE @ 1.76 GHz, 4 Kryo Gold @ 2.80 GHz
        let big_little = vec![
            (0, 1_766_400),
            (1, 1_766_400),
            (2, 1_766_400),
            (3, 1_766_400),
            (4, 2_803_200),
            (5, 2_803_200),
            (6, 2_803_200),
            (7, 2_803_200),
        ];
        assert_eq!(filter_fast_cores_by_freq(&big_little), vec![4, 5, 6, 7]);
        // Raspberry Pi 4 homogeneous 4x Cortex-A72 @ 1.50 GHz -> keeps all 4 cores
        let pi4 = vec![(0, 1_500_000), (1, 1_500_000), (2, 1_500_000), (3, 1_500_000)];
        assert_eq!(filter_fast_cores_by_freq(&pi4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn persistent_pool_executes_chunks_deterministically_across_reuses() {
        let pool = PersistentPool::new(4);
        let mut out = vec![0usize; 257];
        let out_addr = out.as_mut_ptr() as usize;
        for round in 1..=5 {
            pool.par_for_chunks(257, 16, 4, |_cid, s, e| unsafe {
                let p = out_addr as *mut usize;
                for i in s..e {
                    *p.add(i) += i * round;
                }
            });
        }
        // sum_{r=1..5} r = 15
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i * 15, "mismatch at index {i}");
        }
    }

    #[test]
    fn simd_kernels_match_exact_scalar_references() {
        let a: Vec<f32> = (0..133).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.25).collect();
        let b: Vec<f32> = (0..133).map(|i| ((i * 11 % 19) as f32 - 9.0) * 0.15).collect();
        let ref_dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        assert!((simd_dot_f32(&a, &b) - ref_dot).abs() < 1e-4);

        let mut dst = a.clone();
        simd_axpy_f32(&mut dst, 0.375, &b);
        for i in 0..a.len() {
            assert!((dst[i] - (a[i] + 0.375 * b[i])).abs() < 1e-5);
        }

        let wi8: Vec<i8> = (0..133).map(|i| ((i * 13 % 251) as i16 - 125) as i8).collect();
        let ref_i8: f32 = wi8.iter().zip(a.iter()).map(|(&w, &x)| (w as f32) * x).sum();
        assert!((simd_dot_i8_f32(&wi8, &a) - ref_i8).abs() < 1e-3);

        // Orthonormal FWHT involution check: FWHT(FWHT(x)) == x
        let orig: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut w = orig.clone();
        simd_fwht_block_inplace(&mut w, 64).unwrap();
        simd_fwht_block_inplace(&mut w, 64).unwrap();
        for (x0, x1) in orig.iter().zip(w.iter()) {
            assert!((x0 - x1).abs() < 1e-5);
        }
    }
}
