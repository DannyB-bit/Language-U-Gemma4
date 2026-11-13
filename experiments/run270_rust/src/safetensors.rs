//! Cross-platform SafeTensors reader with strict bounds checks.

use crate::codec::{bf16_to_f32, f32_to_bf16, fp16_to_f32};
use crate::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub size: usize,
}

pub struct SafeTensors {
    file: File,
    pub tensors: HashMap<String, TensorMeta>,
}

#[cfg(unix)]
fn read_exact_at_file(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, offset)
}

#[cfg(not(unix))]
fn read_exact_at_file(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut cloned = f.try_clone()?;
    cloned.seek(SeekFrom::Start(offset))?;
    cloned.read_exact(buf)
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let file_len = file.metadata().map_err(|e| format!("stat {}: {e}", path.display()))?.len();
        if file_len < 8 {
            return Err(format!("safetensors {} truncated (< 8 B)", path.display()));
        }
        let mut len8 = [0u8; 8];
        file.read_exact(&mut len8).map_err(|e| format!("header length: {e}"))?;
        let hlen_u64 = u64::from_le_bytes(len8);
        if hlen_u64 > 64 * 1024 * 1024 || 8 + hlen_u64 > file_len {
            return Err(format!("invalid safetensors header length {hlen_u64} for file size {file_len}"));
        }
        let hlen = hlen_u64 as usize;
        let mut hdr = vec![0u8; hlen];
        file.read_exact(&mut hdr).map_err(|e| format!("read header: {e}"))?;
        let parsed = json::parse(&hdr)?;
        let map = parsed.as_obj().ok_or("safetensors header is not a JSON object")?;
        let data_base = 8u64 + hlen_u64;
        let mut tensors = HashMap::with_capacity(map.len());
        for (name, v) in map {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("tensor {name}: missing dtype"))?
                .to_string();
            let shape_arr = v
                .get("shape")
                .and_then(|x| x.as_arr())
                .ok_or_else(|| format!("tensor {name}: missing shape"))?;
            let mut shape = Vec::with_capacity(shape_arr.len());
            for dim in shape_arr {
                shape.push(dim.as_usize().ok_or_else(|| format!("tensor {name}: invalid dim"))?);
            }
            let offs = v
                .get("data_offsets")
                .and_then(|x| x.as_arr())
                .ok_or_else(|| format!("tensor {name}: missing data_offsets"))?;
            if offs.len() != 2 {
                return Err(format!("tensor {name}: data_offsets must have 2 elements"));
            }
            let a = offs[0].as_u64().ok_or_else(|| format!("tensor {name}: bad start offset"))?;
            let b = offs[1].as_u64().ok_or_else(|| format!("tensor {name}: bad end offset"))?;
            if b < a || data_base.saturating_add(b) > file_len {
                return Err(format!("tensor {name}: out-of-bounds offsets [{a}, {b}]"));
            }
            let size = usize::try_from(b - a).map_err(|_| format!("tensor {name}: too large"))?;
            tensors.insert(
                name.clone(),
                TensorMeta {
                    dtype,
                    shape,
                    start: data_base + a,
                    size,
                },
            );
        }
        Ok(Self { file, tensors })
    }

    pub fn read_raw_bytes(&self, meta: &TensorMeta) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; meta.size];
        read_exact_at_file(&self.file, &mut buf, meta.start)
            .map_err(|e| format!("read tensor bytes at {}: {e}", meta.start))?;
        Ok(buf)
    }

    pub fn read_raw_slice(&self, meta: &TensorMeta, rel_offset: usize, len: usize) -> Result<Vec<u8>, String> {
        let end = rel_offset
            .checked_add(len)
            .ok_or_else(|| "safetensors slice overflow".to_string())?;
        if end > meta.size {
            return Err(format!("safetensors slice [{rel_offset}..{end}] exceeds tensor size {}", meta.size));
        }
        let mut buf = vec![0u8; len];
        read_exact_at_file(&self.file, &mut buf, meta.start + rel_offset as u64)
            .map_err(|e| format!("read tensor slice at {}: {e}", meta.start + rel_offset as u64))?;
        Ok(buf)
    }

    pub fn read_tensor_bf16_u16(&self, name: &str) -> Result<Vec<u16>, String> {
        let meta = self.tensors.get(name).ok_or_else(|| format!("tensor {name} not found"))?;
        match meta.dtype.as_str() {
            "BF16" => {
                if meta.size % 2 != 0 {
                    return Err(format!("{name}: odd BF16 byte count"));
                }
                let n = meta.size / 2;
                let mut out = Vec::with_capacity(n);
                let mut scratch = vec![0u8; 65536];
                let mut done = 0usize;
                while done < meta.size {
                    let take = (meta.size - done).min(scratch.len());
                    read_exact_at_file(&self.file, &mut scratch[..take], meta.start + done as u64)
                        .map_err(|e| format!("read bf16 {name}: {e}"))?;
                    for c in scratch[..take].chunks_exact(2) {
                        out.push(u16::from_le_bytes([c[0], c[1]]));
                    }
                    done += take;
                }
                Ok(out)
            }
            "F16" | "F32" => Ok(self.read_tensor_f32(name)?.into_iter().map(f32_to_bf16).collect()),
            other => Err(format!("{name}: unsupported dtype {other} for bf16")),
        }
    }

    pub fn read_tensor_rows_f32(&self, name: &str, row0: usize, n_rows: usize) -> Result<Vec<f32>, String> {
        let meta = self.tensors.get(name).ok_or_else(|| format!("tensor {name} not found"))?;
        if meta.shape.len() != 2 {
            return Err(format!("{name}: expected rank-2 tensor, got {:?}", meta.shape));
        }
        let d_out = meta.shape[0];
        let d_in = meta.shape[1];
        if row0 + n_rows > d_out {
            return Err(format!("{name}: row range [{row0}..{}) exceeds d_out {d_out}", row0 + n_rows));
        }
        let elems = n_rows.checked_mul(d_in).ok_or_else(|| format!("{name}: element count overflow"))?;
        let elem_bytes = match meta.dtype.as_str() {
            "BF16" | "F16" => 2usize,
            "F32" => 4usize,
            other => return Err(format!("{name}: unsupported dtype {other}")),
        };
        let byte_off = row0 * d_in * elem_bytes;
        let raw = self.read_raw_slice(meta, byte_off, elems * elem_bytes)?;
        match meta.dtype.as_str() {
            "BF16" => Ok(raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            "F16" => Ok(raw.chunks_exact(2).map(|c| fp16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            "F32" => Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            _ => unreachable!(),
        }
    }

    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let meta = self.tensors.get(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let raw = self.read_raw_bytes(meta)?;
        match meta.dtype.as_str() {
            "BF16" => {
                if raw.len() % 2 != 0 {
                    return Err(format!("{name}: odd BF16 byte count"));
                }
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            "F16" => {
                if raw.len() % 2 != 0 {
                    return Err(format!("{name}: odd F16 byte count"));
                }
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| fp16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            "F32" => {
                if raw.len() % 4 != 0 {
                    return Err(format!("{name}: unaligned F32 byte count"));
                }
                Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            other => Err(format!("{name}: unsupported dtype {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_bf16_and_f32_tensors_accurately() {
        let hdr = br#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"n":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut data = Vec::new();
        data.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        data.extend_from_slice(hdr);
        data.extend_from_slice(&0x3F80u16.to_le_bytes()); // 1.0 bf16
        data.extend_from_slice(&0xBF80u16.to_le_bytes()); // -1.0 bf16
        data.extend_from_slice(&3.5f32.to_le_bytes());

        let dir = std::env::temp_dir();
        let path = dir.join(format!("run270_st_test_{}.safetensors", std::process::id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        drop(f);

        let st = SafeTensors::open(&path).unwrap();
        assert_eq!(st.read_tensor_f32("w").unwrap(), vec![1.0, -1.0]);
        assert_eq!(st.read_tensor_f32("n").unwrap(), vec![3.5]);
        let _ = std::fs::remove_file(&path);
    }
}
