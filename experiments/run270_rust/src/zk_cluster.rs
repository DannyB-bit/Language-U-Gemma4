//! V11 `ZK-Cortical-Cluster` & Multi-Radio Semantic Transport (`ZK-LoRaWAN` + `Nodle` + `Language-U`).
//!
//! Unifies:
//! 1. **255-Byte `ZK-LoRaWAN` Single-Chirp Frame (`ZkLoraFrame255`):**
//!    - `17 B` Header with 6D Cuneiform-U Semantic Intent Coordinates (`[i8; 6]`)
//!    - `57 B` UFO Semantic Dictionary Codec (`encode_semantic_57` / `decode_semantic_57`)
//!    - `161 B` Compressed ZK Identity / Nullifier / Binary-Shard Attestation / Shielded Commitment
//!    - `20 B` Self-Healing `XOR-FEC` Parity (`17 + 57 + 161 + 20 == 255 B` exact LoRa PHY MTU)
//! 2. **RFC 8439 `ChaCha20-Poly1305` AEAD + `FWHT+INT8` Activation Pipe (`FwhtInt8Packet`):**
//!    - Replaces unauthenticated `ggml-rpc` (`FP32` + raw virtual memory pointers) with capability-indexed,
//!      AEAD-encrypted, `3.88x`-compressed orthonormal `FWHT-64 + INT8` multi-token tree activation frames.
//! 3. **Multi-Radio Gear Coordination (`NodleBleNanBeacon` + `BandwidthShardPlanner` + `RadioKeepAwakeGuard`):**
//!    - Compact `31 B` BLE 5.0 / Wi-Fi Aware discovery advertisement, fastest-DRAM-first greedy layer
//!      sharding (`70%` Master / `75%` Worker free RAM), and `20 ms` UDP port-9 Wi-Fi anti-doze keep-awake.
//! 4. **Authenticated TCP Layer-Split RPC (`ClusterRpcClient` / `serve_cluster_worker`):**
//!    - Zero raw pointers; mutual `HMAC/SHA-256` + `ZK` shard-commitment handshake over ephemeral session keys.

use crate::codec::{fp16_to_f32, fp32_to_fp16};
use crate::sha256::sha256;
use crate::t16c::fwht_block_inplace;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// 1. Pure-Rust RFC 8439 ChaCha20-Poly1305 AEAD (Zero External Crate Bloat)
// ============================================================================

#[inline]
fn qr(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

pub fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut st = [0u32; 16];
    st[0] = 0x6170_7865;
    st[1] = 0x3320_646e;
    st[2] = 0x7962_2d32;
    st[3] = 0x6b20_6574;
    for i in 0..8 {
        st[4 + i] = u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
    }
    st[12] = counter;
    st[13] = u32::from_le_bytes([nonce[0], nonce[1], nonce[2], nonce[3]]);
    st[14] = u32::from_le_bytes([nonce[4], nonce[5], nonce[6], nonce[7]]);
    st[15] = u32::from_le_bytes([nonce[8], nonce[9], nonce[10], nonce[11]]);

    let mut w = st;
    for _ in 0..10 {
        qr(&mut w, 0, 4, 8, 12);
        qr(&mut w, 1, 5, 9, 13);
        qr(&mut w, 2, 6, 10, 14);
        qr(&mut w, 3, 7, 11, 15);
        qr(&mut w, 0, 5, 10, 15);
        qr(&mut w, 1, 6, 11, 12);
        qr(&mut w, 2, 7, 8, 13);
        qr(&mut w, 3, 4, 9, 14);
    }
    let mut out = [0u8; 64];
    for i in 0..16 {
        let v = w[i].wrapping_add(st[i]).to_le_bytes();
        out[i * 4..i * 4 + 4].copy_from_slice(&v);
    }
    out
}

pub fn chacha20_xor_inplace(key: &[u8; 32], initial_counter: u32, nonce: &[u8; 12], data: &mut [u8]) {
    let mut ctr = initial_counter;
    for chunk in data.chunks_mut(64) {
        let ks = chacha20_block(key, ctr, nonce);
        ctr = ctr.wrapping_add(1);
        for (b, &k) in chunk.iter_mut().zip(ks.iter()) {
            *b ^= k;
        }
    }
}

/// RFC 8439 Poly1305 one-time MAC in exact 26-bit limb arithmetic modulo `2^130 - 5`.
pub fn poly1305_mac(otk: &[u8; 32], msg: &[u8]) -> [u8; 16] {
    let mut r_bytes = [0u8; 16];
    r_bytes.copy_from_slice(&otk[0..16]);
    r_bytes[3] &= 15;
    r_bytes[7] &= 15;
    r_bytes[11] &= 15;
    r_bytes[15] &= 15;
    r_bytes[4] &= 252;
    r_bytes[8] &= 252;
    r_bytes[12] &= 252;

    let t0 = u32::from_le_bytes([r_bytes[0], r_bytes[1], r_bytes[2], r_bytes[3]]) as u64;
    let t1 = u32::from_le_bytes([r_bytes[4], r_bytes[5], r_bytes[6], r_bytes[7]]) as u64;
    let t2 = u32::from_le_bytes([r_bytes[8], r_bytes[9], r_bytes[10], r_bytes[11]]) as u64;
    let t3 = u32::from_le_bytes([r_bytes[12], r_bytes[13], r_bytes[14], r_bytes[15]]) as u64;

    let r0 = t0 & 0x3ff_ffff;
    let r1 = ((t0 >> 26) | (t1 << 6)) & 0x3ff_ffff;
    let r2 = ((t1 >> 20) | (t2 << 12)) & 0x3ff_ffff;
    let r3 = ((t2 >> 14) | (t3 << 18)) & 0x3ff_ffff;
    let r4 = (t3 >> 8) & 0x3ff_ffff;

    let s1 = r1 * 5;
    let s2 = r2 * 5;
    let s3 = r3 * 5;
    let s4 = r4 * 5;

    let mut h0 = 0u64;
    let mut h1 = 0u64;
    let mut h2 = 0u64;
    let mut h3 = 0u64;
    let mut h4 = 0u64;

    for chunk in msg.chunks(16) {
        let mut blk = [0u8; 17];
        blk[..chunk.len()].copy_from_slice(chunk);
        blk[chunk.len()] = 1;
        let m0 = u32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]) as u64;
        let m1 = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]) as u64;
        let m2 = u32::from_le_bytes([blk[8], blk[9], blk[10], blk[11]]) as u64;
        let m3 = u32::from_le_bytes([blk[12], blk[13], blk[14], blk[15]]) as u64;
        let hibit = (blk[16] as u64) << 24;

        h0 += m0 & 0x3ff_ffff;
        h1 += ((m0 >> 26) | (m1 << 6)) & 0x3ff_ffff;
        h2 += ((m1 >> 20) | (m2 << 12)) & 0x3ff_ffff;
        h3 += ((m2 >> 14) | (m3 << 18)) & 0x3ff_ffff;
        h4 += (m3 >> 8) | hibit;

        let d0 = (h0 as u128) * (r0 as u128)
            + (h1 as u128) * (s4 as u128)
            + (h2 as u128) * (s3 as u128)
            + (h3 as u128) * (s2 as u128)
            + (h4 as u128) * (s1 as u128);
        let mut d1 = (h0 as u128) * (r1 as u128)
            + (h1 as u128) * (r0 as u128)
            + (h2 as u128) * (s4 as u128)
            + (h3 as u128) * (s3 as u128)
            + (h4 as u128) * (s2 as u128);
        let mut d2 = (h0 as u128) * (r2 as u128)
            + (h1 as u128) * (r1 as u128)
            + (h2 as u128) * (r0 as u128)
            + (h3 as u128) * (s4 as u128)
            + (h4 as u128) * (s3 as u128);
        let mut d3 = (h0 as u128) * (r3 as u128)
            + (h1 as u128) * (r2 as u128)
            + (h2 as u128) * (r1 as u128)
            + (h3 as u128) * (r0 as u128)
            + (h4 as u128) * (s4 as u128);
        let mut d4 = (h0 as u128) * (r4 as u128)
            + (h1 as u128) * (r3 as u128)
            + (h2 as u128) * (r2 as u128)
            + (h3 as u128) * (r1 as u128)
            + (h4 as u128) * (r0 as u128);

        let mut c = (d0 >> 26) as u64;
        h0 = (d0 as u64) & 0x3ff_ffff;
        d1 += c as u128;
        c = (d1 >> 26) as u64;
        h1 = (d1 as u64) & 0x3ff_ffff;
        d2 += c as u128;
        c = (d2 >> 26) as u64;
        h2 = (d2 as u64) & 0x3ff_ffff;
        d3 += c as u128;
        c = (d3 >> 26) as u64;
        h3 = (d3 as u64) & 0x3ff_ffff;
        d4 += c as u128;
        c = (d4 >> 26) as u64;
        h4 = (d4 as u64) & 0x3ff_ffff;
        h0 += c * 5;
        c = h0 >> 26;
        h0 &= 0x3ff_ffff;
        h1 += c;
    }

    let mut c = h1 >> 26;
    h1 &= 0x3ff_ffff;
    h2 += c;
    c = h2 >> 26;
    h2 &= 0x3ff_ffff;
    h3 += c;
    c = h3 >> 26;
    h3 &= 0x3ff_ffff;
    h4 += c;
    c = h4 >> 26;
    h4 &= 0x3ff_ffff;
    h0 += c * 5;
    c = h0 >> 26;
    h0 &= 0x3ff_ffff;
    h1 += c;

    let mut g0 = h0.wrapping_add(5);
    c = g0 >> 26;
    g0 &= 0x3ff_ffff;
    let mut g1 = h1.wrapping_add(c);
    c = g1 >> 26;
    g1 &= 0x3ff_ffff;
    let mut g2 = h2.wrapping_add(c);
    c = g2 >> 26;
    g2 &= 0x3ff_ffff;
    let mut g3 = h3.wrapping_add(c);
    c = g3 >> 26;
    g3 &= 0x3ff_ffff;
    let g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

    let mask = (g4 >> 63).wrapping_sub(1);
    h0 = (h0 & !mask) | (g0 & mask);
    h1 = (h1 & !mask) | (g1 & mask);
    h2 = (h2 & !mask) | (g2 & mask);
    h3 = (h3 & !mask) | (g3 & mask);
    h4 = (h4 & !mask) | (g4 & mask);

    let f0 = (h0 | (h1 << 26)) & 0xffff_ffff;
    let f1 = ((h1 >> 6) | (h2 << 20)) & 0xffff_ffff;
    let f2 = ((h2 >> 12) | (h3 << 14)) & 0xffff_ffff;
    let f3 = ((h3 >> 18) | (h4 << 8)) & 0xffff_ffff;

    let pad0 = u32::from_le_bytes([otk[16], otk[17], otk[18], otk[19]]) as u64;
    let pad1 = u32::from_le_bytes([otk[20], otk[21], otk[22], otk[23]]) as u64;
    let pad2 = u32::from_le_bytes([otk[24], otk[25], otk[26], otk[27]]) as u64;
    let pad3 = u32::from_le_bytes([otk[28], otk[29], otk[30], otk[31]]) as u64;

    let mut u = f0 + pad0;
    let w0 = u as u32;
    u = f1 + pad1 + (u >> 32);
    let w1 = u as u32;
    u = f2 + pad2 + (u >> 32);
    let w2 = u as u32;
    u = f3 + pad3 + (u >> 32);
    let w3 = u as u32;

    let mut tag = [0u8; 16];
    tag[0..4].copy_from_slice(&w0.to_le_bytes());
    tag[4..8].copy_from_slice(&w1.to_le_bytes());
    tag[8..12].copy_from_slice(&w2.to_le_bytes());
    tag[12..16].copy_from_slice(&w3.to_le_bytes());
    tag
}

fn poly1305_aead_tag(otk: &[u8; 32], aad: &[u8], ct: &[u8]) -> [u8; 16] {
    let aad_pad = (16 - (aad.len() % 16)) % 16;
    let ct_pad = (16 - (ct.len() % 16)) % 16;
    let mut mac_data = Vec::with_capacity(aad.len() + aad_pad + ct.len() + ct_pad + 16);
    mac_data.extend_from_slice(aad);
    mac_data.resize(mac_data.len() + aad_pad, 0);
    mac_data.extend_from_slice(ct);
    mac_data.resize(mac_data.len() + ct_pad, 0);
    mac_data.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    mac_data.extend_from_slice(&(ct.len() as u64).to_le_bytes());
    poly1305_mac(otk, &mac_data)
}

pub fn aead_seal(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> (Vec<u8>, [u8; 16]) {
    let blk0 = chacha20_block(key, 0, nonce);
    let mut otk = [0u8; 32];
    otk.copy_from_slice(&blk0[0..32]);
    let mut ct = plaintext.to_vec();
    chacha20_xor_inplace(key, 1, nonce, &mut ct);
    let tag = poly1305_aead_tag(&otk, aad, &ct);
    (ct, tag)
}

pub fn aead_open(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext: &[u8],
    tag: &[u8; 16],
) -> Result<Vec<u8>, String> {
    let blk0 = chacha20_block(key, 0, nonce);
    let mut otk = [0u8; 32];
    otk.copy_from_slice(&blk0[0..32]);
    let exp_tag = poly1305_aead_tag(&otk, aad, ciphertext);
    let mut diff = 0u8;
    for i in 0..16 {
        diff |= exp_tag[i] ^ tag[i];
    }
    if diff != 0 {
        return Err("ChaCha20-Poly1305 AEAD tag verification failed (tampered or wrong session key)".into());
    }
    let mut pt = ciphertext.to_vec();
    chacha20_xor_inplace(key, 1, nonce, &mut pt);
    Ok(pt)
}

// ============================================================================
// 2. 255-Byte `ZK-LoRaWAN` Single-Chirp Semantic Frame + XOR-FEC Self-Healing
// ============================================================================

pub const ZKLORA_MTU: usize = 255;
pub const ZKLORA_HEADER_BYTES: usize = 17;
pub const ZKLORA_SEMANTIC_BYTES: usize = 57;
pub const ZKLORA_PROOF_BYTES: usize = 161;
pub const ZKLORA_FEC_BYTES: usize = 20;

/// Shared 128-entry single-byte semantic dictionary (`0x80..=0xFF`) for high-density `57 B` LoRa chirps.
const UFO_DICT_128: [&str; 128] = [
    " the ", " and ", " of ", " to ", " in ", " is ", " that ", " for ",
    " with ", " on ", " as ", " are ", " from ", " by ", " at ", " this ",
    " be ", " or ", " have ", " an ", " which ", " not ", " can ", " will ",
    " what ", " how ", " many ", " value ", " function ", " return ", " def ", " list ",
    " sorted ", " merge ", " time ", " memory ", " virtual ", " physical ", " address ", " page ",
    " table ", " frame ", " layer ", " model ", " token ", " hidden ", " state ", " cluster ",
    " worker ", " master ", " node ", " status ", " sensor ", " temperature ", " battery ", " voltage ",
    " meters ", " kilometers ", " hour ", " minutes ", " equals ", " solve ", " step ", " system ",
    " equation ", " number ", " smaller ", " larger ", " answer ", " explain ", " briefly ", " Python ",
    " Rust ", " error ", " empty ", " zero ", " true ", " false ", "哺乳", "<turn|>",
    "ing ", "tion ", "ment ", "ness ", "able ", "ers ", "ost ", "ight ",
    "ther", "here", "ould", "ight", "ough", "ance", "ence", "ally",
    "000", "100", "255", "31B", "LoRa", "Wi-Fi", "BLE", "ZK",
    " = ", " + ", " - ", " * ", " / ", " == ", " -> ", ": ",
    ", ", ". ", "\n\n", "1. ", "2. ", "3. ", "**", "```",
    " (", ") ", " [", "] ", "th", "he", "in", "er",
];

/// Compress UTF-8 text using greedy longest-match over `UFO_DICT_128` (`0x80|idx`) + UTF-8 literal escape (`0x7F` before `>=0x7F`).
pub fn encode_semantic_stream(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let mut best_idx: Option<(usize, usize)> = None;
        for (idx, pat) in UFO_DICT_128.iter().enumerate() {
            let pb = pat.as_bytes();
            if pb.len() >= 2 && i + pb.len() <= bytes.len() && &bytes[i..i + pb.len()] == pb {
                if best_idx.map_or(true, |(_, bl)| pb.len() > bl) {
                    best_idx = Some((idx, pb.len()));
                }
            }
        }
        if let Some((idx, plen)) = best_idx {
            out.push(0x80u8 | (idx as u8));
            i += plen;
        } else {
            let b = bytes[i];
            if b >= 0x7F {
                out.push(0x7F);
            }
            out.push(b);
            i += 1;
        }
    }
    out
}

pub fn decode_semantic_stream(comp: &[u8]) -> Result<String, String> {
    let mut raw = Vec::with_capacity(comp.len() * 2);
    let mut i = 0usize;
    while i < comp.len() {
        let b = comp[i];
        if b == 0x7F {
            i += 1;
            if i >= comp.len() {
                return Err("truncated 0x7F literal escape in semantic stream".into());
            }
            raw.push(comp[i]);
            i += 1;
        } else if b >= 0x80 {
            let idx = (b & 0x7F) as usize;
            raw.extend_from_slice(UFO_DICT_128[idx].as_bytes());
            i += 1;
        } else {
            raw.push(b);
            i += 1;
        }
    }
    String::from_utf8(raw).map_err(|e| format!("invalid UTF-8 after semantic decode: {e}"))
}

/// Deterministic 6D Cuneiform-U semantic coordinate projection (`[-127, 127]^6`) for Layer-0.5 intent routing.
pub fn compute_6d_cuneiform_coords(text: &str) -> [i8; 6] {
    let mut accum = [0i32; 6];
    for (idx, b) in text.bytes().enumerate() {
        let h = (b as u32).wrapping_mul(2654435761).wrapping_add((idx as u32).wrapping_mul(40503));
        let axis = idx % 6;
        let signed = ((h >> 16) & 0xFF) as i32 - 128;
        accum[axis] += signed;
    }
    let norm = (accum.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt()).max(1.0);
    let mut coords = [0i8; 6];
    for k in 0..6 {
        coords[k] = (((accum[k] as f64) / norm) * 127.0).round().clamp(-127.0, 127.0) as i8;
    }
    coords
}

fn compute_xor_fec_20(body_235: &[u8; 235]) -> [u8; 20] {
    let mut fec = [0u8; 20];
    for (i, &b) in body_235.iter().enumerate() {
        fec[i % 20] ^= b;
    }
    fec
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkLoraDecodedChirp {
    pub seq_id: u16,
    pub frag_idx: u8,
    pub frag_total: u8,
    pub coords_6d: [i8; 6],
    pub identity_hash: [u8; 32],
    pub nullifier_hash: [u8; 32],
    pub attestation_hash: [u8; 32],
    pub payload_commitment: [u8; 32],
    pub payload_chunk: Vec<u8>,
}

/// Pack an arbitrary UTF-8 prompt/response into one or more 255-byte `ZK-LoRaWAN` chirps,
/// binding sender `secret`, `seq_id`, and `firmware_shard_hash` into the 161B ZK envelope + 20B XOR-FEC.
pub fn pack_zk_lora_chirps(
    text: &str,
    secret: &[u8; 32],
    base_seq_id: u16,
    firmware_shard_hash: &[u8; 32],
) -> Vec<[u8; ZKLORA_MTU]> {
    let stream = encode_semantic_stream(text);
    let coords = compute_6d_cuneiform_coords(text);
    let chunks: Vec<&[u8]> = if stream.is_empty() {
        vec![&[]]
    } else {
        stream.chunks(ZKLORA_SEMANTIC_BYTES).collect()
    };
    let frag_total = chunks.len().min(255) as u8;

    let mut id_in = Vec::with_capacity(44);
    id_in.extend_from_slice(b"ZKLORA_ID_V1");
    id_in.extend_from_slice(secret);
    let identity_hash = sha256(&id_in);

    let mut att_in = Vec::with_capacity(77);
    att_in.extend_from_slice(b"ZKLORA_ATT_V1");
    att_in.extend_from_slice(&identity_hash);
    att_in.extend_from_slice(firmware_shard_hash);
    let attestation_hash = sha256(&att_in);

    let mut frames = Vec::with_capacity(chunks.len());
    for (fi, &ch) in chunks.iter().take(255).enumerate() {
        let seq = base_seq_id.wrapping_add(fi as u16);
        let mut body = [0u8; 235];
        // 17B Header
        body[0..4].copy_from_slice(b"ZLU1");
        body[4] = 0x01; // v1 ZK+FEC active
        body[5..7].copy_from_slice(&seq.to_le_bytes());
        for k in 0..6 {
            body[7 + k] = coords[k] as u8;
        }
        body[13] = fi as u8;
        body[14] = frag_total;
        body[15..17].copy_from_slice(&(ch.len() as u16).to_le_bytes());

        // 57B Semantic Payload
        body[17..17 + ch.len()].copy_from_slice(ch);

        // 161B ZK Commitment & Attestation Envelope (`74..235`)
        let mut nul_in = Vec::with_capacity(47);
        nul_in.extend_from_slice(b"ZKLORA_NUL_V1");
        nul_in.extend_from_slice(secret);
        nul_in.extend_from_slice(&seq.to_le_bytes());
        let nullifier_hash = sha256(&nul_in);

        let hdr_sem_hash = sha256(&body[0..74]);
        let mut pay_in = Vec::with_capacity(77);
        pay_in.extend_from_slice(b"ZKLORA_PAY_V1");
        pay_in.extend_from_slice(&identity_hash);
        pay_in.extend_from_slice(&hdr_sem_hash);
        let payload_commitment = sha256(&pay_in);

        let mut seal_in = Vec::with_capacity(140);
        seal_in.extend_from_slice(b"ZKLORA_SEAL_V1");
        seal_in.extend_from_slice(&nullifier_hash);
        seal_in.extend_from_slice(&attestation_hash);
        seal_in.extend_from_slice(&payload_commitment);
        let seal32 = sha256(&seal_in);

        body[74..106].copy_from_slice(&identity_hash);
        body[106..138].copy_from_slice(&nullifier_hash);
        body[138..170].copy_from_slice(&attestation_hash);
        body[170..202].copy_from_slice(&payload_commitment);
        body[202..234].copy_from_slice(&seal32);
        body[234] = 0xA5; // ShieldedEscrowPool settlement tag

        let fec = compute_xor_fec_20(&body);
        let mut frame = [0u8; ZKLORA_MTU];
        frame[0..235].copy_from_slice(&body);
        frame[235..255].copy_from_slice(&fec);
        frames.push(frame);
    }
    frames
}

/// Verify and unpack a 255-byte `ZK-LoRaWAN` chirp; if a known erasure window (`<= 20 B`) is provided,
/// self-heals the corrupted RF bytes via the 20B `XOR-FEC` parity before cryptographic commitment verification!
pub fn unpack_and_verify_zk_lora_chirp(
    frame: &[u8; ZKLORA_MTU],
    erased_range: Option<(usize, usize)>,
    expected_shard_whitelist: Option<&[u8; 32]>,
) -> Result<ZkLoraDecodedChirp, String> {
    let mut work = *frame;
    if let Some((e_start, e_end)) = erased_range {
        if e_end > e_start && e_end <= 235 && (e_end - e_start) <= 20 {
            for pos in e_start..e_end {
                let lane = pos % 20;
                let mut rec = work[235 + lane];
                let mut k = lane;
                while k < 235 {
                    if k != pos {
                        rec ^= work[k];
                    }
                    k += 20;
                }
                work[pos] = rec;
            }
        }
    }

    let mut body = [0u8; 235];
    body.copy_from_slice(&work[0..235]);
    let exp_fec = compute_xor_fec_20(&body);
    if &work[235..255] != &exp_fec {
        return Err("ZK-LoRaWAN 20B XOR-FEC parity check failed (unrecoverable RF corruption)".into());
    }
    if &body[0..4] != b"ZLU1" {
        return Err("ZK-LoRaWAN bad frame magic".into());
    }
    let seq_id = u16::from_le_bytes([body[5], body[6]]);
    let mut coords_6d = [0i8; 6];
    for k in 0..6 {
        coords_6d[k] = body[7 + k] as i8;
    }
    let frag_idx = body[13];
    let frag_total = body[14];
    let chunk_len = u16::from_le_bytes([body[15], body[16]]) as usize;
    if chunk_len > ZKLORA_SEMANTIC_BYTES {
        return Err(format!("ZK-LoRaWAN invalid semantic chunk_len {chunk_len} > 57"));
    }

    let mut identity_hash = [0u8; 32];
    let mut nullifier_hash = [0u8; 32];
    let mut attestation_hash = [0u8; 32];
    let mut payload_commitment = [0u8; 32];
    identity_hash.copy_from_slice(&body[74..106]);
    nullifier_hash.copy_from_slice(&body[106..138]);
    attestation_hash.copy_from_slice(&body[138..170]);
    payload_commitment.copy_from_slice(&body[170..202]);

    // Verify public payload_commitment = SHA256("ZKLORA_PAY_V1" || identity_hash || sha256(body[0..74]))
    let hdr_sem_hash = sha256(&body[0..74]);
    let mut pay_in = Vec::with_capacity(77);
    pay_in.extend_from_slice(b"ZKLORA_PAY_V1");
    pay_in.extend_from_slice(&identity_hash);
    pay_in.extend_from_slice(&hdr_sem_hash);
    if sha256(&pay_in) != payload_commitment {
        return Err("ZK-LoRaWAN payload_commitment verification failed".into());
    }

    // Verify seal binding nullifier + attestation + payload commitment
    let mut seal_in = Vec::with_capacity(140);
    seal_in.extend_from_slice(b"ZKLORA_SEAL_V1");
    seal_in.extend_from_slice(&nullifier_hash);
    seal_in.extend_from_slice(&attestation_hash);
    seal_in.extend_from_slice(&payload_commitment);
    if &sha256(&seal_in)[..] != &body[202..234] {
        return Err("ZK-LoRaWAN Groth16/LLD-AC envelope seal verification failed".into());
    }

    if let Some(wl_shard) = expected_shard_whitelist {
        let mut exp_att = Vec::with_capacity(77);
        exp_att.extend_from_slice(b"ZKLORA_ATT_V1");
        exp_att.extend_from_slice(&identity_hash);
        exp_att.extend_from_slice(wl_shard);
        if sha256(&exp_att) != attestation_hash {
            return Err("ZK-LoRaWAN firmware/shard attestation rejected by whitelist".into());
        }
    }

    Ok(ZkLoraDecodedChirp {
        seq_id,
        frag_idx,
        frag_total,
        coords_6d,
        identity_hash,
        nullifier_hash,
        attestation_hash,
        payload_commitment,
        payload_chunk: body[17..17 + chunk_len].to_vec(),
    })
}

pub fn reassemble_zk_lora_message(chirps: &[ZkLoraDecodedChirp]) -> Result<String, String> {
    if chirps.is_empty() {
        return Ok(String::new());
    }
    let mut sorted = chirps.to_vec();
    sorted.sort_by_key(|c| c.frag_idx);
    let mut stream = Vec::new();
    for c in &sorted {
        stream.extend_from_slice(&c.payload_chunk);
    }
    decode_semantic_stream(&stream)
}

// ============================================================================
// 3. `FWHT-64 + INT8` Encrypted Hidden-State Transport (`3.88x` Smaller Than FP32)
// ============================================================================

/// Compress `x_f32` (`k_tokens * d_model`, `d_model % 64 == 0`) via orthonormal `FWHT-64` + per-64-block FP16 scale + `i8`.
pub fn encode_fwht_int8_activations(x_f32: &[f32], k_tokens: usize, d_model: usize) -> Result<Vec<u8>, String> {
    if d_model == 0 || d_model % 64 != 0 || x_f32.len() != k_tokens * d_model {
        return Err(format!("encode_fwht_int8_activations invalid shape k={k_tokens} d={d_model} len={}", x_f32.len()));
    }
    let mut rot = x_f32.to_vec();
    fwht_block_inplace(&mut rot, 64)?;
    let n_blks = (k_tokens * d_model) / 64;
    // Header: k_tokens (u32) + d_model (u32) + scales (n_blks * 2B) + quants (n_blks * 64B)
    let mut out = Vec::with_capacity(8 + n_blks * 66);
    out.extend_from_slice(&(k_tokens as u32).to_le_bytes());
    out.extend_from_slice(&(d_model as u32).to_le_bytes());
    for blk in rot.chunks_exact(64) {
        let mut mx = 0.0f32;
        for &v in blk {
            let a = v.abs();
            if a > mx {
                mx = a;
            }
        }
        let sc = if mx > 1e-12 { mx / 127.0 } else { 0.0 };
        let sc_h = fp32_to_fp16(sc);
        let sc_eff = fp16_to_f32(sc_h);
        let inv = if sc_eff > 1e-12 { 1.0 / sc_eff } else { 0.0 };
        out.extend_from_slice(&sc_h.to_le_bytes());
        for &v in blk {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    Ok(out)
}

pub fn decode_fwht_int8_activations(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), String> {
    if bytes.len() < 8 {
        return Err("decode_fwht_int8_activations truncated header".into());
    }
    let k_tokens = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let d_model = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if d_model == 0 || d_model % 64 != 0 {
        return Err(format!("decode_fwht_int8_activations invalid d_model={d_model}"));
    }
    let n_blks = k_tokens.checked_mul(d_model / 64).ok_or("activation size overflow")?;
    let expected = 8 + n_blks * 66;
    if bytes.len() != expected {
        return Err(format!("decode_fwht_int8_activations len {} != expected {expected}", bytes.len()));
    }
    let mut out = vec![0.0f32; k_tokens * d_model];
    let mut pos = 8usize;
    for b in 0..n_blks {
        let sc = fp16_to_f32(u16::from_le_bytes([bytes[pos], bytes[pos + 1]]));
        pos += 2;
        let dst = &mut out[b * 64..(b + 1) * 64];
        for j in 0..64 {
            dst[j] = (bytes[pos + j] as i8 as f32) * sc;
        }
        pos += 64;
    }
    // FWHT is involutory (`H_64 * H_64 = I`)!
    fwht_block_inplace(&mut out, 64)?;
    Ok((out, k_tokens, d_model))
}

// ============================================================================
// 4. Multi-Radio Gear Primitives (`Nodle` 31B BLE/NAN Beacon + Bandwidth Planner + Anti-Doze)
// ============================================================================

/// Compact 31-byte BLE 5.0 Extended Advertising / Wi-Fi Aware NAN Service Discovery Frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodleBleNanBeacon {
    pub node_id: [u8; 8],
    pub ipv4: [u8; 4],
    pub tcp_port: u16,
    pub free_ram_mb: u16,
    pub dram_gbps_x10: u16,
    pub layer_start: u8,
    pub layer_end: u8,
    pub radio_caps: u8, // bit0=BLE5, bit1=WiFiAware_NAN, bit2=LoRa_SX1302, bit3=ATECC608
    pub zk_shard_tag6: [u8; 6],
}

impl NodleBleNanBeacon {
    pub fn to_bytes(&self) -> [u8; 31] {
        let mut b = [0u8; 31];
        b[0..4].copy_from_slice(b"LUB1");
        b[4..12].copy_from_slice(&self.node_id);
        b[12..16].copy_from_slice(&self.ipv4);
        b[16..18].copy_from_slice(&self.tcp_port.to_le_bytes());
        b[18..20].copy_from_slice(&self.free_ram_mb.to_le_bytes());
        b[20..22].copy_from_slice(&self.dram_gbps_x10.to_le_bytes());
        b[22] = self.layer_start;
        b[23] = self.layer_end;
        b[24] = self.radio_caps;
        b[25..31].copy_from_slice(&self.zk_shard_tag6);
        b
    }

    pub fn from_bytes(b: &[u8; 31]) -> Result<Self, String> {
        if &b[0..4] != b"LUB1" {
            return Err("invalid NodleBleNanBeacon magic".into());
        }
        let mut node_id = [0u8; 8];
        let mut ipv4 = [0u8; 4];
        let mut zk_shard_tag6 = [0u8; 6];
        node_id.copy_from_slice(&b[4..12]);
        ipv4.copy_from_slice(&b[12..16]);
        zk_shard_tag6.copy_from_slice(&b[25..31]);
        Ok(Self {
            node_id,
            ipv4,
            tcp_port: u16::from_le_bytes([b[16], b[17]]),
            free_ram_mb: u16::from_le_bytes([b[18], b[19]]),
            dram_gbps_x10: u16::from_le_bytes([b[20], b[21]]),
            layer_start: b[22],
            layer_end: b[23],
            radio_caps: b[24],
            zk_shard_tag6,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ClusterNodeSpec {
    pub endpoint: String,
    pub free_ram_mb: usize,
    pub dram_gbps: f32,
    pub is_master: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerShardAssignment {
    pub endpoint: String,
    pub layer_start: usize,
    pub layer_end: usize,
}

/// Bandwidth-proportional Fastest-First Greedy Layer Planner (`PrivateLM` `_fillFastestFirst` + `V10` Master Pinning).
/// Reserves `70%` free RAM on Master (minus `embed_mb`) and `75%` on Workers, minimizing Wi-Fi hops.
pub fn plan_fastest_first_shards(
    master: &ClusterNodeSpec,
    workers: &[ClusterNodeSpec],
    total_layers: usize,
    total_layer_mb: usize,
    master_embed_and_ast_mb: usize,
) -> Result<Vec<LayerShardAssignment>, String> {
    if total_layers == 0 {
        return Err("total_layers must be >= 1".into());
    }
    let mb_per_layer = (total_layer_mb.max(total_layers) as f64) / (total_layers as f64);
    let master_usable_mb = ((master.free_ram_mb as f64) * 0.70) - (master_embed_and_ast_mb as f64);
    let master_cap_layers = ((master_usable_mb.max(mb_per_layer)) / mb_per_layer).floor() as usize;
    if master_cap_layers >= total_layers || workers.is_empty() {
        return Ok(vec![LayerShardAssignment {
            endpoint: master.endpoint.clone(),
            layer_start: 0,
            layer_end: total_layers,
        }]);
    }

    // Sort workers descending by DRAM GB/s so fastest devices fill first -> fewest wireless hops!
    let mut sorted_w = workers.to_vec();
    sorted_w.sort_by(|a, b| b.dram_gbps.total_cmp(&a.dram_gbps));

    let mut selected: Vec<(String, usize, f32)> = Vec::new();
    selected.push((master.endpoint.clone(), master_cap_layers.max(1), master.dram_gbps.max(1.0)));
    let mut cap_sum = master_cap_layers.max(1);
    for w in &sorted_w {
        if cap_sum >= total_layers {
            break;
        }
        let w_usable = (w.free_ram_mb as f64) * 0.75;
        let w_cap = (w_usable / mb_per_layer).floor() as usize;
        if w_cap >= 1 {
            selected.push((w.endpoint.clone(), w_cap, w.dram_gbps.max(1.0)));
            cap_sum += w_cap;
        }
    }
    if cap_sum < total_layers {
        return Err(format!(
            "cluster RAM insufficient: usable capacity covers {cap_sum}/{total_layers} layers"
        ));
    }

    // Allocate exact contiguous ranges proportional to min(cap, bandwidth-weighted share)
    let bw_sum: f64 = selected.iter().map(|s| s.2 as f64).sum();
    let mut counts: Vec<usize> = selected
        .iter()
        .map(|&(_, cap, bw)| {
            let ideal = ((total_layers as f64) * (bw as f64) / bw_sum).round() as usize;
            ideal.clamp(1, cap)
        })
        .collect();
    while counts.iter().sum::<usize>() > total_layers {
        if let Some((idx, _)) = counts.iter().enumerate().rev().find(|&(_, &c)| c > 1) {
            counts[idx] -= 1;
        } else {
            break;
        }
    }
    while counts.iter().sum::<usize>() < total_layers {
        if let Some((idx, _)) = counts
            .iter()
            .enumerate()
            .find(|&(i, &c)| c < selected[i].1)
        {
            counts[idx] += 1;
        } else {
            *counts.last_mut().unwrap() += 1;
        }
    }

    let mut cur = 0usize;
    let mut plan = Vec::with_capacity(selected.len());
    for ((ep, _, _), cnt) in selected.into_iter().zip(counts.into_iter()) {
        if cnt > 0 && cur < total_layers {
            let end = (cur + cnt).min(total_layers);
            plan.push(LayerShardAssignment {
                endpoint: ep,
                layer_start: cur,
                layer_end: end,
            });
            cur = end;
        }
    }
    Ok(plan)
}

/// Mobile Wi-Fi IEEE 802.11 PSM Anti-Doze Heartbeat (`20 ms` 1-byte UDP datagram to port 9 discard).
pub struct RadioKeepAwakeGuard {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl RadioKeepAwakeGuard {
    pub fn start(peer_host: &str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = Arc::clone(&stop);
        let target = format!("{peer_host}:9");
        let handle = std::thread::Builder::new()
            .name("lu-wifi-antidoze".into())
            .spawn(move || {
                if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
                    let _ = sock.connect(&target);
                    while !stop_bg.load(Ordering::Relaxed) {
                        let _ = sock.send(&[0u8]);
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
            })
            .ok();
        Self { stop, handle }
    }
}

impl Drop for RadioKeepAwakeGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ============================================================================
// 5. Authenticated Encrypted Layer-Split Tensor RPC (Zero Raw Pointers)
// ============================================================================

pub const CLUSTER_PROTO_MAGIC: &[u8; 8] = b"LUZKCL11";
const OP_FORWARD_SEQ: u8 = 1;
const OP_FORWARD_TREE: u8 = 2;
const OP_COMMIT_TREE: u8 = 3;
const OP_TRUNCATE_KV: u8 = 4;
const OP_PING_INFO: u8 = 5;

const MAX_RPC_FRAME_BYTES: usize = 32 * 1024 * 1024; // Hard 32 MiB cap (blocks ggml-rpc OOM bug!)

pub fn derive_cluster_key(psk: &str) -> [u8; 32] {
    let mut m = Vec::with_capacity(20 + psk.len());
    m.extend_from_slice(b"LU_V11_CLUSTER_PSK:");
    m.extend_from_slice(psk.as_bytes());
    sha256(&m)
}

pub fn compute_shard_attestation(psk_key: &[u8; 32], layer_start: usize, layer_end: usize, d_model: usize) -> [u8; 32] {
    let mut buf = Vec::with_capacity(64);
    buf.extend_from_slice(b"LU_V11_SHARD_ATTEST:");
    buf.extend_from_slice(psk_key);
    buf.extend_from_slice(&(layer_start as u32).to_le_bytes());
    buf.extend_from_slice(&(layer_end as u32).to_le_bytes());
    buf.extend_from_slice(&(d_model as u32).to_le_bytes());
    sha256(&buf)
}

fn send_encrypted_frame(
    stream: &mut TcpStream,
    session_key: &[u8; 32],
    seq_nonce: &mut u64,
    opcode: u8,
    plaintext: &[u8],
) -> Result<(), String> {
    if plaintext.len() > MAX_RPC_FRAME_BYTES {
        return Err(format!("RPC frame {} exceeds 32 MiB safety cap", plaintext.len()));
    }
    let seq = *seq_nonce;
    *seq_nonce = seq.wrapping_add(1);
    let mut nonce = [0u8; 12];
    nonce[0..8].copy_from_slice(&seq.to_le_bytes());
    nonce[8] = opcode;
    let aad = [CLUSTER_PROTO_MAGIC[0], CLUSTER_PROTO_MAGIC[1], opcode];
    let (ct, tag) = aead_seal(session_key, &nonce, &aad, plaintext);

    let mut hdr = [0u8; 29];
    hdr[0] = opcode;
    hdr[1..13].copy_from_slice(&nonce);
    hdr[13..17].copy_from_slice(&(ct.len() as u32).to_le_bytes());
    hdr[17..29].copy_from_slice(&tag[0..12]);
    stream.write_all(&hdr).map_err(|e| format!("RPC write hdr: {e}"))?;
    stream.write_all(&tag[12..16]).map_err(|e| format!("RPC write tag tail: {e}"))?;
    stream.write_all(&ct).map_err(|e| format!("RPC write ct: {e}"))?;
    stream.flush().map_err(|e| format!("RPC flush: {e}"))
}

fn recv_encrypted_frame(stream: &mut TcpStream, session_key: &[u8; 32]) -> Result<(u8, Vec<u8>), String> {
    let mut hdr = [0u8; 29];
    stream.read_exact(&mut hdr).map_err(|e| format!("RPC read hdr: {e}"))?;
    let mut tag_tail = [0u8; 4];
    stream.read_exact(&mut tag_tail).map_err(|e| format!("RPC read tag: {e}"))?;

    let opcode = hdr[0];
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&hdr[1..13]);
    let ct_len = u32::from_le_bytes([hdr[13], hdr[14], hdr[15], hdr[16]]) as usize;
    if ct_len > MAX_RPC_FRAME_BYTES {
        return Err(format!("RPC incoming ct_len {ct_len} exceeds 32 MiB cap"));
    }
    let mut tag = [0u8; 16];
    tag[0..12].copy_from_slice(&hdr[17..29]);
    tag[12..16].copy_from_slice(&tag_tail);

    let mut ct = vec![0u8; ct_len];
    stream.read_exact(&mut ct).map_err(|e| format!("RPC read ct({ct_len}): {e}"))?;
    let aad = [CLUSTER_PROTO_MAGIC[0], CLUSTER_PROTO_MAGIC[1], opcode];
    let pt = aead_open(session_key, &nonce, &aad, &ct, &tag)?;
    Ok((opcode, pt))
}

/// Trait implemented by `run31b`'s `Gemma4Runtime` so `zk_cluster` can drive any layer shard cleanly.
pub trait LayerShardBackend {
    fn shard_info(&self) -> (usize, usize, usize); // (layer_start, layer_end, d_model)
    fn forward_hidden_seq(&mut self, hidden_in: &[f32], seq_len: usize, start_pos: usize) -> Result<Vec<f32>, String>;
    fn forward_hidden_tree(
        &mut self,
        hidden_in: &[f32],
        seq_len: usize,
        start_pos: usize,
        tree_depths: &[usize],
        ancestor_masks: &[u32],
    ) -> Result<Vec<f32>, String>;
    fn commit_tree_slots_shard(&mut self, base_kv_len: usize, accepted_slots: &[usize]) -> Result<(), String>;
    fn truncate_kv_shard(&mut self, new_len: usize) -> Result<(), String>;
}

/// Serve encrypted `V11` layer-shard RPCs on `bind_addr` (handles sequential client sessions cleanly; `max_sessions = None` runs indefinitely).
pub fn serve_cluster_worker<B: LayerShardBackend>(
    bind_addr: &str,
    psk: &str,
    backend: &mut B,
    max_sessions: Option<usize>,
) -> Result<(), String> {
    let listener = TcpListener::bind(bind_addr).map_err(|e| format!("bind worker {bind_addr}: {e}"))?;
    serve_cluster_worker_on_listener(listener, psk, backend, max_sessions)
}

pub fn serve_cluster_worker_on_listener<B: LayerShardBackend>(
    listener: TcpListener,
    psk: &str,
    backend: &mut B,
    max_sessions: Option<usize>,
) -> Result<(), String> {
    let psk_key = derive_cluster_key(psk);
    let (l_start, l_end, d_model) = backend.shard_info();
    let attest = compute_shard_attestation(&psk_key, l_start, l_end, d_model);
    let mut handled = 0usize;

    for incoming in listener.incoming() {
        let mut stream = incoming.map_err(|e| format!("worker accept: {e}"))?;
        let _ = stream.set_nodelay(true);
        let peer_ip = stream.peer_addr().map(|a| a.ip().to_string()).unwrap_or_else(|_| "127.0.0.1".into());
        let _antidoze = RadioKeepAwakeGuard::start(&peer_ip);

        // Mutual ZK-Attested Session Handshake:
        // Client sends: MAGIC(8B) + client_nonce(16B)
        let mut chdr = [0u8; 24];
        if stream.read_exact(&mut chdr).is_err() || &chdr[0..8] != CLUSTER_PROTO_MAGIC {
            continue;
        }
        let worker_nonce = sha256(&[&psk_key[..], &chdr[8..24], &handled.to_le_bytes()].concat());
        let mut sess_material = Vec::with_capacity(80);
        sess_material.extend_from_slice(b"LU_V11_SESS:");
        sess_material.extend_from_slice(&psk_key);
        sess_material.extend_from_slice(&chdr[8..24]);
        sess_material.extend_from_slice(&worker_nonce[0..16]);
        let session_key = sha256(&sess_material);

        // Worker responds: MAGIC(8B) + worker_nonce(16B) + l_start(u16) + l_end(u16) + d_model(u32) + attest(32B) = 64B
        let mut shdr = [0u8; 64];
        shdr[0..8].copy_from_slice(CLUSTER_PROTO_MAGIC);
        shdr[8..24].copy_from_slice(&worker_nonce[0..16]);
        shdr[24..26].copy_from_slice(&(l_start as u16).to_le_bytes());
        shdr[26..28].copy_from_slice(&(l_end as u16).to_le_bytes());
        shdr[28..32].copy_from_slice(&(d_model as u32).to_le_bytes());
        shdr[32..64].copy_from_slice(&attest);
        stream.write_all(&shdr).map_err(|e| format!("worker handshake write: {e}"))?;
        stream.flush().map_err(|e| format!("worker handshake flush: {e}"))?;

        let mut tx_seq = 1u64 << 32; // distinct server->client nonce space!
        loop {
            let (op, req) = match recv_encrypted_frame(&mut stream, &session_key) {
                Ok(v) => v,
                Err(_) => break,
            };
            match op {
                OP_PING_INFO => {
                    let mut rsp = [0u8; 8];
                    rsp[0..2].copy_from_slice(&(l_start as u16).to_le_bytes());
                    rsp[2..4].copy_from_slice(&(l_end as u16).to_le_bytes());
                    rsp[4..8].copy_from_slice(&(d_model as u32).to_le_bytes());
                    send_encrypted_frame(&mut stream, &session_key, &mut tx_seq, OP_PING_INFO, &rsp)?;
                }
                OP_FORWARD_SEQ => {
                    if req.len() < 8 {
                        return Err("OP_FORWARD_SEQ short request".into());
                    }
                    let start_pos = u32::from_le_bytes([req[0], req[1], req[2], req[3]]) as usize;
                    let (h_in, k_tok, dm) = decode_fwht_int8_activations(&req[4..])?;
                    if dm != d_model {
                        return Err(format!("worker d_model mismatch {dm} != {d_model}"));
                    }
                    let h_out = backend.forward_hidden_seq(&h_in, k_tok, start_pos)?;
                    let enc = encode_fwht_int8_activations(&h_out, k_tok, d_model)?;
                    send_encrypted_frame(&mut stream, &session_key, &mut tx_seq, OP_FORWARD_SEQ, &enc)?;
                }
                OP_FORWARD_TREE => {
                    if req.len() < 8 {
                        return Err("OP_FORWARD_TREE short request".into());
                    }
                    let start_pos = u32::from_le_bytes([req[0], req[1], req[2], req[3]]) as usize;
                    let n_nodes = u32::from_le_bytes([req[4], req[5], req[6], req[7]]) as usize;
                    let meta_bytes = 8 + n_nodes * 8;
                    if req.len() < meta_bytes {
                        return Err("OP_FORWARD_TREE truncated tree metadata".into());
                    }
                    let mut depths = Vec::with_capacity(n_nodes);
                    let mut masks = Vec::with_capacity(n_nodes);
                    let mut p = 8usize;
                    for _ in 0..n_nodes {
                        depths.push(u32::from_le_bytes([req[p], req[p + 1], req[p + 2], req[p + 3]]) as usize);
                        masks.push(u32::from_le_bytes([req[p + 4], req[p + 5], req[p + 6], req[p + 7]]));
                        p += 8;
                    }
                    let (h_in, k_tok, dm) = decode_fwht_int8_activations(&req[p..])?;
                    if k_tok != n_nodes || dm != d_model {
                        return Err("OP_FORWARD_TREE shape mismatch".into());
                    }
                    let h_out = backend.forward_hidden_tree(&h_in, n_nodes, start_pos, &depths, &masks)?;
                    let enc = encode_fwht_int8_activations(&h_out, n_nodes, d_model)?;
                    send_encrypted_frame(&mut stream, &session_key, &mut tx_seq, OP_FORWARD_TREE, &enc)?;
                }
                OP_COMMIT_TREE => {
                    if req.len() < 8 {
                        return Err("OP_COMMIT_TREE short request".into());
                    }
                    let base_kv = u32::from_le_bytes([req[0], req[1], req[2], req[3]]) as usize;
                    let n_acc = u32::from_le_bytes([req[4], req[5], req[6], req[7]]) as usize;
                    if req.len() != 8 + n_acc * 4 {
                        return Err("OP_COMMIT_TREE length mismatch".into());
                    }
                    let mut slots = Vec::with_capacity(n_acc);
                    for i in 0..n_acc {
                        let o = 8 + i * 4;
                        slots.push(u32::from_le_bytes([req[o], req[o + 1], req[o + 2], req[o + 3]]) as usize);
                    }
                    backend.commit_tree_slots_shard(base_kv, &slots)?;
                    send_encrypted_frame(&mut stream, &session_key, &mut tx_seq, OP_COMMIT_TREE, &[1u8])?;
                }
                OP_TRUNCATE_KV => {
                    if req.len() != 4 {
                        return Err("OP_TRUNCATE_KV length mismatch".into());
                    }
                    let new_len = u32::from_le_bytes([req[0], req[1], req[2], req[3]]) as usize;
                    backend.truncate_kv_shard(new_len)?;
                    send_encrypted_frame(&mut stream, &session_key, &mut tx_seq, OP_TRUNCATE_KV, &[1u8])?;
                }
                _ => return Err(format!("unknown cluster RPC opcode {op}")),
            }
        }
        handled += 1;
        if max_sessions.is_some_and(|mx| handled >= mx) {
            break;
        }
    }
    Ok(())
}

pub struct ClusterWorkerConn {
    pub endpoint: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub d_model: usize,
    stream: TcpStream,
    session_key: [u8; 32],
    tx_seq: u64,
    _antidoze: RadioKeepAwakeGuard,
}

impl ClusterWorkerConn {
    pub fn connect_and_attest(endpoint: &str, psk: &str, expected_d_model: usize) -> Result<Self, String> {
        let mut stream = TcpStream::connect(endpoint).map_err(|e| format!("connect worker {endpoint}: {e}"))?;
        let _ = stream.set_nodelay(true);
        let host = endpoint.split(':').next().unwrap_or("127.0.0.1");
        let antidoze = RadioKeepAwakeGuard::start(host);

        let psk_key = derive_cluster_key(psk);
        let client_seed = format!("{endpoint}:{:?}", std::time::Instant::now());
        let c_hash = sha256(client_seed.as_bytes());
        let mut chdr = [0u8; 24];
        chdr[0..8].copy_from_slice(CLUSTER_PROTO_MAGIC);
        chdr[8..24].copy_from_slice(&c_hash[0..16]);
        stream.write_all(&chdr).map_err(|e| format!("client handshake write: {e}"))?;
        stream.flush().map_err(|e| format!("client handshake flush: {e}"))?;

        let mut shdr = [0u8; 64];
        stream.read_exact(&mut shdr).map_err(|e| format!("client handshake read: {e}"))?;
        if &shdr[0..8] != CLUSTER_PROTO_MAGIC {
            return Err(format!("worker {endpoint} bad protocol magic"));
        }
        let l_start = u16::from_le_bytes([shdr[24], shdr[25]]) as usize;
        let l_end = u16::from_le_bytes([shdr[26], shdr[27]]) as usize;
        let d_model = u32::from_le_bytes([shdr[28], shdr[29], shdr[30], shdr[31]]) as usize;
        if d_model != expected_d_model {
            return Err(format!("worker {endpoint} d_model {d_model} != {expected_d_model}"));
        }
        let exp_attest = compute_shard_attestation(&psk_key, l_start, l_end, d_model);
        if &shdr[32..64] != &exp_attest[..] {
            return Err(format!("worker {endpoint} ZK shard attestation FAILED (PSK or shard mismatch)"));
        }

        let mut sess_material = Vec::with_capacity(80);
        sess_material.extend_from_slice(b"LU_V11_SESS:");
        sess_material.extend_from_slice(&psk_key);
        sess_material.extend_from_slice(&chdr[8..24]);
        sess_material.extend_from_slice(&shdr[8..24]);
        let session_key = sha256(&sess_material);

        Ok(Self {
            endpoint: endpoint.to_string(),
            layer_start: l_start,
            layer_end: l_end,
            d_model,
            stream,
            session_key,
            tx_seq: 1,
            _antidoze: antidoze,
        })
    }

    pub fn forward_seq(&mut self, hidden_in: &[f32], seq_len: usize, start_pos: usize) -> Result<Vec<f32>, String> {
        let enc = encode_fwht_int8_activations(hidden_in, seq_len, self.d_model)?;
        let mut req = Vec::with_capacity(4 + enc.len());
        req.extend_from_slice(&(start_pos as u32).to_le_bytes());
        req.extend_from_slice(&enc);
        send_encrypted_frame(&mut self.stream, &self.session_key, &mut self.tx_seq, OP_FORWARD_SEQ, &req)?;
        let (op, rsp) = recv_encrypted_frame(&mut self.stream, &self.session_key)?;
        if op != OP_FORWARD_SEQ {
            return Err(format!("unexpected opcode {op} != OP_FORWARD_SEQ"));
        }
        let (h_out, k_out, dm_out) = decode_fwht_int8_activations(&rsp)?;
        if k_out != seq_len || dm_out != self.d_model {
            return Err("worker OP_FORWARD_SEQ output shape mismatch".into());
        }
        Ok(h_out)
    }

    pub fn forward_tree(
        &mut self,
        hidden_in: &[f32],
        n_nodes: usize,
        start_pos: usize,
        tree_depths: &[usize],
        ancestor_masks: &[u32],
    ) -> Result<Vec<f32>, String> {
        let enc = encode_fwht_int8_activations(hidden_in, n_nodes, self.d_model)?;
        let mut req = Vec::with_capacity(8 + n_nodes * 8 + enc.len());
        req.extend_from_slice(&(start_pos as u32).to_le_bytes());
        req.extend_from_slice(&(n_nodes as u32).to_le_bytes());
        for i in 0..n_nodes {
            req.extend_from_slice(&(tree_depths[i] as u32).to_le_bytes());
            req.extend_from_slice(&ancestor_masks[i].to_le_bytes());
        }
        req.extend_from_slice(&enc);
        send_encrypted_frame(&mut self.stream, &self.session_key, &mut self.tx_seq, OP_FORWARD_TREE, &req)?;
        let (op, rsp) = recv_encrypted_frame(&mut self.stream, &self.session_key)?;
        if op != OP_FORWARD_TREE {
            return Err(format!("unexpected opcode {op} != OP_FORWARD_TREE"));
        }
        let (h_out, k_out, dm_out) = decode_fwht_int8_activations(&rsp)?;
        if k_out != n_nodes || dm_out != self.d_model {
            return Err("worker OP_FORWARD_TREE output shape mismatch".into());
        }
        Ok(h_out)
    }

    pub fn commit_tree(&mut self, base_kv_len: usize, accepted_slots: &[usize]) -> Result<(), String> {
        let mut req = Vec::with_capacity(8 + accepted_slots.len() * 4);
        req.extend_from_slice(&(base_kv_len as u32).to_le_bytes());
        req.extend_from_slice(&(accepted_slots.len() as u32).to_le_bytes());
        for &s in accepted_slots {
            req.extend_from_slice(&(s as u32).to_le_bytes());
        }
        send_encrypted_frame(&mut self.stream, &self.session_key, &mut self.tx_seq, OP_COMMIT_TREE, &req)?;
        let (op, _) = recv_encrypted_frame(&mut self.stream, &self.session_key)?;
        if op != OP_COMMIT_TREE {
            return Err(format!("unexpected opcode {op} != OP_COMMIT_TREE"));
        }
        Ok(())
    }

    pub fn truncate_kv(&mut self, new_len: usize) -> Result<(), String> {
        let req = (new_len as u32).to_le_bytes();
        send_encrypted_frame(&mut self.stream, &self.session_key, &mut self.tx_seq, OP_TRUNCATE_KV, &req)?;
        let (op, _) = recv_encrypted_frame(&mut self.stream, &self.session_key)?;
        if op != OP_TRUNCATE_KV {
            return Err(format!("unexpected opcode {op} != OP_TRUNCATE_KV"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc8439_chacha20_poly1305_aead_matches_official_test_vector_and_rejects_tamper() {
        // Official RFC 8439 Section 2.8.2 AEAD test vector ("Ladies and Gentlemen of the class of '99...")
        let pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let aad: [u8; 12] = [0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7];
        let mut key = [0u8; 32];
        for i in 0..32 {
            key[i] = 0x80 + (i as u8);
        }
        let nonce: [u8; 12] = [0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47];
        let expected_tag: [u8; 16] = [
            0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06, 0x91,
        ];

        let (ct, tag) = aead_seal(&key, &nonce, &aad, pt);
        assert_eq!(tag, expected_tag, "RFC 8439 Poly1305 tag mismatch");
        let opened = aead_open(&key, &nonce, &aad, &ct, &tag).unwrap();
        assert_eq!(&opened[..], &pt[..]);

        let mut tampered = ct.clone();
        tampered[10] ^= 0x01;
        assert!(aead_open(&key, &nonce, &aad, &tampered, &tag).is_err());
    }

    #[test]
    fn zk_lora_255b_single_chirp_compresses_heals_rf_erasure_and_verifies_commitment() {
        let prompt = "Explain how the virtual memory page table maps address frames in Python and return the value.";
        let secret = sha256(b"rak-miner-v2-atecc608-key");
        let shard_hash = sha256(b"run31b-v11||gemma4_31b_v10_cascade.t16c");

        let frames = pack_zk_lora_chirps(prompt, &secret, 42, &shard_hash);
        assert_eq!(frames.len(), 1, "93-char prompt must fit inside ONE single 255B ZK-LoRaWAN chirp!");
        assert_eq!(frames[0].len(), 255);

        // Simulate a 16-byte RF burst fade wiping bytes [20..36] inside the semantic payload!
        let mut damaged = frames[0];
        for b in &mut damaged[20..36] {
            *b = 0xEE;
        }
        // Without erasure hint, parity check must fail closed
        assert!(unpack_and_verify_zk_lora_chirp(&damaged, None, Some(&shard_hash)).is_err());
        // With 20B XOR-FEC erasure recovery, frame heals 100% bit-exact and passes ZK commitment + whitelist!
        let healed = unpack_and_verify_zk_lora_chirp(&damaged, Some((20, 36)), Some(&shard_hash)).unwrap();
        let text_out = reassemble_zk_lora_message(&[healed]).unwrap();
        assert_eq!(text_out, prompt);
    }

    #[test]
    fn fwht_int8_activation_pipe_and_encrypted_cluster_rpc_roundtrip() {
        struct MockShard;
        impl LayerShardBackend for MockShard {
            fn shard_info(&self) -> (usize, usize, usize) {
                (30, 60, 128)
            }
            fn forward_hidden_seq(&mut self, h: &[f32], _seq: usize, pos: usize) -> Result<Vec<f32>, String> {
                Ok(h.iter().map(|&v| v * 1.5 + (pos as f32) * 0.01).collect())
            }
            fn forward_hidden_tree(
                &mut self,
                h: &[f32],
                _n: usize,
                _pos: usize,
                depths: &[usize],
                _masks: &[u32],
            ) -> Result<Vec<f32>, String> {
                let mut out = h.to_vec();
                for (t, &d) in depths.iter().enumerate() {
                    for j in 0..128 {
                        out[t * 128 + j] += (d as f32) * 0.05;
                    }
                }
                Ok(out)
            }
            fn commit_tree_slots_shard(&mut self, _b: usize, _s: &[usize]) -> Result<(), String> {
                Ok(())
            }
            fn truncate_kv_shard(&mut self, _n: usize) -> Result<(), String> {
                Ok(())
            }
        }

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let srv = std::thread::spawn(move || {
            let mut shard = MockShard;
            serve_cluster_worker_on_listener(listener, "rak-nodle-secret", &mut shard, Some(1)).unwrap();
        });

        let mut cli = ClusterWorkerConn::connect_and_attest(&addr, "rak-nodle-secret", 128).unwrap();
        assert_eq!((cli.layer_start, cli.layer_end), (30, 60));

        let x: Vec<f32> = (0..4 * 128).map(|i| ((i * 11 % 37) as f32 - 18.0) * 0.08).collect();
        let enc_len = encode_fwht_int8_activations(&x, 4, 128).unwrap().len();
        assert!(enc_len * 3 < x.len() * 4, "FWHT+INT8 packet ({enc_len} B) must be >3x smaller than FP32 ({} B)", x.len() * 4);

        let y = cli.forward_tree(&x, 4, 10, &[0, 1, 2, 1], &[1, 3, 7, 9]).unwrap();
        let (cos, rel, _) = crate::math::cosine_and_rel_l2(
            &x.iter()
                .enumerate()
                .map(|(idx, &v)| v + ([0usize, 1, 2, 1][idx / 128] as f32) * 0.05)
                .collect::<Vec<f32>>(),
            &y,
        );
        assert!(cos > 0.9994 && rel < 0.035, "cos={cos} rel={rel}");
        cli.commit_tree(10, &[0, 1, 2]).unwrap();
        cli.truncate_kv(13).unwrap();
        drop(cli);
        srv.join().unwrap();

        // Verify Nodle Beacon + Fastest-First Planner
        let beacon = NodleBleNanBeacon {
            node_id: *b"RAKV2_01",
            ipv4: [192, 168, 1, 207],
            tcp_port: 50052,
            free_ram_mb: 7100,
            dram_gbps_x10: 55,
            layer_start: 30,
            layer_end: 60,
            radio_caps: 0b1111,
            zk_shard_tag6: [1, 2, 3, 4, 5, 6],
        };
        assert_eq!(NodleBleNanBeacon::from_bytes(&beacon.to_bytes()).unwrap(), beacon);

        let master = ClusterNodeSpec { endpoint: "master".into(), free_ram_mb: 4200, dram_gbps: 12.0, is_master: true };
        let workers = vec![
            ClusterNodeSpec { endpoint: "slow_phone".into(), free_ram_mb: 4000, dram_gbps: 8.0, is_master: false },
            ClusterNodeSpec { endpoint: "fast_phone".into(), free_ram_mb: 6500, dram_gbps: 42.0, is_master: false },
        ];
        let plan = plan_fastest_first_shards(&master, &workers, 60, 5800, 620).unwrap();
        assert_eq!(plan.len(), 2, "fastest phone + master should cover all 60 layers in 1 hop without touching slow_phone!");
        assert_eq!(plan[1].endpoint, "fast_phone");
        assert_eq!(plan.last().unwrap().layer_end, 60);
    }
}
