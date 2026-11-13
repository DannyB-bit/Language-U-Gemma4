//! Zero-dependency JSON parser for SafeTensors headers, model config, layout, and tokenizer files.

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        if let Json::Obj(kv) = self {
            kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(n) => Some(*n),
            Json::Num(n) if n.is_finite() && n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 => {
                Some(*n as i64)
            }
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Int(n) if *n >= 0 => Some(*n as u64),
            Json::Num(n) if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 && *n <= u64::MAX as f64 => {
                Some(*n as u64)
            }
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().and_then(|v| usize::try_from(v).ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(n) => Some(*n as f64),
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self { Some(a) } else { None }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(o) = self { Some(o) } else { None }
    }
}

const MAX_DEPTH: usize = 64;

pub fn parse(b: &[u8]) -> Result<Json, String> {
    let mut p = Parser { b, i: 0 };
    p.ws();
    let v = p.val(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing bytes at offset {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn val(&mut self, depth: usize) -> Result<Json, String> {
        if depth >= MAX_DEPTH {
            return Err("JSON nesting depth exceeds limit".into());
        }
        self.ws();
        match self.peek().ok_or("unexpected EOF")? {
            b'{' => self.obj(depth + 1),
            b'[' => self.arr(depth + 1),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.lit("true", Json::Bool(true)),
            b'f' => self.lit("false", Json::Bool(false)),
            b'n' => self.lit("null", Json::Null),
            _ => self.num(),
        }
    }

    fn lit(&mut self, s: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            Err(format!("bad literal at offset {}", self.i))
        }
    }

    fn num(&mut self) -> Result<Json, String> {
        let start = self.i;
        let mut is_float = false;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let digit_start = self.i;
        while self.i < self.b.len() && matches!(self.b[self.i], b'0'..=b'9') {
            self.i += 1;
        }
        if self.i == digit_start {
            return Err(format!("invalid number at offset {start}"));
        }
        if self.peek() == Some(b'.') {
            is_float = true;
            self.i += 1;
            let frac_start = self.i;
            while self.i < self.b.len() && matches!(self.b[self.i], b'0'..=b'9') {
                self.i += 1;
            }
            if self.i == frac_start {
                return Err(format!("invalid fraction at offset {start}"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            let exp_start = self.i;
            while self.i < self.b.len() && matches!(self.b[self.i], b'0'..=b'9') {
                self.i += 1;
            }
            if self.i == exp_start {
                return Err(format!("invalid exponent at offset {start}"));
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        if !is_float {
            if let Ok(iv) = s.parse::<i64>() {
                return Ok(Json::Int(iv));
            }
        }
        let fv = s.parse::<f64>().map_err(|e| format!("bad number '{s}': {e}"))?;
        if !fv.is_finite() {
            return Err(format!("non-finite JSON number '{s}'"));
        }
        Ok(Json::Num(fv))
    }

    fn string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err("expected '\"'".into());
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or("unterminated JSON string")?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek().ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let h1 = self.hex4()?;
                            let ch = if (0xD800..0xDC00).contains(&h1) {
                                if self.peek() == Some(b'\\') {
                                    self.i += 1;
                                    if self.peek() == Some(b'u') {
                                        self.i += 1;
                                        let h2 = self.hex4()?;
                                        if !(0xDC00..=0xDFFF).contains(&h2) {
                                            return Err("invalid low surrogate".into());
                                        }
                                        let cp = 0x10000 + ((h1 - 0xD800) << 10) + (h2 - 0xDC00);
                                        char::from_u32(cp).ok_or("bad surrogate codepoint")?
                                    } else {
                                        return Err("expected \\u for surrogate pair".into());
                                    }
                                } else {
                                    return Err("unpaired high surrogate".into());
                                }
                            } else {
                                char::from_u32(h1).ok_or("invalid unicode escape")?
                            };
                            out.push(ch);
                        }
                        _ => return Err(format!("unsupported escape \\{}", e as char)),
                    }
                }
                _ => {
                    if c < 0x20 {
                        return Err("unescaped control character in JSON string".into());
                    }
                    let start = self.i - 1;
                    let len = utf8_len(c)?;
                    if start + len > self.b.len() {
                        return Err("truncated UTF-8 sequence in JSON string".into());
                    }
                    self.i = start + len;
                    let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
                    out.push_str(s);
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("short \\u escape".into());
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|e| e.to_string())?;
        self.i += 4;
        u32::from_str_radix(s, 16).map_err(|e| e.to_string())
    }

    fn arr(&mut self, depth: usize) -> Result<Json, String> {
        self.i += 1;
        let mut v = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(v));
        }
        loop {
            v.push(self.val(depth)?);
            self.ws();
            match self.peek().ok_or("unterminated JSON array")? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                _ => return Err("expected ',' or ']' in array".into()),
            }
        }
    }

    fn obj(&mut self, depth: usize) -> Result<Json, String> {
        self.i += 1;
        let mut v = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(v));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err("expected ':' in JSON object".into());
            }
            self.i += 1;
            let val = self.val(depth)?;
            v.push((k, val));
            self.ws();
            match self.peek().ok_or("unterminated JSON object")? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(Json::Obj(v));
                }
                _ => return Err("expected ',' or '}' in object".into()),
            }
        }
    }
}

fn utf8_len(b: u8) -> Result<usize, String> {
    if b < 0x80 {
        Ok(1)
    } else if (b >> 5) == 0b110 {
        Ok(2)
    } else if (b >> 4) == 0b1110 {
        Ok(3)
    } else if (b >> 3) == 0b11110 {
        Ok(4)
    } else {
        Err(format!("invalid UTF-8 lead byte 0x{b:02x}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_integers_and_floats() {
        let j = parse(br#"{"big": 536870912, "neg": -1, "flt": 1.25e-2, "arr": [1, true, null]}"#).unwrap();
        assert_eq!(j.get("big").unwrap().as_u64(), Some(536_870_912));
        assert_eq!(j.get("neg").unwrap().as_i64(), Some(-1));
        assert_eq!(j.get("neg").unwrap().as_usize(), None);
        assert!((j.get("flt").unwrap().as_f64().unwrap() - 0.0125).abs() < 1e-12);
    }

    #[test]
    fn parses_unicode_and_rejects_deep_nesting() {
        let j = parse(br#"{"t": "a\u0041\ud83d\ude00"}"#).unwrap();
        assert_eq!(j.get("t").unwrap().as_str(), Some("aA\u{1F600}"));
        let deep = "[".repeat(80) + &"]".repeat(80);
        assert!(parse(deep.as_bytes()).is_err());
    }
}
