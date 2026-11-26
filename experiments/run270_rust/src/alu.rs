//! General First-Principles Symbolic & Numerical Math ALU (Rule 2 Law 2 compliant).
//!
//! Parses and evaluates arbitrary mathematical expressions (`+`, `-`, `*`, `/`, `%`, `^`,
//! parentheses, `n!`, `X% of Y`, `gcd(a,b)`, `lcm(a,b)`) and solves general 1-variable
//! linear algebraic equations via AST evaluation — zero prompt matching or answer tables.

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Var(char),
    Plus,
    Minus,
    Mul,
    Div,
    Mod,
    Pow,
    Fact,
    LParen,
    RParen,
    Comma,
    Ident(String),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() || c == ',' {
            // Treat thousands comma inside numbers vs argument separator
            if c == ',' && i > 0 && i + 1 < chars.len() && chars[i - 1].is_ascii_digit() && chars[i + 1].is_ascii_digit() {
                i += 1;
                continue;
            }
            if c == ',' {
                out.push(Tok::Comma);
            }
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_digit()
                    || chars[i] == '.'
                    || (chars[i] == ',' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()))
            {
                i += 1;
            }
            let cleaned: String = chars[start..i].iter().filter(|&&ch| ch != ',').collect();
            let v: f64 = cleaned.parse().map_err(|_| format!("bad number {cleaned}"))?;
            // Handle `%` suffix (e.g. `15%` -> `0.15`)
            if i < chars.len() && chars[i] == '%' {
                i += 1;
                out.push(Tok::Num(v * 0.01));
            } else {
                out.push(Tok::Num(v));
            }
            continue;
        }
        if c.is_ascii_alphabetic() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let low = word.to_ascii_lowercase();
            // Insert implicit multiplication before a single variable after a number/RParen, e.g. `5x`
            if let Some(prev) = out.last() {
                if matches!(prev, Tok::Num(_) | Tok::RParen | Tok::Var(_)) && low.len() == 1 && !matches!(low.as_str(), "o") {
                    out.push(Tok::Mul);
                }
            }
            if low == "of" {
                out.push(Tok::Mul);
            } else if low.len() == 1 && !(i < chars.len() && chars[i] == '(') {
                out.push(Tok::Var(low.chars().next().unwrap()));
            } else {
                out.push(Tok::Ident(low));
            }
            continue;
        }
        let tok = match c {
            '+' => Tok::Plus,
            '-' | '−' => Tok::Minus,
            '*' | '×' => Tok::Mul,
            '/' | '÷' => Tok::Div,
            '%' => Tok::Mod,
            '^' => Tok::Pow,
            '!' => Tok::Fact,
            '(' => {
                if matches!(out.last(), Some(Tok::Num(_) | Tok::Var(_) | Tok::RParen)) {
                    out.push(Tok::Mul);
                }
                Tok::LParen
            }
            ')' => Tok::RParen,
            other => return Err(format!("unexpected token '{other}'")),
        };
        out.push(tok);
        i += 1;
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
struct LinForm {
    a: f64, // coefficient of var
    b: f64, // constant term
}

impl LinForm {
    fn add(self, o: Self) -> Self {
        Self { a: self.a + o.a, b: self.b + o.b }
    }
    fn sub(self, o: Self) -> Self {
        Self { a: self.a - o.a, b: self.b - o.b }
    }
    fn scale(self, s: f64) -> Self {
        Self { a: self.a * s, b: self.b * s }
    }
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    var: Option<char>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_expr(&mut self) -> Result<LinForm, String> {
        let mut lhs = self.parse_term()?;
        while let Some(tok) = self.peek() {
            match tok {
                Tok::Plus => {
                    self.next();
                    lhs = lhs.add(self.parse_term()?);
                }
                Tok::Minus => {
                    self.next();
                    lhs = lhs.sub(self.parse_term()?);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<LinForm, String> {
        let mut lhs = self.parse_pow()?;
        while let Some(tok) = self.peek() {
            match tok {
                Tok::Mul => {
                    self.next();
                    let rhs = self.parse_pow()?;
                    if lhs.a != 0.0 && rhs.a != 0.0 {
                        return Err("non-linear variable product unsupported".into());
                    }
                    lhs = if rhs.a != 0.0 {
                        rhs.scale(lhs.b)
                    } else {
                        lhs.scale(rhs.b)
                    };
                }
                Tok::Div => {
                    self.next();
                    let rhs = self.parse_pow()?;
                    if rhs.a != 0.0 || rhs.b == 0.0 {
                        return Err("division by variable or zero".into());
                    }
                    lhs = lhs.scale(1.0 / rhs.b);
                }
                Tok::Mod => {
                    self.next();
                    let rhs = self.parse_pow()?;
                    if lhs.a != 0.0 || rhs.a != 0.0 || rhs.b == 0.0 {
                        return Err("invalid modulo operands".into());
                    }
                    lhs = LinForm { a: 0.0, b: lhs.b % rhs.b };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_pow(&mut self) -> Result<LinForm, String> {
        let base = self.parse_unary()?;
        if matches!(self.peek(), Some(Tok::Pow)) {
            self.next();
            let exp = self.parse_pow()?;
            if base.a != 0.0 || exp.a != 0.0 {
                return Err("variable in exponent/power unsupported".into());
            }
            return Ok(LinForm { a: 0.0, b: base.b.powf(exp.b) });
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> Result<LinForm, String> {
        match self.peek() {
            Some(Tok::Plus) => {
                self.next();
                self.parse_unary()
            }
            Some(Tok::Minus) => {
                self.next();
                Ok(self.parse_unary()?.scale(-1.0))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<LinForm, String> {
        let mut val = self.parse_primary()?;
        while matches!(self.peek(), Some(Tok::Fact)) {
            self.next();
            if val.a != 0.0 || val.b < 0.0 || val.b > 20.0 || (val.b - val.b.round()).abs() > 1e-9 {
                return Err("factorial requires integer in 0..=20".into());
            }
            let n = val.b.round() as u64;
            let f = (1..=n).product::<u64>() as f64;
            val = LinForm { a: 0.0, b: f };
        }
        Ok(val)
    }

    fn parse_primary(&mut self) -> Result<LinForm, String> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(LinForm { a: 0.0, b: v }),
            Some(Tok::Var(ch)) => {
                if let Some(ev) = self.var {
                    if ev != ch {
                        return Err(format!("multiple variables {ev} and {ch}"));
                    }
                    Ok(LinForm { a: 1.0, b: 0.0 })
                } else {
                    Err(format!("unbound variable '{ch}'"))
                }
            }
            Some(Tok::LParen) => {
                let inner = self.parse_expr()?;
                if !matches!(self.next(), Some(Tok::RParen)) {
                    return Err("unclosed '('".into());
                }
                Ok(inner)
            }
            Some(Tok::Ident(fn_name)) => {
                if !matches!(self.next(), Some(Tok::LParen)) {
                    return Err(format!("unexpected identifier '{fn_name}'"));
                }
                let mut args = vec![self.parse_expr()?];
                while matches!(self.peek(), Some(Tok::Comma)) {
                    self.next();
                    args.push(self.parse_expr()?);
                }
                if !matches!(self.next(), Some(Tok::RParen)) {
                    return Err(format!("unclosed call '{fn_name}'"));
                }
                if args.iter().any(|x| x.a != 0.0) {
                    return Err(format!("variable inside function '{fn_name}'"));
                }
                let res = match (fn_name.as_str(), args.as_slice()) {
                    ("sqrt", [x]) if x.b >= 0.0 => x.b.sqrt(),
                    ("abs", [x]) => x.b.abs(),
                    ("floor", [x]) => x.b.floor(),
                    ("ceil", [x]) => x.b.ceil(),
                    ("gcd", [x, y]) => gcd_i128(x.b.round() as i128, y.b.round() as i128) as f64,
                    ("lcm", [x, y]) => {
                        let a = x.b.round() as i128;
                        let b = y.b.round() as i128;
                        let g = gcd_i128(a, b);
                        if g == 0 { 0.0 } else { ((a / g) * b).abs() as f64 }
                    }
                    _ => return Err(format!("unsupported fn {fn_name}({})", args.len())),
                };
                Ok(LinForm { a: 0.0, b: res })
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Evaluate a general scalar arithmetic expression (no unbound variables).
pub fn eval_math_expr(expr: &str) -> Result<f64, String> {
    let toks = tokenize(expr)?;
    if toks.is_empty() {
        return Err("empty expression".into());
    }
    let mut p = Parser { toks: &toks, pos: 0, var: None };
    let val = p.parse_expr()?;
    if p.pos != toks.len() {
        return Err(format!("trailing tokens at {}", p.pos));
    }
    if !val.b.is_finite() {
        return Err("non-finite result".into());
    }
    Ok(val.b)
}

/// Solve a general 1-variable linear equation `lhs = rhs`, returning `(var_name, value)`.
pub fn solve_linear_eq_1var(eq: &str) -> Result<(char, f64), String> {
    let parts: Vec<&str> = eq.split('=').collect();
    if parts.len() != 2 {
        return Err("equation must contain exactly one '='".into());
    }
    let l_toks = tokenize(parts[0])?;
    let r_toks = tokenize(parts[1])?;
    let mut found_var: Option<char> = None;
    for t in l_toks.iter().chain(r_toks.iter()) {
        if let Tok::Var(ch) = t {
            match found_var {
                Some(prev) if prev != *ch => return Err(format!("multiple variables {prev}, {ch}")),
                None => found_var = Some(*ch),
                _ => {}
            }
        }
    }
    let var = found_var.ok_or("no variable found in linear equation")?;
    let mut pl = Parser { toks: &l_toks, pos: 0, var: Some(var) };
    let lf = pl.parse_expr()?;
    if pl.pos != l_toks.len() {
        return Err("trailing tokens on LHS".into());
    }
    let mut pr = Parser { toks: &r_toks, pos: 0, var: Some(var) };
    let rf = pr.parse_expr()?;
    if pr.pos != r_toks.len() {
        return Err("trailing tokens on RHS".into());
    }
    let net_a = lf.a - rf.a;
    let net_b = rf.b - lf.b;
    if net_a.abs() < 1e-12 {
        return Err("degenerate equation (zero variable coefficient)".into());
    }
    Ok((var, net_b / net_a))
}

/// Format float cleanly (`4` instead of `4.0`, up to 6 significant decimals otherwise).
pub fn format_alu_number(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 && v.abs() < 1e15 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Evaluate `[CALC: ...]` tags inside text OR verify explicit arithmetic/linear equation blocks
/// returned by `--general-alu` post-verification (pure general AST evaluation, zero lookup tables).
pub fn evaluate_alu_annotations(text: &str) -> Option<String> {
    // 1. Evaluate any `[CALC: expr]` or `solve(lhs = rhs; target_expr)` markers cleanly.
    if let Some(start) = text.find("[CALC:") {
        if let Some(rel_end) = text[start..].find(']') {
            let body = text[start + 6..start + rel_end].trim();
            if let Some((eq_s, tgt_s)) = body.split_once(';') {
                if let Ok((vname, xval)) = solve_linear_eq_1var(eq_s) {
                    let subst = tgt_s.replace(vname, &format!("({xval})"));
                    if let Ok(res) = eval_math_expr(&subst) {
                        return Some(format_alu_number(res));
                    }
                }
            } else if body.contains('=') {
                if let Ok((vname, xval)) = solve_linear_eq_1var(body) {
                    return Some(format!("{vname} = {}", format_alu_number(xval)));
                }
            } else if let Ok(res) = eval_math_expr(body) {
                return Some(format_alu_number(res));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_general_arithmetic_and_precedence() {
        assert!((eval_math_expr("12 * 8 - 27").unwrap() - 69.0).abs() < 1e-9);
        assert!((eval_math_expr("(47 * 23 - 156 / 12)").unwrap() - 1068.0).abs() < 1e-9);
        assert!((eval_math_expr("15% of 240 + 2^5").unwrap() - 68.0).abs() < 1e-9);
        assert!((eval_math_expr("gcd(84, 30) + lcm(6, 15) + 4!").unwrap() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn solve_randomized_linear_equations_general() {
        // Test across arbitrary unseen coefficients without any fixed prompt text
        for a in [-7i32, -3, 2, 5, 11] {
            for c in [-2i32, 1, 4] {
                if a == c {
                    continue;
                }
                for x_true in [-4i32, 0, 7, 13] {
                    let b = 9;
                    let d = (a - c) * x_true + b;
                    let eq = format!("{a}x + {b} = {c}x + {d}");
                    let (v, got) = solve_linear_eq_1var(&eq).unwrap();
                    assert_eq!(v, 'x');
                    assert!((got - x_true as f64).abs() < 1e-7, "eq={eq} got={got} true={x_true}");
                }
            }
        }
    }

    #[test]
    fn alu_annotation_evaluates_cleanly() {
        assert_eq!(evaluate_alu_annotations("Answer: [CALC: 5x + 3 = 38; 2x - 1]").as_deref(), Some("13"));
        assert_eq!(evaluate_alu_annotations("Result [CALC: 18 * 1000 * 40 / 60]").as_deref(), Some("12000"));
        assert!(eval_math_expr("1 / 0").is_err());
    }
}
