/-
  Language-U / Trinity-Law Formal Machine Contract in Lean 4
  Originated in: DannyB-bit/trinity-hybrid-retrieval
  Integrated into: DannyB-bit/Language-U-Gemma4 (trinity_retrieval/formal_verification/)
  Module: TrinityGates

  Design:
    All retrieval scores, channel weights, sieve weights, and thresholds are modeled in
    exact integer basis points (BP4 = 10_000, corresponding to `round(..., 4)` in Python),
    making every gate law and inequality 100% kernel-decidable with zero axioms and zero
    `sorry`. Matches verbatim:
      - `trinity_retrieval/roadmap_hybrid_v5_gemma.py` (6-channel RRF + Two-Tier Trust Law)
      - `trinity_retrieval/noise_stability_sieve.py` (3-scale stability sieve)
      - `experiments/run270_rust/src/t16c.rs` & `neon_runner/container_sniff.py` (Tile-16 byte laws)
-/

namespace Trinity

/-- Basis-point scale (`1.0000 = 10_000`), matching Python 4-decimal scoring precision. -/
def BP4 : Nat := 10000

/-- Trust decisions emitted by the Trinity Hybrid Retrieval Engine. -/
inductive Tier where
  | STRONG : Tier  -- Corroborated high-confidence hit; autonomous admission permitted
  | WEAK   : Tier  -- Plausible hit; admitted iff caller explicitly passes `weak = true`
  | MISS   : Tier  -- Honest refusal to guess; caller returns None
  deriving Repr, DecidableEq

/-- Verdicts emitted by `NoiseStabilitySieve` (`trinity_retrieval/noise_stability_sieve.py`). -/
inductive SieveVerdict where
  | PASS        : SieveVerdict
  | FORCE_WEAK  : SieveVerdict
  | REJECT_MISS : SieveVerdict
  deriving Repr, DecidableEq

/-- Combined output of the retrieval trust gate. -/
structure GateResult where
  tier     : Tier
  admitted : Bool
  deriving Repr, DecidableEq

/-- Parameterized retrieval trust-gate configuration (in `BP4` basis points). -/
structure GateConfig where
  w_bm25           : Nat
  w_sym            : Nat
  w_char           : Nat
  w_lsa            : Nat
  w_gemma          : Nat
  w_s7             : Nat
  tau_strong_score : Nat
  tau_weak_score   : Nat
  lex_strong_floor : Nat
  lex_weak_floor   : Nat
  deriving Repr, DecidableEq

/-- Exact production configuration from `trinity_retrieval/roadmap_hybrid_v5_gemma.py`. -/
def canonConfig : GateConfig :=
  { w_bm25           := 2800   -- W_BM25 = 0.28
  , w_sym            := 1600   -- W_SYM  = 0.16
  , w_char           := 2000   -- W_CHAR = 0.20
  , w_lsa            := 1200   -- W_LSA  = 0.12
  , w_gemma          := 2700   -- W_GEMMA = 0.27
  , w_s7             := 300    -- W_S7   = 0.03
  , tau_strong_score := 3020   -- STRONG_GATE = 0.302
  , tau_weak_score   := 2800   -- WEAK_GATE   = 0.280
  , lex_strong_floor := 800    -- LEX_STRONG_FLOOR = 0.080
  , lex_weak_floor   := 500    -- LEX_WEAK_FLOOR   = 0.050
  }

/-- Structural ordering invariant required of any sound trust-gate config. -/
def WellFormedConfig (cfg : GateConfig) : Prop :=
  cfg.tau_weak_score ≤ cfg.tau_strong_score ∧
  cfg.lex_weak_floor ≤ cfg.lex_strong_floor ∧
  cfg.lex_weak_floor > 0

theorem canonConfig_wellFormed : WellFormedConfig canonConfig := by
  unfold WellFormedConfig canonConfig
  decide

/-! ---------------------------------------------------------------------------
    Multi-Scale Noise Stability Sieve (`noise_stability_sieve.py`)
    --------------------------------------------------------------------------- -/

def SIEVE_W1 : Nat := 4500
def SIEVE_W2 : Nat := 3500
def SIEVE_W3 : Nat := 2000

def SIEVE_PASS_TH : Nat := 5500
def SIEVE_WEAK_TH : Nat := 3800
def SIEVE_PASS_S1 : Nat := 2500
def SIEVE_PASS_S2 : Nat := 3000
def SIEVE_WEAK_S1 : Nat := 1500

theorem sieve_weights_unit : SIEVE_W1 + SIEVE_W2 + SIEVE_W3 = BP4 := by
  rfl

/-- Exact fixed-point fusion formula from `NoiseStabilitySieve.evaluate`. -/
def sieveFusedBp (s1 s2 s3 : Nat) : Nat :=
  (SIEVE_W1 * s1 + SIEVE_W2 * s2 + SIEVE_W3 * s3) / BP4

/-- Exact verdict branching from `NoiseStabilitySieve.evaluate`. -/
def classifySieve (s1 s2 s3 : Nat) : SieveVerdict :=
  let fused := sieveFusedBp s1 s2 s3
  if fused ≥ SIEVE_PASS_TH ∧ s1 ≥ SIEVE_PASS_S1 ∧ s2 ≥ SIEVE_PASS_S2 then
    SieveVerdict.PASS
  else if fused ≥ SIEVE_WEAK_TH ∧ s1 ≥ SIEVE_WEAK_S1 then
    SieveVerdict.FORCE_WEAK
  else
    SieveVerdict.REJECT_MISS

theorem sieve_reject_when_s1_too_low (s1 s2 s3 : Nat) (h : s1 < SIEVE_WEAK_S1) :
    classifySieve s1 s2 s3 = SieveVerdict.REJECT_MISS := by
  unfold classifySieve
  have h1 : ¬ (s1 ≥ SIEVE_PASS_S1) := by unfold SIEVE_PASS_S1 SIEVE_WEAK_S1 at *; omega
  have h2 : ¬ (s1 ≥ SIEVE_WEAK_S1) := by omega
  simp [h1, h2]

theorem sieve_zero_input_rejected : classifySieve 0 0 0 = SieveVerdict.REJECT_MISS := by
  rfl

/-! ---------------------------------------------------------------------------
    Two-Tier Hybrid Search Gate (`roadmap_hybrid_v5_gemma.py`)
    --------------------------------------------------------------------------- -/

/-- Exact search tier classifier matching `roadmap_hybrid_v5_gemma.py` lines 424-429. -/
def classifyTier (cfg : GateConfig) (score lex : Nat) : Tier :=
  if score ≥ cfg.tau_strong_score ∧ lex ≥ cfg.lex_strong_floor then
    Tier.STRONG
  else if score ≥ cfg.tau_weak_score ∧ lex ≥ cfg.lex_weak_floor then
    Tier.WEAK
  else
    Tier.MISS

/-- Composed sieve + search gate evaluation with caller opt-in (`weak=True/False`). -/
def evaluateSearchGate
    (cfg : GateConfig)
    (is_noise : Bool)
    (verdict : SieveVerdict)
    (score lex : Nat)
    (caller_opt_in : Bool) : GateResult :=
  let base := classifyTier cfg score lex
  let eff :=
    if is_noise then
      Tier.MISS
    else
      match verdict with
      | SieveVerdict.REJECT_MISS => Tier.MISS
      | SieveVerdict.FORCE_WEAK =>
          match base with
          | Tier.STRONG => Tier.WEAK
          | other       => other
      | SieveVerdict.PASS => base
  let ok :=
    match eff with
    | Tier.STRONG => true
    | Tier.WEAK   => caller_opt_in
    | Tier.MISS   => false
  { tier := eff, admitted := ok }

/-! ---------------------------------------------------------------------------
    Safety & Monotonicity Theorems (Zero `sorry`, Zero Custom Axioms)
    --------------------------------------------------------------------------- -/

/-- Noise Refusal Law: any noise-flagged query unconditionally yields `MISS` and `admitted = false`. -/
theorem noise_refusal_guarantee
    (cfg : GateConfig) (verdict : SieveVerdict) (score lex : Nat) (caller_opt_in : Bool)
    (h_noise : is_noise = true) :
    evaluateSearchGate cfg is_noise verdict score lex caller_opt_in =
      { tier := Tier.MISS, admitted := false } := by
  subst h_noise
  rfl

/-- Sieve Refusal Law: a `REJECT_MISS` verdict unconditionally forces `MISS` and blocks admission. -/
theorem sieve_reject_forces_miss
    (cfg : GateConfig) (is_noise : Bool) (score lex : Nat) (caller_opt_in : Bool) :
    evaluateSearchGate cfg is_noise SieveVerdict.REJECT_MISS score lex caller_opt_in =
      { tier := Tier.MISS, admitted := false } := by
  cases is_noise <;> rfl

/-- Zero-Confabulation Law: sub-strong lexical corroboration can NEVER emit `STRONG`. -/
theorem zero_confabulation_guarantee
    (cfg : GateConfig) (is_noise : Bool) (verdict : SieveVerdict)
    (score lex : Nat) (caller_opt_in : Bool)
    (h_lex : lex < cfg.lex_strong_floor) :
    (evaluateSearchGate cfg is_noise verdict score lex caller_opt_in).tier ≠ Tier.STRONG := by
  have h_not_strong : classifyTier cfg score lex ≠ Tier.STRONG := by
    unfold classifyTier
    split
    · omega
    · split <;> decide
  cases is_noise <;> cases verdict <;> simp [evaluateSearchGate]
  · cases h_cls : classifyTier cfg score lex <;> simp
    exact h_not_strong h_cls
  · exact h_not_strong

/-- Lexical Floor Law: below `lex_weak_floor`, neither `STRONG` nor `WEAK` can ever fire. -/
theorem weak_lexical_floor_guarantee
    (cfg : GateConfig) (h_wf : WellFormedConfig cfg)
    (is_noise : Bool) (verdict : SieveVerdict)
    (score lex : Nat) (caller_opt_in : Bool)
    (h_lex : lex < cfg.lex_weak_floor) :
    evaluateSearchGate cfg is_noise verdict score lex caller_opt_in =
      { tier := Tier.MISS, admitted := false } := by
  rcases h_wf with ⟨_, h_lex_ord, _⟩
  have h_miss : classifyTier cfg score lex = Tier.MISS := by
    unfold classifyTier
    split
    · omega
    · split
      · omega
      · rfl
  cases is_noise <;> cases verdict <;> simp [evaluateSearchGate, h_miss]

/-- Pure Neural Zero-Lex Refusal: with `lex = 0`, even infinite neural score is rejected. -/
theorem pure_neural_zero_lex_never_admitted
    (cfg : GateConfig) (h_wf : WellFormedConfig cfg)
    (is_noise : Bool) (verdict : SieveVerdict) (score : Nat) (caller_opt_in : Bool) :
    evaluateSearchGate cfg is_noise verdict score 0 caller_opt_in =
      { tier := Tier.MISS, admitted := false } := by
  have h_pos : 0 < cfg.lex_weak_floor := h_wf.2.2
  exact weak_lexical_floor_guarantee cfg h_wf is_noise verdict score 0 caller_opt_in h_pos

/-- Opt-In Safeguard: an admitted `WEAK` result strictly requires `caller_opt_in = true`. -/
theorem weak_requires_opt_in
    (cfg : GateConfig) (is_noise : Bool) (verdict : SieveVerdict)
    (score lex : Nat) (caller_opt_in : Bool)
    (h_tier : (evaluateSearchGate cfg is_noise verdict score lex caller_opt_in).tier = Tier.WEAK)
    (h_adm : (evaluateSearchGate cfg is_noise verdict score lex caller_opt_in).admitted = true) :
    caller_opt_in = true := by
  cases caller_opt_in
  · cases is_noise <;> cases verdict <;> simp [evaluateSearchGate] at h_tier h_adm
    · cases h_cls : classifyTier cfg score lex <;> simp [h_cls] at h_tier h_adm
    · cases h_cls : classifyTier cfg score lex <;> simp [h_cls] at h_tier h_adm
  · rfl

/-- Non-Weak Opt-Out Law: with `caller_opt_in = false`, only `STRONG` may be admitted. -/
theorem opt_out_never_emits_weak
    (cfg : GateConfig) (is_noise : Bool) (verdict : SieveVerdict) (score lex : Nat)
    (h_adm : (evaluateSearchGate cfg is_noise verdict score lex false).admitted = true) :
    (evaluateSearchGate cfg is_noise verdict score lex false).tier = Tier.STRONG := by
  cases is_noise <;> cases verdict <;> simp [evaluateSearchGate] at h_adm ⊢
  · cases h_cls : classifyTier cfg score lex <;> simp [h_cls] at h_adm ⊢
  · cases h_cls : classifyTier cfg score lex <;> simp [h_cls] at h_adm

/-- Sieve Downgrade Law: `FORCE_WEAK` strictly bars autonomous `STRONG` emission. -/
theorem sieve_force_weak_prevents_strong
    (cfg : GateConfig) (is_noise : Bool) (score lex : Nat) (caller_opt_in : Bool) :
    (evaluateSearchGate cfg is_noise SieveVerdict.FORCE_WEAK score lex caller_opt_in).tier ≠
      Tier.STRONG := by
  cases is_noise <;> simp [evaluateSearchGate]
  cases h_cls : classifyTier cfg score lex <;> simp

/-- Score Floor Law: below `tau_weak_score`, retrieval always returns `MISS`. -/
theorem score_below_weak_always_miss
    (cfg : GateConfig) (h_wf : WellFormedConfig cfg)
    (is_noise : Bool) (verdict : SieveVerdict)
    (score lex : Nat) (caller_opt_in : Bool)
    (h_score : score < cfg.tau_weak_score) :
    evaluateSearchGate cfg is_noise verdict score lex caller_opt_in =
      { tier := Tier.MISS, admitted := false } := by
  rcases h_wf with ⟨h_sc_ord, _, _⟩
  have h_miss : classifyTier cfg score lex = Tier.MISS := by
    unfold classifyTier
    split
    · omega
    · split
      · omega
      · rfl
  cases is_noise <;> cases verdict <;> simp [evaluateSearchGate, h_miss]

/-- Monotonicity of `STRONG` classification under non-decreasing score and lexical corroboration. -/
theorem strong_preserved_under_increase
    (cfg : GateConfig) (score score' lex lex' : Nat)
    (h_strong : classifyTier cfg score lex = Tier.STRONG)
    (h_s : score ≤ score') (h_l : lex ≤ lex') :
    classifyTier cfg score' lex' = Tier.STRONG := by
  unfold classifyTier at h_strong ⊢
  split at h_strong
  · next h_cur =>
    have h_next : score' ≥ cfg.tau_strong_score ∧ lex' ≥ cfg.lex_strong_floor := by omega
    simp [h_next]
  · split at h_strong <;> cases h_strong

/-- Monotonicity of admission under non-decreasing `(score, lex)` with opt-in preserved. -/
theorem admitted_monotone_pass
    (cfg : GateConfig) (_h_wf : WellFormedConfig cfg)
    (score score' lex lex' : Nat) (optIn optIn' : Bool)
    (h_adm : (evaluateSearchGate cfg false SieveVerdict.PASS score lex optIn).admitted = true)
    (h_s : score ≤ score') (h_l : lex ≤ lex')
    (h_opt : optIn = false ∨ optIn' = true) :
    (evaluateSearchGate cfg false SieveVerdict.PASS score' lex' optIn').admitted = true := by
  simp only [evaluateSearchGate, Bool.false_eq_true, ↓reduceIte] at h_adm ⊢
  by_cases h_str : score ≥ cfg.tau_strong_score ∧ lex ≥ cfg.lex_strong_floor
  · have h_str' : score' ≥ cfg.tau_strong_score ∧ lex' ≥ cfg.lex_strong_floor := by omega
    simp [classifyTier, h_str']
  · by_cases h_wk : score ≥ cfg.tau_weak_score ∧ lex ≥ cfg.lex_weak_floor
    · have h_opt_in : optIn = true := by
        simpa [classifyTier, h_str, h_wk] using h_adm
      rcases h_opt with h_f | h_t
      · simp [h_f] at h_opt_in
      · subst h_t
        by_cases h_str' : score' ≥ cfg.tau_strong_score ∧ lex' ≥ cfg.lex_strong_floor
        · simp [classifyTier, h_str']
        · have h_wk' : score' ≥ cfg.tau_weak_score ∧ lex' ≥ cfg.lex_weak_floor := by omega
          simp [classifyTier, h_str', h_wk']
    · simp [classifyTier, h_str, h_wk] at h_adm

/-! ---------------------------------------------------------------------------
    Language-U Container Size Laws & Pi 4 RAM Budget Bounds
    --------------------------------------------------------------------------- -/

/-- Exact Tile-16 block byte formula matching `experiments/run270_rust/src/t16c.rs` & `pack_t16c_v3.py`. -/
def tile16BlkBytes (bsize nBits : Nat) (isAsym : Bool) : Nat :=
  if nBits == 5 then
    32 + ((bsize + 4) / 5) * 16
  else
    let hdr := if isAsym then 64 else 32
    hdr + (bsize / 8) * 16 * nBits

theorem tile16_mode0_1b_asym_bytes : tile16BlkBytes 256 1 true = 576 := by rfl
theorem tile16_mode1_tern5_bytes  : tile16BlkBytes 256 5 false = 864 := by rfl
theorem tile16_mode2_2b_sym_bytes : tile16BlkBytes 256 2 false = 1056 := by rfl
theorem tile16_mode3_2b_asym_bytes : tile16BlkBytes 256 2 true = 1088 := by rfl
theorem tile16_mode4_3b_sym_bytes : tile16BlkBytes 256 3 false = 1568 := by rfl

theorem tile16_blk_pos (bsize nBits : Nat) (isAsym : Bool) :
    tile16BlkBytes bsize nBits isAsym ≥ 32 := by
  unfold tile16BlkBytes
  split
  · omega
  · dsimp only
    split <;> omega

/-- Exact census remainder-zero laws & Pi 4 V3 RAM fit (`6,375,598,887 B < 6,849,200,128 B`). -/
theorem qwen9b_census_law_exact :
    8630071296 = 33711216 * 256 ∧ 8630071296 % 256 = 0 := by
  decide

theorem gemma31b_census_law_exact :
    114401280 * 256 = 29286727680 ∧ 29286727680 % 256 = 0 := by
  decide

theorem pi4_v3_ram_budget_fits_cold_boot :
    6375598887 < 6849200128 := by
  omega

/-! ---------------------------------------------------------------------------
    Kernel-Checked Regression Battery Witnesses (`rfl`)
    --------------------------------------------------------------------------- -/

theorem battery_strong_hit :
    evaluateSearchGate canonConfig false SieveVerdict.PASS 3100 900 false =
      { tier := Tier.STRONG, admitted := true } := by
  rfl

theorem battery_weak_blocked_without_opt_in :
    evaluateSearchGate canonConfig false SieveVerdict.PASS 2900 600 false =
      { tier := Tier.WEAK, admitted := false } := by
  rfl

theorem battery_weak_admitted_with_opt_in :
    evaluateSearchGate canonConfig false SieveVerdict.PASS 2900 600 true =
      { tier := Tier.WEAK, admitted := true } := by
  rfl

theorem battery_pure_neural_zero_lex_refused :
    evaluateSearchGate canonConfig false SieveVerdict.PASS 9900 0 true =
      { tier := Tier.MISS, admitted := false } := by
  rfl

theorem battery_sieve_reject_refused :
    evaluateSearchGate canonConfig false SieveVerdict.PASS 3500 1200 true =
      { tier := Tier.STRONG, admitted := true } ∧
    evaluateSearchGate canonConfig false SieveVerdict.REJECT_MISS 3500 1200 true =
      { tier := Tier.MISS, admitted := false } := by
  exact ⟨rfl, rfl⟩

theorem battery_sieve_force_weak_caps_strong :
    evaluateSearchGate canonConfig false SieveVerdict.FORCE_WEAK 3500 1200 false =
      { tier := Tier.WEAK, admitted := false } := by
  rfl

end Trinity
