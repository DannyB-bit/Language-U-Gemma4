"""Byte arithmetic for unbuilt binary-factor + Q6 embedding candidates."""
from fractions import Fraction
import json
from pathlib import Path

from paths import DATA as audit
layout=json.loads((audit/"range-cache/gemma4_31b_tensor_layout.json").read_text())
matrices=layout["projections"]
weights=sum(p["d_in"]*p["d_out"] for p in matrices)
records=[]
for bits in (Fraction(4,5),Fraction(9,10),Fraction(1),Fraction(9,8)):
    total=0
    for p in matrices:
        m,n=p["d_out"],p["d_in"]
        budget=m*n*bits.numerator//bits.denominator
        rank=((budget-512-16*(m+n))//(m+n))//8*8
        if rank<8:
            raise ValueError("Unsupported tiny rank")
        total+=64+2*(m+n)+(m+n)*rank//8
    embedding=32+(262144*5376//256)*194
    records.append(dict(target_linear_bpw=float(bits),linear_bytes=total,
                        proposed_q6_embedding_bytes=embedding,
                        existing_companion_allowance=3263993,
                        assumed_runtime_bytes=500000000,
                        projected_bytes_before_guidance=total+embedding+3263993+500000000))
print(json.dumps({"scope":"Dimension calculation only; no built full model, quality/RAM claim.",
                  "records":records,
                  "optional_rank8_guidance_payload":sum(p["d_in"]+p["d_out"] for p in matrices)*16,
                  "optional_extra_codebook_and_guidance_headers":65536+64*len(matrices)},indent=2))
