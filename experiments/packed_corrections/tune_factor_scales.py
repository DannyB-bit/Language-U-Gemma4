"""Fit existing BF16 scale slots against real calibration projection outputs.

Does not change the binary factors or increase their storage. This is a local
teacher-guided projection fit, not self-supervised accuracy recovery or a
whole-model NanoQuant reconstruction.
"""
import json
from pathlib import Path
import numpy as np
from binary_factor_probe import PackedFactors, save_packed, metric

from paths import REPO as ROOT, DATA as AUDIT
data=np.load(AUDIT/"real_activation_inputs.npz",allow_pickle=False)
w=data["w"].astype("f8")
train,val,test=[data[k].astype("f8") for k in ("xtrain","xval","xtest")]
from neon_runner.checkpoint_io import round_bf16
fit=json.loads((AUDIT/"binary_factor_probe.json").read_text())
selection=fit["variants"][fit["selected_index"]]
rank=selection["rank"];shrinkage=selection["input_shrinkage"]
path=AUDIT/f"factors-r{rank}-s{shrinkage}.bin"
p=PackedFactors(path)
u=np.unpackbits(p.u,axis=1).astype("f8")*2-1
v=np.unpackbits(p.v,axis=1).astype("f8")*2-1
f=u@v
base_so=p.sout.astype("f8");base_si=p.sin.astype("f8")
y=train@w.T


def cg(operator,b,initial,diagonal,steps=40):
    x=initial.copy()
    r=b-operator(x);z=r/diagonal;direction=z.copy();rz=r@z
    for _ in range(steps):
        hd=operator(direction);denom=direction@hd
        if denom<=0 or rz<1e-28:
            break
        alpha=rz/denom
        x+=alpha*direction;r-=alpha*hd
        newz=r/diagonal;newrz=r@newz
        if np.linalg.norm(r) < 1e-8*max(np.linalg.norm(b),1e-20):
            break
        direction=newz+(newrz/rz)*direction
        z=newz;rz=newrz
    return x


variants=[];models=[]
for strength in (.01,.1,1.0):
    so=base_so.copy();si=base_si.copy()
    for _ in range(2):
        # Fit output gains, then input gains, using only calibration tokens.
        pred=(train*si)@f.T
        so=round_bf16(np.sum(pred*y,axis=0)/np.maximum(np.sum(pred*pred,axis=0),1e-24)).astype("f8")
        diag=np.mean(train*train,axis=0)*np.sum((f*so[:,None])**2,axis=0)
        ridge=max(strength*float(diag.mean()),1e-16)
        def hessian(vector):
            error=((train*vector)@f.T)*(so*so)
            return np.mean(train*(error@f),axis=0)+ridge*vector
        rhs=np.mean(train*((y*so)@f),axis=0)+ridge*base_si
        si=round_bf16(cg(hessian,rhs,si,diag+ridge)).astype("f8")
    q=(so[:,None]*f)*si
    destination=AUDIT/f"factors-scale-fit-{strength}.bin"
    size=save_packed(destination,u,v,so,si)
    stored=PackedFactors(destination)
    pred=np.stack([stored.project(x) for x in val])
    dense=val@q.T
    parity=float(np.linalg.norm(pred-dense)/np.linalg.norm(dense))
    if parity>1e-5:
        raise ValueError("Packed inference mismatch after scale fitting")
    rec={"ridge_strength":strength,"file_bytes":size,
         "train":metric(q,train,y),"validation":metric(q,val,val@w.T),
         "packed_vs_dense_relative_error":parity}
    variants.append(rec);models.append(q)
    print(json.dumps(rec),flush=True)
best=min(range(len(variants)),key=lambda i:variants[i]["validation"]["relative_squared_error"])
result={"scope":"Existing BF16 scale-slot fitting on one real Q excerpt. Fixed factors, no model-level accuracy/coherence proof.",
        "selected_variant":best,"variants":variants,
        "before_test":metric((base_so[:,None]*f)*base_si,test,test@w.T),
        "after_test":metric(models[best],test,test@w.T)}
(AUDIT/"scale_calibration_probe.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"before_test":result["before_test"],"after_test":result["after_test"]}),flush=True)
