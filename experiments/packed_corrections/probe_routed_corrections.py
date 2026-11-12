"""Equal-rank-budget context routing; development projection experiment."""
import json
from pathlib import Path
import numpy as np
from binary_factor_probe import PackedFactors

from paths import DATA as P
d=np.load(P/"real_activation_inputs.npz",allow_pickle=False)
w=d["w"]
tr,va,te=[d[k] for k in ("xtrain","xval","xtest")]
old=json.loads((P/"scale_calibration_probe.json").read_text())
s=old["variants"][old["selected_variant"]]["ridge_strength"]
f=PackedFactors(P/f"factors-scale-fit-{s}.bin")
u=np.unpackbits(f.u,axis=1).astype("f8")*2-1
v=np.unpackbits(f.v,axis=1).astype("f8")*2-1
q=f.sout[:,None]*(u@v)*f.sin
err=tr@(w-q).T

def stored(x):
    return np.asarray(x,dtype="f2").astype("f8")

def fit(x,e,rank,ridge):
    U,S,V=np.linalg.svd(e,full_matrices=False)
    gram=x@x.T
    proj=np.linalg.solve(gram+np.eye(len(x))*ridge*np.diag(gram).mean(),x)
    A=stored((np.sqrt(S[:rank])[:,None]*U[:,:rank].T)@proj)
    B=stored(V[:rank].T*np.sqrt(S[:rank]))
    return A,B

def score(pred,target):
    return dict(relative_squared_error=float(np.sum((pred-target)**2)/np.sum(target**2)),
                output_cosine=float(np.sum(pred*target)/np.linalg.norm(pred)/np.linalg.norm(target)))

# Only eight selected channels form the router: no extra dense projection.
idx=np.argsort(np.var(tr,axis=0))[-8:].astype("u2")
mean=stored(tr[:,idx].mean(0))
std=stored(np.maximum(tr[:,idx].std(0),1e-4))
features=(tr[:,idx]-mean)/std

def cluster(k):
    rng=np.random.default_rng(42)
    centers=features[rng.choice(len(features),k,replace=False)].copy()
    for _ in range(40):
        assignments=((features[:,None]-centers)**2).sum(2).argmin(1)
        updated=np.stack([features[assignments==j].mean(0) if np.any(assignments==j) else centers[j] for j in range(k)])
        if np.allclose(updated,centers):break
        centers=updated
    return stored(centers)

def route(x,centers):
    z=stored((x[:,idx]-mean)/std)
    return ((z[:,None]-centers)**2).sum(2).argmin(1)

models=[];rows=[]
for k in (1,2,4):
    centers=cluster(k) if k>1 else None
    assignment=route(tr,centers) if centers is not None else np.zeros(len(tr),dtype=int)
    counts=np.bincount(assignment,minlength=k)
    if min(counts)<8//k+2:
        continue
    for ridge in (.01,.1,1.,10.):
        factors=[fit(tr[assignment==j],err[assignment==j],8//k,ridge) for j in range(k)]
        def predict(x,shuffle=False):
            selected=route(x,centers) if centers is not None else np.zeros(len(x),dtype=int)
            if shuffle:selected=np.random.default_rng(113).permutation(selected)
            out=x@q.T
            for j,(A,B) in enumerate(factors):
                mask=selected==j
                out[mask]+=(x[mask]@A.T)@B.T
            return out
        result=dict(experts=k,total_rank=8,ridge=ridge,training_counts=counts.tolist(),
                    factor_payload_bytes=sum((a.size+b.size)*2 for a,b in factors),
                    router_bytes=0 if k==1 else int(idx.nbytes+2*(mean.size+std.size+centers.size)),
                    validation=score(predict(va),va@w.T))
        rows.append(result);models.append((centers,factors))
        print(json.dumps(result),flush=True)
best=min(range(len(rows)),key=lambda i:rows[i]["validation"]["relative_squared_error"])
centers,factors=models[best]
def evaluate(x,shuffle=False):
    selected=route(x,centers) if centers is not None else np.zeros(len(x),dtype=int)
    if shuffle:selected=np.random.default_rng(113).permutation(selected)
    out=x@q.T
    for j,(A,B) in enumerate(factors):
        mask=selected==j
        out[mask]+=(x[mask]@A.T)@B.T
    return score(out,x@w.T)
report=dict(scope="One 128x5376 projection, reused development test. No full-model/coherence proof.",
            selection="Validation error; test never used to fit or select.",
            activation_representation="Unquantized 8D latent in control; 16 bytes if stored FP16, versus prior 4-byte codebook guide. Computation here uses float64.",
            shuffled_control_interpretation="If one expert wins, shuffling routes is a no-op and is not evidence of routing benefit.",
            candidates=rows,selected=rows[best],test=evaluate(te),shuffled_route_test=evaluate(te,True),
            previous_quantized_8d_test=json.loads((P/"glyph_guidance_probe.json").read_text())["test_correct_guidance"])
state={"selected_channels":idx,"mean":mean.astype("f2"),"std":std.astype("f2")}
if centers is not None:state["centers"]=centers.astype("f2")
for j,(A,B) in enumerate(factors):
    state[f"A{j}"]=A.astype("f2")
    state[f"B{j}"]=B.astype("f2")
np.savez(P/"routed_correction_selected.npz",**state)
(P/"routed_correction_probe.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("selected","test","shuffled_route_test","previous_quantized_8d_test")}),flush=True)
