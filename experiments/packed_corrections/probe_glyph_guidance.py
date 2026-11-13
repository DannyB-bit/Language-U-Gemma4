"""Test an 8D context glyph as learned projection-error guidance.

The glyph is a 32-bit numeric record: 11-bit direction index in a uint16 slot,
plus fp16 magnitude. It is not an arbitrary-text codec, a 20.4-token proof,
an existing Cuneiform wire format, or a whole-model English-generation test.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from binary_factor_probe import PackedFactors

from paths import REPO as ROOT, DATA as AUDIT
data=np.load(AUDIT/"real_activation_inputs.npz",allow_pickle=False)
w=data["w"].astype("f8")
train,val,test=[data[k].astype("f8") for k in ("xtrain","xval","xtest")]
calibration=json.loads((AUDIT/"scale_calibration_probe.json").read_text())
chosen=calibration["variants"][calibration["selected_variant"]]
factor=PackedFactors(AUDIT/f"factors-scale-fit-{chosen['ridge_strength']}.bin")
u=np.unpackbits(factor.u,axis=1).astype("f8")*2-1
v=np.unpackbits(factor.v,axis=1).astype("f8")*2-1
q=(factor.sout[:,None]*(u@v))*factor.sin
teacher_train=train@w.T
residual=teacher_train-train@q.T
uu,ss,vv=np.linalg.svd(residual,full_matrices=False)
rank=8
cbraw=(AUDIT/"range-cache/s7_codebook_2048x8.bin").read_bytes()
if hashlib.sha256(cbraw).hexdigest()!="099cfea75908bf4ca74820c732e040179e5354f6cd023af099faf1d439c37a47":
    raise ValueError("Unexpected public codebook")
cb=np.frombuffer(cbraw,dtype="<f4").reshape(2048,8)
cb=cb/np.linalg.norm(cb,axis=1,keepdims=True)


def encode_guidance(x,a):
    z=np.asarray(x,dtype="f4")@a.T
    mag=np.linalg.norm(z,axis=1)
    direction=np.divide(z,mag[:,None],out=np.zeros_like(z),where=mag[:,None]>0)
    index=np.argmax(direction@cb.T,axis=1).astype("<u2")
    stored_mag=mag.astype("<f2")
    if not np.isfinite(stored_mag).all():
        raise ValueError("Guidance amplitude exceeds record range")
    records=np.empty((len(x),4),dtype="u1")
    records[:,:2]=index[:,None].view("u1")
    records[:,2:]=stored_mag[:,None].view("u1")
    return records


def decode_guidance(records):
    r=np.asarray(records,dtype="u1")
    if r.ndim!=2 or r.shape[1]!=4:
        raise ValueError("Expected 4-byte guidance records")
    index=r[:,:2].copy().view("<u2").ravel()
    magnitude=r[:,2:].copy().view("<f2").astype("f4")
    if np.any(index>=2048) or not np.isfinite(magnitude).all() or np.any(magnitude<0):
        raise ValueError("Invalid guidance index or amplitude")
    return cb[index]*magnitude


def metrics(pred,target):
    return {"relative_squared_error":float(np.sum((pred-target)**2)/np.sum(target*target)),
            "output_cosine":float(np.sum(pred*target)/np.linalg.norm(pred)/np.linalg.norm(target))}


models=[];variants=[]
gram=train@train.T
for ridge in (.01,.1,1.0):
    projection=np.linalg.solve(gram+np.eye(len(gram))*ridge*np.diag(gram).mean(),train)
    a=((np.sqrt(ss[:rank])[:,None]*uu[:,:rank].T)@projection).astype("f2").astype("f4")
    glyphs=encode_guidance(train,a)
    g=decode_guidance(glyphs).astype("f8")
    # Fit the decoder to the quantized guidance that inference actually sees.
    gh=g.T@g
    b=np.linalg.solve(gh+np.eye(8)*max(1e-12,1e-4*np.trace(gh)/8),
                      g.T@residual).T.astype("f2").astype("f4")
    pred=val@q.T+decode_guidance(encode_guidance(val,a))@b.T
    rec={"ridge":ridge,"validation":metrics(pred,val@w.T),
         "guide_bytes_per_projection_and_token":4,
         "guidance_projection_and_decoder_payload_bytes":int(a.size+b.size)*2,
         "shared_codebook_bytes":int(cb.nbytes)}
    models.append((a,b));variants.append(rec)
    print(json.dumps(rec),flush=True)
best=min(range(len(variants)),key=lambda i:variants[i]["validation"]["relative_squared_error"])
a,b=models[best]
base=test@q.T
records=encode_guidance(test,a)
g=decode_guidance(records)
correct=base+g@b.T
permutation=np.random.default_rng(20261112).permutation(len(test))
shuffled=base+g[permutation]@b.T
target=test@w.T
report={"scope":"8D quantized activation guidance on one real 128x5376 first-layer Q excerpt. Numeric error correction, not sentence compression or whole-model coherence.",
        "selection":"Validation error only; final control set reused from preceding component experiments.",
        "variants":variants,"selected_variant":best,
        "test_no_guidance":metrics(base,target),
        "test_correct_guidance":metrics(correct,target),
        "test_shuffled_guidance":metrics(shuffled,target),
        "note":"Every variant needs learned A/B sidecars. 4 bytes is the per-input temporary glyph, not the full cost of this capability."}
np.savez(AUDIT/"guidance_selected.npz",A=a.astype("f2"),B=b.astype("f2"),codebook=cb.astype("<f4"))
(AUDIT/"glyph_guidance_probe.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("test_no_guidance","test_correct_guidance","test_shuffled_guidance")}),flush=True)
