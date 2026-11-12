# NumPy component port of SamsungLabs/NanoQuant's LB-ADMM initializer.
# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for this experiment from commit a9e0a430881ff80d83b622c3129e330dc33c04f5,
# src/nanoquant/core/admm_nq.py. This is not the full NanoQuant algorithm:
# no output-gradient KFAC, block STE refinement or model KL reconstruction.
# Original LICENSE/NOTICE accompany the experiment bundle.
from pathlib import Path
import hashlib
import json
import struct
import sys

import numpy as np

from paths import REPO as ROOT, DATA as AUDIT
sys.path.insert(0,str(ROOT))
from neon_runner.checkpoint_io import bf16_to_float32, round_bf16


def sign(x):
    return np.where(x >= 0, 1., -1.).astype("f4")


def project_magnitudes(x, rng):
    """Rank-one absolute magnitudes with fixed signs, as in upstream SVID."""
    a=np.abs(x)
    v=rng.standard_normal(a.shape[1]).astype("f4")
    v/=max(float(np.linalg.norm(v)),1e-12)
    for _ in range(5):
        u=a@v;u/=max(float(np.linalg.norm(u)),1e-12)
        v=a.T@u;v/=max(float(np.linalg.norm(v)),1e-12)
    u=a@v
    return u[:,None]*v[None,:]*sign(x)


def solve_step(x,y,z,u,rho,reg=.03):
    gram=x.T@x
    gram=(gram+gram.T)*.5
    diag=np.diag_indices_from(gram)
    gram[diag]+=max(float(rho*np.diag(gram).mean()+reg),1e-12)
    return np.linalg.solve(gram,x.T@y+rho*(z-u)).astype("f4")


def initialize(w,input_variance,rank,iterations=400,seed=42):
    rng=np.random.default_rng(seed)
    norm_i=np.sqrt(np.maximum(input_variance,1e-12)).astype("f4")
    target=w*norm_i
    # Upstream initializes duals from unconstrained minus projected factors.
    # Preserve the actual random unprojected factors explicitly.
    als=rng.standard_normal((len(w),rank)).astype("f4")
    bls=rng.standard_normal((rank,w.shape[1])).astype("f4")
    az=project_magnitudes(als,rng)
    bz=project_magnitudes(bls,rng)
    au=als-az;bu=bls-bz
    for i in range(iterations):
        rho=i/iterations
        nb=np.maximum(np.linalg.norm(bz,axis=1),1e-12)
        als=solve_step(bz.T/nb,target.T,az.T,au.T,rho).T
        na=np.maximum(np.linalg.norm(az,axis=0),1e-12)
        bls=solve_step(az/na,target,bz,bu,rho)
        az=project_magnitudes(als+au,rng)
        bz=project_magnitudes(bls+bu,rng)
        au+=als-az;bu+=bls-bz
    # Actual packed-binary inference scales, not the floating-factor W_final.
    af=az
    bf=bz/norm_i
    balance=np.sqrt(max(float(np.linalg.norm(bf)),1e-12)/max(float(np.linalg.norm(af)),1e-12))
    af=af*balance/np.maximum(np.linalg.norm(az,axis=0),1e-12)
    bf=bf/balance
    s_out=round_bf16(np.abs(af).mean(1))
    s_in=round_bf16(np.abs(bf).mean(0))
    return sign(af),sign(bf),s_out,s_in


HEADER=struct.Struct("<4s7I4Q")


def save_packed(path,u,v,sout,sin):
    m,r=u.shape
    assert v.shape[0]==r
    n=v.shape[1]
    if r%8 or n%8:
        raise ValueError("This component format requires byte-aligned rows")
    s=round_bf16(np.concatenate((sout,sin))).view("u4")
    scales=(s>>16).astype("<u2").tobytes()
    up=np.packbits(u>0,axis=1).tobytes()
    vp=np.packbits(v>0,axis=1).tobytes()
    raw=HEADER.pack(b"BFP1",1,m,n,r,1,0,0,len(scales),len(up),len(vp),len(scales)+len(up)+len(vp))
    with path.open("xb") as f:
        f.write(raw);f.write(scales);f.write(up);f.write(vp)
    return path.stat().st_size


class PackedFactors:
    def __init__(self,path):
        with path.open("rb") as f:
            raw=f.read(64)
        if len(raw)!=64:
            raise ValueError("Truncated header")
        magic,ver,m,n,r,dt,flags,res,ss,us,vs,ps=HEADER.unpack(raw)
        if (magic!=b"BFP1" or ver!=1 or dt!=1 or flags or res
                or min(m,n,r)<=0 or r%8 or n%8 or ss!=2*(m+n)
                or us!=m*r//8 or vs!=r*n//8 or ps!=ss+us+vs
                or path.stat().st_size!=64+ps):
            raise ValueError("Invalid packed factor format")
        self.shape=(m,n,r)
        self.data=np.memmap(path,dtype="u1",mode="r")
        self.scales=bf16_to_float32(self.data[64:64+ss].copy().view("<u2"))
        if not np.isfinite(self.scales).all():
            raise ValueError("Nonfinite scales")
        self.sout,self.sin=self.scales[:m],self.scales[m:]
        self.u=self.data[64+ss:64+ss+us].reshape(m,r//8)
        self.v=self.data[64+ss+us:].reshape(r,n//8)

    @staticmethod
    def matvec(packed,x,chunk=32):
        out=np.empty(len(packed),dtype="f4")
        for begin in range(0,len(packed),chunk):
            decoded=np.unpackbits(packed[begin:begin+chunk],axis=1).astype("f4")*2-1
            out[begin:begin+chunk]=decoded@x
        return out

    def project(self,x):
        h=self.matvec(self.v,np.asarray(x,dtype="f4")*self.sin)
        return self.matvec(self.u,h)*self.sout


def metric(q,x,y):
    pred=x@q.T
    return {"relative_squared_error":float(np.sum((pred-y)**2)/np.sum(y*y)),
            "output_cosine":float(np.sum(pred*y)/np.linalg.norm(pred)/np.linalg.norm(y))}


def main():
    data=np.load(AUDIT/"real_activation_inputs.npz",allow_pickle=False)
    w=data["w"].astype("f4")
    xtrain,xval,xtest=[data[k].astype("f4") for k in ("xtrain","xval","xtest")]
    variance=np.mean(xtrain*xtrain,axis=0)
    records=[];models=[]
    for rank in (80,104,120):
        for shrinkage in (0.0,0.6):
            iv=(1-shrinkage)*variance+shrinkage*variance.mean()
            u,v,so,si=initialize(w,iv,rank)
            q=(so[:,None]*(u@v))*si
            path=AUDIT/f"factors-r{rank}-s{shrinkage}.bin"
            size=save_packed(path,u,v,so,si)
            packed=PackedFactors(path)
            # Compare independent dense-float64 and chunked stored-factor paths
            # on every validation vector, not only a constructed toy vector.
            packed_y=np.stack([packed.project(x) for x in xval])
            dense_y=xval.astype("f8")@q.astype("f8").T
            parity=float(np.linalg.norm(packed_y-dense_y)/np.linalg.norm(dense_y))
            if parity>1e-5:
                raise ValueError("Stored-factor inference mismatch")
            rec={"rank":rank,"input_shrinkage":shrinkage,"bytes":size,
                 "bpw_including_header_and_scales":size*8/w.size,
                 "validation":metric(q,xval,xval@w.T),
                 "packed_vs_dense_relative_error":parity,
                 "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            records.append(rec);models.append(q)
            print(json.dumps(rec),flush=True)
    selected_idx=min(range(len(records)),key=lambda i:records[i]["validation"]["relative_squared_error"])
    result={"scope":"NumPy initializer and packed-factor component only; not complete NanoQuant, no 31B model quality/Pi proof.",
            "upstream_code_revision":"a9e0a430881ff80d83b622c3129e330dc33c04f5",
            "variants":records,"selected_index":selected_idx,
            "selected_test":metric(models[selected_idx],xtest,xtest@w.T)}
    (AUDIT/"binary_factor_probe.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"selected_index":selected_idx,"selected_test":result["selected_test"]}),flush=True)


if __name__=="__main__":
    main()
