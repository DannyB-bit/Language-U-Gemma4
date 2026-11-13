#!/usr/bin/env python3
"""Independent PyTorch mathematical reference for Gemma3 270M (NOT Transformers).
Validation only; no Python in native C++ runtime. Uses local verified Safetensors.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
from safetensors import safe_open

def load_checkpoint(path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {key: f.get_tensor(key).float().contiguous() for key in f.keys()}

@torch.inference_mode()
def logits_for_ids(w, ids):
    eps=1e-6
    cache=[[[], []] for _ in range(18)]
    def norm(x,key):
        return x*torch.rsqrt((x*x).mean(dim=-1,keepdim=True)+eps)*(1.0+w[key])
    def linear(x,key):
        return torch.nn.functional.linear(x,w[key])
    def rope(q,pos,base):
        idx=torch.arange(128,dtype=torch.float64)
        theta=pos*torch.pow(torch.tensor(base,dtype=torch.float64),-idx/128)
        c=theta.cos().float()
        s=theta.sin().float()
        a=q[...,:128]
        b=q[...,128:]
        return torch.cat((a*c-b*s,a*s+b*c),dim=-1)
    final=None
    for pos,token in enumerate(ids):
        h=w["model.embed_tokens.weight"][int(token)].clone()*math.sqrt(640)
        for layer in range(18):
            pref="model.layers."+str(layer)+"."
            att=pref+"self_attn."
            local=layer%6!=5
            base=10000.0 if local else 1000000.0
            x=norm(h,pref+"input_layernorm.weight")
            q=linear(x,att+"q_proj.weight").view(4,256)
            k=linear(x,att+"k_proj.weight").view(1,256)
            v=linear(x,att+"v_proj.weight").view(1,256)
            q=norm(q,att+"q_norm.weight")
            k=norm(k,att+"k_norm.weight")
            q=rope(q,pos,base)
            k=rope(k,pos,base)
            keys,values=cache[layer]
            keys.append(k.reshape(256))
            values.append(v.reshape(256))
            if local:
                keys=keys[-512:]
                values=values[-512:]
            ks=torch.stack(keys,dim=0)
            vs=torch.stack(values,dim=0)
            score=(q@ks.T)*(1.0/16.0)
            prob=torch.softmax(score.float(),dim=-1)
            att_result=(prob@vs).reshape(1024)
            o=linear(att_result,att+"o_proj.weight")
            h=h+norm(o,pref+"post_attention_layernorm.weight")
            ff=norm(h,pref+"pre_feedforward_layernorm.weight")
            gate=linear(ff,pref+"mlp.gate_proj.weight")
            up=linear(ff,pref+"mlp.up_proj.weight")
            activated=torch.nn.functional.gelu(gate,approximate="tanh")*up
            down=linear(activated,pref+"mlp.down_proj.weight")
            h=h+norm(down,pref+"post_feedforward_layernorm.weight")
        final=h
    return linear(norm(final,"model.norm.weight"),"model.embed_tokens.weight")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--ids",required=True)
    p.add_argument("--logits-out",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    ids=[int(t) for t in args.ids.split(",")]
    if not ids or len(ids)>64 or any(x<0 or x>=262144 for x in ids):
        p.error("invalid token IDs")
    if args.logits_out.exists():
        p.error("refusing to overwrite logits")
    torch.set_num_threads(args.threads)
    weights=load_checkpoint(args.checkpoint)
    if len(weights)!=236 or sum(t.numel() for t in weights.values())!=268098176:
        raise RuntimeError("source tensor census mismatch")
    y=logits_for_ids(weights,ids)
    if y.size()!=(262144,) or not torch.isfinite(y).all():
        raise RuntimeError("nonfinite logits")
    y.numpy().astype("<f4",copy=False).tofile(args.logits_out)
    print(json.dumps({"reference":"independent_PyTorch_not_official_Transformers",
                      "argmax":int(y.argmax()),"token_count":len(ids),
                      "output":str(args.logits_out)}))

if __name__=="__main__":
    main()
