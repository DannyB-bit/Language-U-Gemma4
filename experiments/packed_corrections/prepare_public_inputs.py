"""Download only the pinned public excerpts required for the component probes."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

from paths import DATA
CACHE=DATA/"range-cache"
CACHE.mkdir(parents=True,exist_ok=True)
SOURCE="https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/resolve/52f3f65bc7a02d555763bc923bd1d9094898219d/model.safetensors?download=true"
SIZE=23265352448
EXPECTED={"header.bin":"1a3708d583d5c13949aaea986c2be3cc8bd7031b547ea57c1a5c4b6e76f139cb","model.language_model.layers.0.self_attn.q_proj.weight_packed.bin":"3dc90ad3509ba1a2f057f4a6d04635f83b2ec8ea3e07fa8edb98875436787654","model.language_model.layers.0.self_attn.q_proj.weight_scale.bin":"36439c06300c935c5ebdee90731fb715a1839983b1e0bce4f1e6493fb210f13b"}


def ranged(label,start,end):
    path=CACHE/label
    if path.exists():
        raw=path.read_bytes()
    else:
        req=Request(SOURCE+f"&range_start={start}",headers={"Range":f"bytes={start}-{end}"})
        with urlopen(req,timeout=90) as response:
            if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZE}":
                raise ValueError("Unexpected HTTP range")
            raw=response.read(end-start+2)
    if len(raw)!=end-start+1 or hashlib.sha256(raw).hexdigest()!=EXPECTED[label]:
        raise ValueError("Excerpt length/hash mismatch")
    if not path.exists():
        path.write_bytes(raw)
    return raw


header=json.loads(ranged("header.bin",0,266887)[8:])
for suffix,width in (("weight_packed",4),("weight_scale",2)):
    key="model.language_model.layers.0.self_attn.q_proj."+suffix
    item=header[key]
    start=266888+item["data_offsets"][0]
    length=128*item["shape"][1]*width
    ranged(key+".bin",start,start+length-1)
for name,expected in [
    ("s7_codebook_2048x8.bin","099cfea75908bf4ca74820c732e040179e5354f6cd023af099faf1d439c37a47"),
    ("gemma4_31b_tensor_layout.json","5229403e3d93dde3f7719eef16f735b5c1a76611afd9423c75b98af214306bc8")]:
    path=CACHE/name
    if path.exists():
        raw=path.read_bytes()
    else:
        url="https://huggingface.co/TheAiCollectiveART/Glyph-Stream-31b/resolve/bd9c48f1da41b51521570674ba2696657eb7358e/"+name
        raw=urlopen(url,timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError("Public companion hash mismatch")
    if not path.exists():
        path.write_bytes(raw)
print("Verified pinned public header, Q excerpt/scales, codebook and layout")

