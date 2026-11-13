"""Train a narrow neural operator policy, select on validation, freeze a bank.

NumPy STE training for QAT4 and a float32 control. No test data is imported.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from joint_codec import (Glyph,QAT4,EXACT32,ADD,compile_chunk,decode_weights,
                         exact_encode,instruction)
from neural_planner import Planner,features,MODEL_TEXT,TOKENIZER_SHA,FEATURE_VERSION,MODEL_BLUEPRINT
from planner_data import TRAIN,VALIDATION,examples


def quantize(weights):
    w=np.asarray(weights,dtype='f4').reshape(40,32)
    scale=np.maximum(np.abs(w).max(1)/7,1e-8)
    bits=scale.view('u4')
    scale=((bits+0x7fff+((bits>>16)&1))&0xffff0000).view('f4')
    codes=np.clip(np.rint(w/scale[:,None]),-8,7).astype('i2')+8
    lanes=[]
    for i in range(5):
        c=codes[i*8:(i+1)*8].reshape(-1).astype('u1')
        packed=(c[::2] | c[1::2]<<4).tobytes()
        scales=(scale[i*8:(i+1)*8].view('u4')>>16).astype('<u2').tobytes()
        lanes.append(packed+scales)
    decoded=np.concatenate([decode_weights(QAT4,lane) for lane in lanes])
    return decoded,lanes


def metrics(weights,texts,labels):
    probs=Planner(weights).probabilities(texts)
    return {'accuracy':float(np.mean(probs.argmax(1)==labels)),
            'cross_entropy':float(-np.log(np.maximum(probs[np.arange(len(labels)),labels],1e-12)).mean())}


def train(seed,mode,validation):
    rng=np.random.default_rng(seed)
    texts,labels=zip(*examples(TRAIN))
    x=np.stack([features(t) for t in texts]).astype('f8'); y=np.asarray(labels)
    master=np.concatenate([rng.normal(0,.3,1024),rng.normal(0,.15,256)])
    m=np.zeros_like(master);v=np.zeros_like(master)
    candidates=[]
    for step in range(1,1601):
        # The backward estimator is identity through quantization, explicitly STE.
        forward=quantize(master)[0].astype('f8') if mode=='qat4' and step>=300 else master
        a,b=forward[:1024].reshape(32,32),forward[1024:].reshape(32,8)
        h=np.tanh(x@a);scores=h@b;scores-=scores.max(1,keepdims=True)
        p=np.exp(scores);p/=p.sum(1,keepdims=True)
        p[np.arange(len(y)),y]-=1;p/=len(y)
        gb=h.T@p;ga=x.T@((p@b.T)*(1-h*h))
        grad=np.concatenate([ga.reshape(-1),gb.reshape(-1)])+1e-4*master
        m=.9*m+.1*grad;v=.999*v+.001*grad*grad
        master-=.015*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step in (400,800,1200,1600):
            stored=quantize(master)[0] if mode=='qat4' else master.astype('f4')
            val=metrics(stored,*validation)
            candidates.append(({'seed':seed,'mode':mode,'step':step,'validation':val},master.copy(),stored.copy()))
    return candidates


def make_bank(weights,mode,tokenizer):
    ids=tokenizer.encode(MODEL_TEXT,add_special_tokens=False).ids
    if len(ids)!=102:raise ValueError(f'Model description must have exactly 102 tokens, got {len(ids)}')
    counts=(20,20,20,21,21);glyphs=[];pos=0
    if mode=='qat4':
        restored,lanes=quantize(weights)
        codec=QAT4
    else:
        restored=np.asarray(weights,dtype='<f4')
        lanes=[exact_encode(w,'fp32') for w in restored.reshape(5,256)]
        codec=EXACT32
    for n,lane,descriptor in zip(counts,lanes,MODEL_BLUEPRINT):
        glyphs.append(Glyph(ids[pos:pos+n],codec,lane,instruction(ADD,left=descriptor,right=0)))
        pos+=n
    return compile_chunk(glyphs,bytes.fromhex(TOKENIZER_SHA),hashlib.sha256(restored.tobytes()).digest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    raw=args.tokenizer.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TOKENIZER_SHA:raise ValueError('Tokenizer mismatch')
    tokenizer=Tokenizer.from_str(raw.decode())
    texts,labels=zip(*examples(VALIDATION,61));validation=(texts,np.asarray(labels))
    candidates=[]
    for mode in ('fp32','qat4'):
        for seed in (7,29,83):
            candidates.extend(train(seed,mode,validation))
            print(json.dumps({'mode':mode,'seed':seed,'last':candidates[-1][0]}),flush=True)
    # Accuracy first, then compact QAT representation, then validation log loss.
    selected=min(range(len(candidates)),key=lambda i:(-candidates[i][0]['validation']['accuracy'],
                    candidates[i][0]['mode']!='qat4',candidates[i][0]['validation']['cross_entropy']))
    info,master,stored=candidates[selected]
    bank=make_bank(master,info['mode'],tokenizer)
    loaded=Planner.from_glyphs(bank)
    np.testing.assert_array_equal(loaded.weights,stored)
    (args.output/'planner.luj').write_bytes(bank)
    manifest={'scope':'Narrow arithmetic operator classifier; not a language model or GPT-equivalence proof.',
              'architecture':[32,32,8],'parameter_count':1280,'feature_extractor':'fixed signed word/character hashing',
              'feature_version':FEATURE_VERSION,
              'inference_source_sha256':hashlib.sha256((Path(__file__).parent/'neural_planner.py').read_bytes()).hexdigest(),
              'training_templates':len(examples(TRAIN)),'validation_templates':len(examples(VALIDATION)),
              'test_accessed':False,'selection':'Validation accuracy, then QAT4 compactness, then validation cross entropy.',
              'selected':info,'candidates':[c[0] for c in candidates],
              'model_bytes':len(bank),'model_sha256':hashlib.sha256(bank).hexdigest(),
              'decoded_fp32_sha256':hashlib.sha256(loaded.weights.tobytes()).hexdigest(),
              'tokenizer_sha256':TOKENIZER_SHA,'tokenizer_bytes_used_for_metadata_encoding':len(raw),
              'metadata_tokens':102,'metadata_is_user_question':False,
              'training_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data_source_sha256':hashlib.sha256((Path(__file__).parent/'planner_data.py').read_bytes()).hexdigest()}
    (args.output/'planner_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'selected':info,'model_bytes':len(bank),'model_sha256':manifest['model_sha256']}),flush=True)


if __name__=='__main__':main()
