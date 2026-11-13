"""A complete 1,280-parameter learned operator classifier carried in five glyphs.

This is a narrow neural planner, not a transformer or a GPT-equivalent model.
Question segmentation and operand parsing follow an explicit small grammar.
Operation selection is neural; executable validity is not semantic correctness.
"""
import hashlib
import re

import numpy as np

from joint_codec import (Glyph, EXACT32, EXACT16, QAT4, ADD, SUB, MUL, DIV_EXACT, MINIMUM,
                         MAXIMUM, POWER, REMAINDER, compile_chunk, decode_chunk,
                         decode_weights, instruction, execute)

TOKENIZER_SHA = 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'
OPS = (ADD,SUB,MUL,DIV_EXACT,MINIMUM,MAXIMUM,POWER,REMAINDER)
NAMES = ('add','subtract','multiply','divide','minimum','maximum','power','remainder')
MIN_CONFIDENCE = 0.55
FEATURE_VERSION = 'signed-hash-stopwords-v2'
MODEL_BLUEPRINT = (32,32,8,1,2)  # input, hidden, output, tanh ID, feature version ID
STOPWORDS = frozenset(('a an the this that value total current running amount by with '
                      'from of to into and another extra additional now units whole '
                      'number').split())
NEGATION = re.compile(r"\b(not|never|unless|except|don't|instead|ignore)\b",re.I)
MODEL_TEXT = (
 'These five glyphs contain the complete trained weights of a small learned neural planner. '
 'The planner reads an arithmetic instruction and proposes an operation. A separate integer '
 'interpreter checks dependencies, division, bounds, and execution. The available operations '
 'are addition, subtraction, multiplication, exact division, minimum, maximum, integer power, '
 'and remainder. Every prediction uses the restored weights. The model has thirty two input '
 'features, thirty two hidden units, and eight outputs. Successful arithmetic execution does '
 'not prove that the model understood the question correctly.'
)


def features(text):
    """Fixed signed word/character hashing, with masked numbers; not learned embeddings."""
    text=re.sub(r'-?\d+', ' number ', text.lower())
    words=[w for w in re.findall(r'[a-z]+',text) if w not in STOPWORDS]
    atoms=[('w:'+w,1.) for w in words]
    atoms += [('c:'+w[i:i+3],.3) for w in words for i in range(max(0,len(w)-2))]
    atoms += [('b:'+a+' '+b,.4) for a,b in zip(words,words[1:])]
    vector=np.zeros(32,'f4')
    for atom,scale in atoms:
        digest=hashlib.blake2s(atom.encode(),digest_size=8).digest()
        index=int.from_bytes(digest[:4],'little')%31
        vector[index] += scale * (1 if digest[4]&1 else -1)
    norm=np.linalg.norm(vector)
    if norm:vector/=norm
    vector[-1]=1.
    return vector


class Planner:
    def __init__(self, weights):
        w=np.asarray(weights,dtype='<f4')
        if w.shape!=(1280,) or not np.isfinite(w).all():
            raise ValueError('Expected all 1,280 finite learned weights')
        self.weights=w.copy()
        self.first=self.weights[:1024].reshape(32,32)
        self.second=self.weights[1024:].reshape(32,8)

    @classmethod
    def from_glyphs(cls,raw,expected_sha=None):
        if expected_sha is not None and hashlib.sha256(raw).hexdigest()!=expected_sha:
            raise ValueError('Model-bank digest mismatch')
        glyphs,_=decode_chunk(raw,bytes.fromhex(TOKENIZER_SHA))
        if len(glyphs)!=5 or any(g.codec not in (EXACT32,EXACT16,QAT4) for g in glyphs):
            raise ValueError('Expected five recoverable neural model glyphs')
        if tuple(execute(glyphs)['values']) != MODEL_BLUEPRINT:
            raise ValueError('Incompatible model architecture or feature blueprint')
        return cls(np.concatenate([decode_weights(g.codec,g.weights) for g in glyphs]))

    def probabilities(self,texts):
        x=np.stack([features(text) for text in texts])
        scores=np.tanh(x@self.first)@self.second
        scores-=scores.max(axis=1,keepdims=True)
        p=np.exp(scores)
        return p/p.sum(axis=1,keepdims=True)


def parse_question(question):
    if not isinstance(question,str) or len(question)>2000 or NEGATION.search(question):
        raise ValueError('Unsupported question or negation/conditional')
    segments=[s.strip() for s in question.rstrip('.').split(';')]
    if not 2<=len(segments)<=6:
        raise ValueError('Use Start with <integer>; followed by one to five clauses')
    first=re.fullmatch(r'Start with (-?\d+)',segments[0],re.I)
    if first is None:
        raise ValueError('Missing explicit initial integer')
    start=int(first.group(1))
    if not -(2**63)<=start<2**63:
        raise ValueError('Initial value out of range')
    args=[]
    for clause in segments[1:]:
        if not re.fullmatch(r'[A-Za-z0-9\s-]+',clause):
            raise ValueError('Unsupported clause syntax')
        numbers=re.findall(r'(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])',clause)
        if len(numbers)!=1:
            raise ValueError('Each clause requires exactly one integer operand')
        value=int(numbers[0])
        if not -(2**63)<=value<2**63:
            raise ValueError('Operand out of range')
        args.append(value)
    return start,segments[1:],args


def solve(question,model_bank,*,threshold=MIN_CONFIDENCE,expected_model_sha=None):
    """No expected answer or labeled operations are accepted by the inference API."""
    if not 0<=threshold<=1:
        raise ValueError('Invalid confidence threshold')
    model=Planner.from_glyphs(model_bank,expected_model_sha)
    start,clauses,args=parse_question(question)
    p=model.probabilities(clauses)
    choices=p.argmax(axis=1)
    confidence=p[np.arange(len(clauses)),choices]
    if np.any(confidence<threshold):
        raise ValueError('Neural planner abstained on low confidence')
    bank,_=decode_chunk(model_bank,bytes.fromhex(TOKENIZER_SHA))
    generated=[]
    for i,glyph in enumerate(bank):
        if i<len(clauses):
            op=instruction(OPS[int(choices[i])],left_ref=-1 if i==0 else i-1,
                           left=start if i==0 else 0,right=args[i])
        else:
            # Explicit identity padding for the fixed five-record model bank.
            op=instruction(ADD,left_ref=i-1,right=0)
        generated.append(Glyph(glyph.tokens,glyph.codec,glyph.weights,op))
    source=hashlib.sha256(model.weights.tobytes()).digest()
    packed=compile_chunk(generated,bytes.fromhex(TOKENIZER_SHA),source)
    restored,_=decode_chunk(packed,bytes.fromhex(TOKENIZER_SHA))
    result=execute(restored)
    return {'answer':result['result'],'operator_names':[NAMES[int(i)] for i in choices],
            'confidence':[float(x) for x in confidence], 'values':result['values'][:len(clauses)],
            'english':f"For the proposed operations, the result is {result['result']}.",
            'model_sha256':hashlib.sha256(model_bank).hexdigest(),
            'question_sha256':hashlib.sha256(question.encode()).hexdigest(),
            'semantic_correctness_certified':False,'packed_chunk':packed}
