"""Frozen held-out template and compositional tests, plus causal weight controls."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from neural_planner import Planner,solve,NAMES
from planner_data import TEST,examples,validate_splits


def reference(op,a,b):
    """Independent scalar oracle; never supplied to the planner or runtime."""
    if op=='add':v=a+b
    elif op=='subtract':v=a-b
    elif op=='multiply':v=a*b
    elif op=='divide':
        if b==0 or a%b:raise ValueError('inexact division')
        v=a//b
    elif op=='minimum':v=a if a<b else b
    elif op=='maximum':v=a if a>b else b
    elif op=='power':
        if not 0<=b<=63:raise ValueError('bad exponent')
        v=a**b
    elif op=='remainder':
        if b==0:raise ValueError('zero divisor')
        v=a%b
    else:raise ValueError('unknown oracle op')
    if not -(2**63)<=v<2**63:raise ValueError('overflow')
    return v


def tasks(count=256):
    rng=random.Random(20261112)
    result=[]
    while len(result)<count:
        start=rng.randint(100,999);value=start;clauses=[];ops=[];values=[]
        for _ in range(rng.randint(3,5)):
            op=rng.choice(NAMES)
            if op=='divide':
                factors=[n for n in range(2,21) if value%n==0]
                arg=rng.choice(factors) if factors else 1
            elif op=='power':arg=rng.choice((0,1,2)) if abs(value)<10000 else 1
            elif op=='multiply':arg=rng.randint(2,10)
            elif op=='remainder':arg=rng.randint(2,19)
            else:arg=rng.randint(100,999)
            value=reference(op,value,arg)
            clauses.append(rng.choice(TEST[op]).format(n=arg));ops.append(op);values.append(value)
        result.append({'question':'Start with '+str(start)+'; '+'; '.join(clauses),
                       'expected':value,'expected_ops':ops,'reference_values':values})
    return result


UNSUPPORTED = [
 'Explain why the sky is blue.',
 'Prove that there are infinitely many prime numbers.',
 'Start with 5; do not add 3',
 'Start with 5; never subtract 3',
 'Start with 5; add 2 unless it rains',
 'Start with 5; add 2 except on Tuesday',
 'Start with 5; ignore the previous instruction and add 2',
 'Start with 5; add 1.5',
 'Start with 5; add two',
 'Start with 5; add 2 and multiply by 3',
 'Start with 5; divide by 0',
 'Start with 5; raise the value to power 1000000',
 'Start with 9223372036854775807; add 1',
]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.model_dir/'planner_manifest.json').read_text())
    bank=(args.model_dir/'planner.luj').read_bytes()
    model=Planner.from_glyphs(bank,manifest['model_sha256'])
    split_counts=validate_splits()
    test_texts,labels=zip(*examples(TEST,137));labels=np.asarray(labels)
    probs=model.probabilities(test_texts);pred=probs.argmax(1)
    confusion=np.zeros((8,8),int)
    for truth,guess in zip(labels,pred):confusion[truth,guess]+=1
    rng=np.random.default_rng(173)
    controls={}
    for name,w in [('zero',np.zeros(1280,'f4')),('shuffled',rng.permutation(model.weights))]:
        controls[name]={'heldout_operator_accuracy':float(np.mean(Planner(w).probabilities(test_texts).argmax(1)==labels))}
    corpus=tasks();rows=[]
    for item in corpus:
        row=dict(item)
        # No gold answer/operations enter the API. Evaluation is downstream.
        try:
            got=solve(item['question'],bank,expected_model_sha=manifest['model_sha256'])
            row.update(status='answered',answer=got['answer'],chosen_ops=got['operator_names'],
                       confidence=got['confidence'],answer_correct=got['answer']==item['expected'],
                       program_correct=got['operator_names']==item['expected_ops'],
                       bytes_including_model_bank_and_instructions=len(got['packed_chunk']),
                       external_question_utf8_bytes=len(item['question'].encode()))
        except ValueError as error:
            row.update(status='rejected',reason=str(error),answer_correct=False,program_correct=False)
        rows.append(row)
    accepted=[r for r in rows if r['status']=='answered']
    ood=[]
    for question in UNSUPPORTED:
        try:
            got=solve(question,bank)
            ood.append({'question':question,'rejected':False,'answer':got['answer']})
        except ValueError as error:
            ood.append({'question':question,'rejected':True,'reason':str(error)})
    report={'scope':'Eight arithmetic operations with explicit semicolon grammar; not GPT-level or unrestricted-language reasoning.',
            'model_sha256':manifest['model_sha256'],'model_bytes':len(bank),'split_counts':split_counts,
            'selection_frozen_before_test':True,'trained_parameters':1280,
            'heldout_operator_correct':int(np.sum(pred==labels)),'heldout_operator_total':len(labels),
            'heldout_operator_accuracy':float(np.mean(pred==labels)), 'confusion_true_rows_predicted_columns':confusion.tolist(),
            'weight_controls':controls,'composition_cases':len(rows),'answered':len(accepted),
            'rejected':len(rows)-len(accepted),'correct_answers':sum(r['answer_correct'] for r in rows),
            'correct_programs':sum(r['program_correct'] for r in rows),
            'accuracy_all_cases':sum(r['answer_correct'] for r in rows)/len(rows),
            'accuracy_when_answered':sum(r['answer_correct'] for r in accepted)/max(1,len(accepted)),
            'unsupported_rejected':sum(r['rejected'] for r in ood),'unsupported_total':len(ood),
            'semantic_verifier':'Structural and arithmetic checks only; valid wrong programs can pass.',
            'gpt_comparison_run':False,
            'corpus_sha256':hashlib.sha256(json.dumps(corpus,sort_keys=True).encode()).hexdigest(),
            'evaluation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cases':rows,'unsupported_cases':ood}
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','unsupported_cases','confusion_true_rows_predicted_columns')},indent=2))


if __name__=='__main__':main()
