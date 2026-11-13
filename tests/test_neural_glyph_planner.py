"""Neural-bank use, executable constraints and honest result boundaries."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/joint_glyphs'))
from joint_codec import (Glyph,ADD,DIV_EXACT,MINIMUM,MAXIMUM,POWER,REMAINDER,
                         instruction,compile_chunk,decode_chunk,execute)
from neural_planner import Planner,solve,parse_question,TOKENIZER_SHA,MODEL_BLUEPRINT
from planner_data import validate_splits


class NeuralPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'experiments/joint_glyphs/planner'
        cls.bank=(p/'planner.luj').read_bytes()
        cls.manifest=json.loads((p/'planner_manifest.json').read_text())

    def test_entire_model_decodes_from_five_glyphs(self):
        model=Planner.from_glyphs(self.bank,self.manifest['model_sha256'])
        self.assertEqual(model.weights.size,1280)
        self.assertEqual(hashlib.sha256(model.weights.tobytes()).hexdigest(),self.manifest['decoded_fp32_sha256'])
        glyphs,_=decode_chunk(self.bank,bytes.fromhex(TOKENIZER_SHA))
        self.assertEqual(sum(len(g.tokens) for g in glyphs),102)
        self.assertEqual(tuple(execute(glyphs)['values']),MODEL_BLUEPRINT)

    def test_actual_learned_plan_and_stored_execution(self):
        answer=solve('Start with 12; multiply by 8; subtract 27; add 15',self.bank)
        self.assertEqual(answer['operator_names'],['multiply','subtract','add'])
        self.assertEqual(answer['answer'],84)
        self.assertFalse(answer['semantic_correctness_certified'])
        glyphs,_=decode_chunk(answer['packed_chunk'],bytes.fromhex(TOKENIZER_SHA))
        self.assertEqual(execute(glyphs)['result'],84)
        # Executable plans cannot silently masquerade as architecture descriptors.
        with self.assertRaises(ValueError):Planner.from_glyphs(answer['packed_chunk'])

    def test_bad_model_digest_and_blueprint(self):
        with self.assertRaises(ValueError):Planner.from_glyphs(self.bank,'0'*64)
        glyphs,source=decode_chunk(self.bank,bytes.fromhex(TOKENIZER_SHA))
        glyphs[-1].instruction=instruction(ADD,left=999,right=0)
        bad=compile_chunk(glyphs,bytes.fromhex(TOKENIZER_SHA),source)
        with self.assertRaises(ValueError):Planner.from_glyphs(bad)

    def test_zero_weights_change_learned_distribution(self):
        loaded=Planner.from_glyphs(self.bank)
        text=['multiply by 8']
        p=loaded.probabilities(text)
        zero=Planner(np.zeros(1280,'f4')).probabilities(text)
        np.testing.assert_allclose(zero,np.full((1,8),.125))
        self.assertGreater(float(np.linalg.norm(p-zero)),.2)

    def test_negation_ambiguity_and_unbounded_input_rejected(self):
        for q in ('Start with 1; do not add 2','Start with 1; add 2 and multiply by 3',
                  'Start with 1; add 1.5','Explain mathematics.','Start with 1; '+('a'*2100)):
            with self.assertRaises(ValueError):parse_question(q)

    def run_op(self,op,a,b):
        glyphs,_=decode_chunk(self.bank,bytes.fromhex(TOKENIZER_SHA))
        g=Glyph(glyphs[0].tokens,glyphs[0].codec,glyphs[0].weights,instruction(op,left=a,right=b))
        return execute([g])['result']

    def test_new_arithmetic_opcodes(self):
        for op,a,b,expected in [(DIV_EXACT,42,7,6),(MINIMUM,-3,9,-3),
                                (MAXIMUM,-3,9,9),(POWER,3,4,81),(REMAINDER,23,5,3)]:
            self.assertEqual(self.run_op(op,a,b),expected)

    def test_new_arithmetic_failures(self):
        for op,a,b in [(DIV_EXACT,5,2),(DIV_EXACT,5,0),(REMAINDER,5,0),
                        (POWER,2,-1),(POWER,2,64),(POWER,2,63)]:
            with self.assertRaises(ValueError):self.run_op(op,a,b)

    def test_splits_and_observed_semantic_failure_remain_visible(self):
        self.assertEqual(validate_splits(),{'train':80,'validation':32,'test':48})
        r=json.loads((ROOT/'experiments/joint_glyphs/planner/heldout_results.json').read_text())
        wrong=[x for x in r['cases'] if x['status']=='answered' and not x['answer_correct']]
        self.assertGreater(len(wrong),0)
        self.assertFalse(r['gpt_comparison_run'])


if __name__=='__main__':unittest.main()
