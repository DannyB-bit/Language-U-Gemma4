"""Locally authored, template-disjoint arithmetic language development suite.

Not an external benchmark or evidence of broad language-model reasoning.
The split is frozen before training. Numeric strings are masked by the learner.
"""

NAMES = ('add','subtract','multiply','divide','minimum','maximum','power','remainder')
TRAIN = {
 'add': ['add {n} to the value','increase by {n}','sum the value and {n}',
         'increment the value by {n}','put {n} more into the total','add another {n}',
         'raise the total by {n}','the value plus {n}','gain {n} units','receive {n} more'],
 'subtract': ['subtract {n} from the value','decrease by {n}','reduce the value by {n}',
              'deduct {n} from the total','remove {n} units','take away {n}',
              'the value minus {n}','lose {n} units','spend {n} units','subtract another {n}'],
 'multiply': ['multiply the value by {n}','scale by a factor of {n}','take the product with {n}',
              'times {n}','multiply by {n}','scale the value by {n}',
              'make {n} copies of the total','repeat the whole amount {n} times',
              'the value multiplied by {n}','product of the value and {n}'],
 'divide': ['divide the value by {n}','take the integer quotient by {n}',
            'split into {n} equal shares','divide evenly by {n}','the value divided by {n}',
            'distribute equally among {n} groups','quotient of the value and {n}',
            'divide into {n} identical portions','split the amount equally into {n}',
            'divide the total by {n}'],
 'minimum': ['take the minimum with {n}','choose the smaller of the value and {n}',
             'cap the value at {n}','use the lesser value compared with {n}',
             'minimum of the total and {n}','keep the smaller amount versus {n}',
             'limit the total to at most {n}','select the lower value against {n}',
             'take whichever is smaller compared to {n}','the lesser of the value and {n}'],
 'maximum': ['take the maximum with {n}','choose the larger of the value and {n}',
             'floor the value at {n}','use the greater value compared with {n}',
             'maximum of the total and {n}','keep the larger amount versus {n}',
             'ensure the total is at least {n}','select the higher value against {n}',
             'take whichever is larger compared to {n}','the greater of the value and {n}'],
 'power': ['raise the value to power {n}','use exponent {n}','exponentiate to {n}',
           'compute the value raised to {n}','take power {n} of the value',
           'the value to the power of {n}','apply exponent {n} to the total',
           'raise the total to exponent {n}','compute power {n}',
           'exponentiate the value with exponent {n}'],
 'remainder': ['take the remainder modulo {n}','compute the remainder after division by {n}',
               'use the value mod {n}','reduce modulo {n}','the remainder with divisor {n}',
               'take modulo {n}','keep the remainder when divided by {n}',
               'apply modulus {n}','compute the value modulo {n}',
               'return the leftover remainder after dividing by {n}'],
}
VALIDATION = {
 'add':['increase the current total by {n}','add {n} additional units','sum with {n}','receive another {n}'],
 'subtract':['reduce the current total by {n}','subtract {n} units','deduct another {n}','lose another {n}'],
 'multiply':['multiply the current total by {n}','scale the total by {n}','product with {n}','repeat this amount {n} times'],
 'divide':['divide the current total by {n}','quotient with divisor {n}','split the total into {n} equal parts','divide equally by {n}'],
 'minimum':['minimum with {n}','choose the lesser of the total and {n}','cap the total at {n}','keep whichever is smaller than {n}'],
 'maximum':['maximum with {n}','choose the greater of the total and {n}','floor the total at {n}','keep whichever is larger than {n}'],
 'power':['raise the current total to power {n}','exponentiate with {n}','take the total to exponent {n}','compute the total raised to {n}'],
 'remainder':['remainder modulo {n}','compute the remainder with divisor {n}','use modulus {n}','keep the remainder after dividing by {n}'],
}
TEST = {
 'add':['add {n} extra units to the running total','increase this amount by {n}',
         'sum the running value with {n}','increment by {n}','gain another {n}',
         'the current amount plus {n}'],
 'subtract':['subtract {n} units from the running total','decrease this amount by {n}',
              'deduct {n} units now','remove another {n}','the current amount minus {n}',
              'take {n} away from the total'],
 'multiply':['multiply this amount by {n}','scale the current amount by {n}',
              'compute the product with {n}','make {n} copies of this amount',
              'the current amount times {n}','repeat the current amount {n} times'],
 'divide':['divide this amount by {n}','compute the integer quotient with {n}',
            'split this amount into {n} equal groups','the current amount divided by {n}',
            'distribute the total among {n} equal groups','divide the running total by {n}'],
 'minimum':['keep the minimum of the current amount and {n}','select the smaller amount versus {n}',
             'cap this amount at {n}','choose the lesser amount versus {n}',
             'take the lower of this amount and {n}','limit this amount to at most {n}'],
 'maximum':['keep the maximum of the current amount and {n}','select the larger amount versus {n}',
             'floor this amount at {n}','choose the greater amount versus {n}',
             'take the higher of this amount and {n}','ensure this amount is at least {n}'],
 'power':['raise this amount to the power {n}','apply exponent {n}',
           'compute this amount raised to {n}','take the current amount to power {n}',
           'use exponent {n} on this amount','raise the running total to exponent {n}'],
 'remainder':['take the current amount modulo {n}','keep the remainder on division by {n}',
               'find the remainder after dividing the total by {n}','use the current amount mod {n}',
               'reduce this amount modulo {n}','apply modulus {n} to the current total'],
}

def examples(split, number=17):
    return [(text.format(n=number), i) for i,name in enumerate(NAMES) for text in split[name]]

def validate_splits():
    a,b,c=[{s for items in split.values() for s in items} for split in (TRAIN,VALIDATION,TEST)]
    if a&b or a&c or b&c:
        raise ValueError('Template overlap')
    return {'train':len(a),'validation':len(b),'test':len(c)}
