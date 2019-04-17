## tokenize code
from tokenize import generate_tokens
try:
    from cStringIO import StringIO
except:
    from io import StringIO
import token as tk

from nltk.tokenize import word_tokenize
import re
import ast
import astor

QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")

def canonicalize_intent(intent):
    # handle the following special case: quote is `''`
    marked_token_matches = QUOTED_TOKEN_RE.findall(intent)

    slot_map = dict()
    var_id = 0
    str_id = 0
    for match in marked_token_matches:
        quote = match[0]
        value = match[1]
        quoted_value = quote + value + quote

        # determine slot type
        if quote == '`' and value.isidentifier():
            slot_type = 'var'
        else:
            slot_type = 'str'

        if slot_type == 'var':
            slot_name = 'var_%d' % var_id
            var_id += 1
        else:
            slot_name = 'str_%d' % str_id
            str_id += 1


        intent = intent.replace(quoted_value, slot_name)
        slot_map[slot_name] = {'value': value.strip().encode().decode('unicode_escape', 'ignore'),
                               'quote': quote,
                               'type': slot_type}

    return intent, slot_map
   
def replace_identifiers_in_ast(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue

            if isinstance(v, str):
                if v in identifier2slot:
                    slot_name = identifier2slot[v]

                    setattr(node, k, slot_name)
    return
    
def is_enumerable_str(identifier_value):
    """
    Test if the quoted identifier value is a list
    """
    return len(identifier_value) > 2 and identifier_value[0] in ('{', '(', '[') and identifier_value[-1] in ('}', ']', ')')
   
def canonicalize_code(code, slot_map):
    string2slot = {x['value']: slot_name for slot_name, x in list(slot_map.items())}

    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, string2slot)
    canonical_code = astor.to_source(py_ast).strip()

    entries_that_are_lists = [slot_name for slot_name, val in slot_map.items() if is_enumerable_str(val['value'])]
    if entries_that_are_lists:
        for slot_name in entries_that_are_lists:
            list_repr = slot_map[slot_name]['value']
            first_token = list_repr[0]
            last_token = list_repr[-1]
            fake_list = first_token + slot_name + last_token
            slot_map[fake_list] = slot_map[slot_name]
            canonical_code = canonical_code.replace(list_repr, fake_list)
    return canonical_code

def tokenize_intent(intent):
    intent, slot_map = canonicalize_intent(intent)
    return word_tokenize(intent), slot_map
    
def tokenize_code(code, slot_map=None):
    if slot_map != None:
        code = canonicalize_code(code, slot_map)
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    for toknum, tokval, _, _, _ in token_stream:
        if toknum == tk.ENDMARKER:
            break
        tokens.append(tokval)

    return tokens