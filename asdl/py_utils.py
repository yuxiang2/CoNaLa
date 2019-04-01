# coding=utf-8

from __future__ import print_function

import token as tk
try:
    from cStringIO import StringIO
except:
    from io import StringIO
from tokenize import generate_tokens

import re
import nltk
import ast
import astor

import sys
from asdl_ast import RealizedField, AbstractSyntaxTree

QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")

## tokenize code
def tokenize_code(code, mode=None):
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.ENDMARKER:
            break

        if mode == 'decoder':
            if toknum == tk.STRING:
                quote = tokval[0]
                tokval = tokval[1:-1]
                tokens.append(quote)
                tokens.append(tokval)
                tokens.append(quote)
            elif toknum == tk.DEDENT:
                continue
            else:
                tokens.append(tokval)
        elif mode == 'canonicalize':
            if toknum == tk.STRING:
                tokens.append('_STR_')
            elif toknum == tk.DEDENT:
                continue
            else:
                tokens.append(tokval)
        else:
            tokens.append(tokval)

    return tokens

## tokenize intent    
def tokenize_intent(intent):
    lower_intent = intent.lower()
    tokens = nltk.word_tokenize(lower_intent)

    return tokens

## decide value is variable or string    
def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'

## identify variable and strings in the intent, and replace them
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

        slot_type = infer_slot_type(quote, value)

        if slot_type == 'var':
            slot_name = 'var_%d' % var_id
            var_id += 1
            slot_type = 'var'
        else:
            slot_name = 'str_%d' % str_id
            str_id += 1
            slot_type = 'str'

        intent = intent.replace(quoted_value, slot_name)
        slot_map[slot_name] = {'value': value.strip().encode().decode('unicode_escape', 'ignore'),
                               'quote': quote,
                               'type': slot_type}

    return intent, slot_map
    
## 
def replace_identifiers_in_ast(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            # Python 3
            # if isinstance(v, str) or isinstance(v, unicode):
            if isinstance(v, str):
                if v in identifier2slot:
                    slot_name = identifier2slot[v]
                    # Python 3
                    # if isinstance(slot_name, unicode):
                    #     try: slot_name = slot_name.encode('ascii')
                    #     except: pass

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
            #if list_repr[0] == '[' and list_repr[-1] == ']':
            first_token = list_repr[0]  # e.g. `[`
            last_token = list_repr[-1]  # e.g., `]`
            fake_list = first_token + slot_name + last_token
            slot_map[fake_list] = slot_map[slot_name]
            # else:
            #     fake_list = slot_name

            canonical_code = canonical_code.replace(list_repr, fake_list)

    return canonical_code

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def python_ast_to_asdl_ast(py_ast_node, grammar):
    # node should be composite
    py_node_name = type(py_ast_node).__name__
    # assert py_node_name.startswith('_ast.')

    production = grammar.get_prod_by_ctr_name(py_node_name)

    fields = []
    for field in production.fields:
        field_value = getattr(py_ast_node, field.name)
        asdl_field = RealizedField(field)
        if field.cardinality == 'single' or field.cardinality == 'optional':
            if field_value is not None:  # sometimes it could be 0
                if grammar.is_composite_type(field.type):
                    child_node = python_ast_to_asdl_ast(field_value, grammar)
                    asdl_field.add_value(child_node)
                else:
                    asdl_field.add_value(str(field_value))
        # field with multiple cardinality
        elif field_value is not None:
                if grammar.is_composite_type(field.type):
                    for val in field_value:
                        child_node = python_ast_to_asdl_ast(val, grammar)
                        asdl_field.add_value(child_node)
                else:
                    for val in field_value:
                        asdl_field.add_value(str(val))

        fields.append(asdl_field)

    asdl_node = AbstractSyntaxTree(production, realized_fields=fields)

    return asdl_node


def asdl_ast_to_python_ast(asdl_ast_node, grammar):
    py_node_type = getattr(sys.modules['ast'], asdl_ast_node.production.constructor.name)
    py_ast_node = py_node_type()

    for field in asdl_ast_node.fields:
        # for composite node
        field_value = None
        if grammar.is_composite_type(field.type):
            if field.value and field.cardinality == 'multiple':
                field_value = []
                for val in field.value:
                    node = asdl_ast_to_python_ast(val, grammar)
                    field_value.append(node)
            elif field.value and field.cardinality in ('single', 'optional'):
                field_value = asdl_ast_to_python_ast(field.value, grammar)
        else:
            # for primitive node, note that primitive field may have `None` value
            if field.value is not None:
                if field.type.name == 'object':
                    if '.' in field.value or 'e' in field.value:
                        field_value = float(field.value)
                    elif isint(field.value):
                        field_value = int(field.value)
                    else:
                        raise ValueError('cannot convert [%s] to float or int' % field.value)
                elif field.type.name == 'int':
                    field_value = int(field.value)
                else:
                    field_value = field.value

            # FIXME: hack! if int? is missing value in ImportFrom(identifier? module, alias* names, int? level), fill with 0
            elif field.name == 'level':
                field_value = 0

        # must set unused fields to default value...
        if field_value is None and field.cardinality == 'multiple':
            field_value = list()

        setattr(py_ast_node, field.name, field_value)

    return py_ast_node