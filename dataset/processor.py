import astor
from common.registerable import Registrable
from .dataset import asdl_ast_to_python_ast
from .utils import canonicalize_intent, tokenize_intent, decanonicalize_code


class ExampleProcessor(object):
    """
    Process a raw input utterance using domain-specific procedures (e.g., stemming),
    and post-process a generated hypothesis to the final form
    """
    def pre_process_utterance(self, utterance):
        raise NotImplementedError

    def post_process_hypothesis(self, hyp, meta_info, **kwargs):
        raise NotImplementedError


def get_example_processor_cls(dataset):
    if dataset == 'conala':
        from datasets.conala.example_processor import ConalaExampleProcessor
        return ConalaExampleProcessor
    else:
        raise RuntimeError()

@Registrable.register('conala_example_processor')
class ConalaExampleProcessor(ExampleProcessor):
    def __init__(self, transition_system):
        self.transition_system = transition_system

    def pre_process_utterance(self, utterance):
        canonical_intent, slot_map = canonicalize_intent(utterance)
        intent_tokens = tokenize_intent(canonical_intent)

        return intent_tokens, slot_map

    def post_process_hypothesis(self, hyp, meta_info, utterance=None):
        """traverse the AST and replace slot ids with original strings"""
        hyp_ast = asdl_ast_to_python_ast(hyp.tree, self.transition_system.grammar)
        code_from_hyp = astor.to_source(hyp_ast).strip()
        hyp.code = decanonicalize_code(code_from_hyp, meta_info)
