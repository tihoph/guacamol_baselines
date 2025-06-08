import nltk
import numpy as np

from . import smiles_grammar


def get_smiles_tokenizer(cfg):
    long_tokens = [a for a in cfg._lexical_index if len(a) > 1]  # noqa: SLF001
    # there are currently 6 double letter entities in the grammar
    # these are their replacement, with no particular meaning
    # they need to be ascii and not part of the SMILES symbol vocabulary
    replacements = ["!", "?", ".", ",", ";", "$"]
    assert len(long_tokens) == len(replacements)
    for token in replacements:
        assert token not in cfg._lexical_index  # noqa: SLF001

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except Exception:  # noqa: BLE001,PERF203
                tokens.append(token)
        return tokens

    return tokenize


def encode(smiles):
    GCFG = smiles_grammar.GCFG
    tokenize = get_smiles_tokenizer(GCFG)
    tokens = tokenize(smiles)
    parser = nltk.ChartParser(GCFG)
    parse_tree = parser.parse(tokens).__next__()
    productions_seq = parse_tree.productions()
    productions = GCFG.productions()
    prod_map = {prod: ix for ix, prod in enumerate(productions)}
    return np.array([prod_map[prod] for prod in productions_seq], dtype=int)


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == "Nothing":
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1 :]
                break
    try:
        return "".join(seq)
    except Exception:  # noqa: BLE001
        return ""


def decode(rule):
    productions = smiles_grammar.GCFG.productions()
    prod_seq = [productions[i] for i in rule]
    return prods_to_eq(prod_seq)
