import pandas as pd


def compute_signature(exps):
    plus_genes = pd.read_csv('TEAD_plus.txt', sep='\t', header=None)[0].values
    minus_genes = pd.read_csv('TEAD_minus.txt', sep='\t', header=None)[0].values
    pgenes = exps.columns.intersection(plus_genes)
    mgenes = exps.columns.intersection(minus_genes)
    fractional_ranks = exps.loc[:, pgenes.union(mgenes)].transform(lambda x : x.rank() / len(x), axis=1)
    score = fractional_ranks.loc[:, pgenes].mean(axis=1) - \
        fractional_ranks.loc[:, mgenes].mean(axis=1)
    return score
