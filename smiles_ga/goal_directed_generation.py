from __future__ import annotations

import argparse
import copy
import json
import os
from time import time
from typing import TYPE_CHECKING

import nltk
import numpy as np
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.utils.chemistry import canonicalize, canonicalize_list
from guacamol.utils.helpers import setup_default_logger
from guacamol.utils.parallelize import parallelize
from rdkit import rdBase

from . import cfg_util, smiles_grammar
from .cfg_util import Molecule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from guacamol.scoring_function import ScoringFunction
    from numpy.typing import NDArray

rdBase.DisableLog("rdApp.error")
GCFG = smiles_grammar.GCFG


def cfg_to_gene(prod_rules, max_len: int = -1) -> list[int]:
    gene: list[int] = []
    for r in prod_rules:
        lhs = GCFG.productions()[r].lhs()
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions()) if rule.lhs() == lhs]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [np.random.randint(0, 256) for _ in range(max_len - len(gene))]
    return gene


def gene_to_cfg(gene: list[int]) -> list[int]:
    prod_rules = []
    stack = [GCFG.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except Exception:  # noqa: BLE001
            break
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions()) if rule.lhs() == lhs]
        rule = possible_rules[g % len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(
            lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != "None"),  # noqa: E721
            smiles_grammar.GCFG.productions()[rule].rhs(),
        )
        stack.extend(list(rhs)[::-1])
    return prod_rules


def select_parent(
    population: list[tuple[int, str, NDArray[np.int32]]], tournament_size: int = 3
) -> tuple[int, str, NDArray[np.int32]]:
    idx = np.random.randint(len(population), size=tournament_size)
    best = population[idx[0]]
    for i in idx[1:]:
        if population[i][0] > best[0]:
            best = population[i]
    return best


def mutation(gene: list[int]) -> list[int]:
    idx = np.random.choice(len(gene))
    gene_mutant = copy.deepcopy(gene)
    gene_mutant[idx] = np.random.randint(0, 256)
    return gene_mutant


def deduplicate(
    population: list[tuple[float, str, NDArray[np.int32]]],
) -> list[tuple[float, str, NDArray[np.int32]]]:
    unique_smiles: set[str] = set()
    unique_population: list[tuple[float, str, int]] = []
    for item in population:
        score, smiles, gene = item
        if smiles not in unique_smiles:
            unique_population.append(item)
        unique_smiles.add(smiles)
    return unique_population


def mutate(p_gene: list[int], scoring_function: ScoringFunction) -> Molecule:
    c_gene = mutation(p_gene)
    c_smiles = canonicalize(cfg_util.decode(gene_to_cfg(c_gene)))
    c_score = scoring_function.score(c_smiles)
    return Molecule(c_score, c_smiles, c_gene)


class ChemGEGenerator(GoalDirectedGenerator):
    def __init__(
        self,
        smi_file: str,
        population_size: int,
        n_mutations: int,
        gene_size: int,
        generations: int,
        random_start: bool = False,
        patience: int = 5,
    ) -> None:
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.n_mutations = n_mutations
        self.gene_size = gene_size
        self.generations = generations
        self.random_start = random_start
        self.patience = patience

    def load_smiles_from_file(self, smi_file: str) -> list[str]:
        with open(smi_file) as f:
            smiles = [s.strip() for s in f]
        canonicals = canonicalize_list(smiles)
        canonicals = [s for s in canonicals if s is not None]
        if len(canonicals) < len(smiles):
            print(f"{len(smiles) - len(canonicals)} invalid SMILES strings found.")
        return canonicals

    def top_k(self, smiles: list[str], scoring_function: ScoringFunction, k: int) -> list[str]:
        scores = parallelize(
            scoring_function.score, [(s,) for s in smiles], desc="Scoring", verbose=1
        )
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: list[str] | None = None,
    ) -> list[str]:
        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(
                f"Benchmark requested more molecules than expected: new population is {number_molecules}",
            )

        # fetch initial population?
        if starting_population is None:
            print("selecting initial population...")
            init_size = self.population_size + self.n_mutations
            all_smiles = copy.deepcopy(self.all_smiles)
            if self.random_start:
                starting_population = np.random.choice(all_smiles, init_size)
            else:
                starting_population = self.top_k(all_smiles, scoring_function, init_size)

        # The smiles GA cannot deal with '%' in SMILES strings (used for two-digit ring numbers).
        starting_population = [smiles for smiles in starting_population if "%" not in smiles]

        # calculate initial genes
        initial_genes = [
            cfg_to_gene(cfg_util.encode(s), max_len=self.gene_size) for s in starting_population
        ]

        # score initial population
        initial_scores = scoring_function.score_list(starting_population)
        population = [Molecule(*m) for m in zip(initial_scores, starting_population, initial_genes)]
        population = sorted(population, key=lambda x: x.score, reverse=True)[: self.population_size]
        population_scores = [p.score for p in population]

        # evolution: go go go!!
        t0 = time()

        patience = 0

        for generation in range(self.generations):
            old_scores = population_scores
            # select random genes
            all_genes = [molecule.genes for molecule in population]
            choice_indices: Sequence[int] = np.random.choice(
                len(all_genes), self.n_mutations, replace=True
            )
            genes_to_mutate = [all_genes[i] for i in choice_indices]

            # evolve genes
            new_population = parallelize(
                mutate,
                [(g, scoring_function) for g in genes_to_mutate],
                desc="Mutating",
                leave=False,
                verbose=1,
            )

            # join and dedup
            population += new_population
            population = deduplicate(population)

            # survival of the fittest
            population = sorted(population, key=lambda x: x.score, reverse=True)[
                : self.population_size
            ]

            # stats
            gen_time = time() - t0
            mol_sec = (self.population_size + self.n_mutations) / gen_time
            t0 = time()

            population_scores = [p.score for p in population]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f"Failed to progress: {patience}")
                if patience >= self.patience:
                    print("No more patience, bailing...")
                    break
            else:
                patience = 0

            print(
                f"{generation} | "
                f"max: {np.max(population_scores):.3f} | "
                f"avg: {np.mean(population_scores):.3f} | "
                f"min: {np.min(population_scores):.3f} | "
                f"std: {np.std(population_scores):.3f} | "
                f"{gen_time:.2f} sec/gen | "
                f"{mol_sec:.2f} mol/sec",
            )

        # finally
        return [molecule.smiles for molecule in population[:number_molecules]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--n_mutations", type=int, default=200)
    parser.add_argument("--gene_size", type=int, default=300)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--suite", default="v2")

    args = parser.parse_args()

    np.random.seed(args.seed)

    setup_default_logger()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    # save command line args
    with open(os.path.join(args.output_dir, "goal_directed_params.json"), "w") as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)

    optimiser = ChemGEGenerator(
        smi_file=args.smiles_file,
        population_size=args.population_size,
        n_mutations=args.n_mutations,
        gene_size=args.gene_size,
        generations=args.generations,
        random_start=args.random_start,
        patience=args.patience,
    )

    json_file_path = os.path.join(args.output_dir, "goal_directed_results.json")
    assess_goal_directed_generation(
        optimiser,
        json_output_file=json_file_path,
        benchmark_version=args.suite,
    )


if __name__ == "__main__":
    main()
