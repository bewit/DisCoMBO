"""
Rejection sampling from a probabilistic circuit using p-value filtering.

Given a learned PC and conditioning evidence, sample a large pool of candidates
from the conditioned circuit, compute p-values on the original (unconditioned)
circuit using Fisher's method, and accept candidates whose p-value falls within
a specified range.
"""

import torch

from .structural_manipulation import structural_conditioning
from .pvalue_combination_functions import (
    TestStrategy, ArithmeticMean, FishersMethod
)
from .nodes import SumNode, ProductNode, LeafNode


def rejection_sample(
    circuit,
    evidence: torch.Tensor,
    n_samples: int,
    test_strategy: TestStrategy,
    num_candidates: int = 10000,
    p_min: float = 0.0001,
    p_max: float = 0.1,
    max_attempts: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rejection sampling from a conditioned circuit, filtered by p-values
    on the unconditioned circuit.

    1. Condition ``circuit`` on ``evidence`` via structural_conditioning.
    2. Sample ``num_candidates`` from the conditioned circuit.
    3. Compute p-values on the ORIGINAL (unconditioned) ``circuit``.
    4. Accept candidates with p-value in [p_min, p_max].
    5. Repeat until ``n_samples`` accepted or ``max_attempts`` reached.
    6. Return accepted candidates sorted by p-value ascending (most novel first).

    Args:
        circuit: Root node of the unconditioned PC.
        evidence: Conditioning evidence, shape (1, D). NaN = unobserved,
            non-NaN = observed/conditioned.
        n_samples: Number of samples to return.
        test_strategy: TestStrategy for p-value computation (should use
            FishersMethod at product nodes).
        num_candidates: Candidates to draw per attempt.
        p_min: Lower bound of p-value acceptance range.
        p_max: Upper bound of p-value acceptance range.
        max_attempts: Max sampling rounds before fallback.

    Returns:
        selected: Tensor of shape (n_selected, D) with full rows (conditioned
            variables filled in, unobserved variables sampled and filtered).
            n_selected = min(n_samples, total_accepted).
        pvalues: Tensor of shape (n_selected,) with p-values of selected rows.
    """
    n_vars = len(circuit.scope)

    # Identify which columns are conditioned (non-NaN in evidence)
    conditioned_cols = set()
    for i in range(evidence.shape[1]):
        if not torch.isnan(evidence[0, i]):
            conditioned_cols.add(i)

    # Condition circuit on evidence
    try:
        conditioned_mc = structural_conditioning(circuit, evidence)
    except (ValueError, Exception):
        conditioned_mc = None

    accepted_samples = []
    accepted_pvalues = []
    last_full_samples = None

    for attempt in range(max_attempts):
        # Sample from conditioned circuit
        if conditioned_mc is not None:
            raw_samples = conditioned_mc.sample(num_candidates)
            # Map conditioned scope back to full variable layout
            full_samples = torch.full(
                (num_candidates, n_vars), float('nan'), dtype=torch.float64)
            for j, scope_idx in enumerate(conditioned_mc.scope):
                full_samples[:, scope_idx] = raw_samples[:, j]
            # Fill conditioned variables from evidence
            for col in conditioned_cols:
                full_samples[:, col] = evidence[0, col]
        else:
            # Fallback: ancestral conditional sampling (row-by-row)
            full_samples = evidence.repeat(num_candidates, 1).clone()
            for i in range(num_candidates):
                _ancestral_sample_row(circuit, full_samples[i])

        last_full_samples = full_samples

        # Compute p-values on UNCONDITIONED circuit.
        # NaN out conditioned columns so p-value reflects only sampled dims.
        pval_input = full_samples.clone()
        for col in conditioned_cols:
            pval_input[:, col] = float('nan')

        with torch.no_grad():
            circuit.log_pdf(pval_input, write_cache=True)
            pvalues = circuit.test_membership_distribution(
                pval_input, test_strategy)

        pvalues_flat = pvalues.squeeze()
        if pvalues_flat.dim() == 0:
            pvalues_flat = pvalues_flat.unsqueeze(0)

        # Accept candidates in [p_min, p_max]
        mask = (pvalues_flat >= p_min) & (pvalues_flat <= p_max)
        if mask.any():
            accepted_samples.append(full_samples[mask])
            accepted_pvalues.append(pvalues_flat[mask])

        total = sum(a.shape[0] for a in accepted_samples)
        if total >= n_samples:
            break

    if accepted_samples:
        all_samples = torch.cat(accepted_samples, dim=0)
        all_pvalues = torch.cat(accepted_pvalues, dim=0)
        # Sort by p-value ascending (most novel first)
        idx = torch.argsort(all_pvalues)[:n_samples]
        return all_samples[idx], all_pvalues[idx]
    else:
        # Fallback: return unfiltered conditioned samples
        n_ret = min(n_samples, last_full_samples.shape[0])
        return last_full_samples[:n_ret], torch.zeros(n_ret)


def _ancestral_sample_row(node, row):
    """Recursive ancestral sampling for a single evidence row (fallback path)."""
    if isinstance(node, LeafNode):
        idx = node.scope[0]
        if torch.isnan(row[idx]):
            dist = node._get_distribution()
            row[idx] = dist.sample().squeeze().to(dtype=row.dtype)
    elif isinstance(node, ProductNode):
        for child in node.children_circuit:
            _ancestral_sample_row(child, row)
    elif isinstance(node, SumNode):
        with torch.no_grad():
            child_lls = []
            for child in node.children_circuit:
                ll = child.log_pdf(row.unsqueeze(0))
                child_lls.append(ll.squeeze().item())
            child_lls = torch.tensor(child_lls)
            log_posterior = node.log_weights + child_lls
            if torch.all(torch.isinf(log_posterior) | torch.isnan(log_posterior)):
                posterior = torch.softmax(node.log_weights, dim=0)
            else:
                log_posterior = torch.where(
                    torch.isnan(log_posterior),
                    torch.tensor(float('-inf')),
                    log_posterior
                )
                posterior = torch.softmax(log_posterior, dim=0)
        child_idx = torch.multinomial(posterior, 1).item()
        _ancestral_sample_row(node.children_circuit[child_idx], row)
