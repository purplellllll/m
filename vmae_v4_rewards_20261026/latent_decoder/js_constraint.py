"""Per-sequence augmented Lagrangian for the existing mutable-site JS limit."""
import math

import torch


def mutable_js(probabilities, reference, active):
    middle = .5 * (probabilities + reference)
    js = .5 * ((probabilities * (probabilities.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(-1)
               + (reference * (reference.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(-1))
    return (js * active).sum(-1) / active.sum(-1).clamp_min(1)


class JSConstraint:
    def __init__(self, reference, active, limit, rho):
        if not math.isfinite(limit) or not math.isfinite(rho) or limit <= 0 or rho <= 0:
            raise ValueError('JS limit and augmented-Lagrangian rho must be finite and positive')
        if not bool((active.sum(-1) > 0).all()):
            raise ValueError('JS constraint requires mutable positions')
        self.reference = reference.detach()
        self.active = active
        self.limit, self.rho = limit, rho
        self.dual = reference.new_zeros(len(reference))

    def penalty(self, probabilities):
        self.js_before = mutable_js(probabilities, self.reference, self.active)
        violation = self.js_before / self.limit - 1
        self.dual_before = self.dual.clone()
        self.penalty_before = ((self.dual + self.rho * violation).relu().square() - self.dual.square()) / (2 * self.rho)
        return self.penalty_before

    @torch.no_grad()
    def update(self, probabilities):
        js = mutable_js(probabilities, self.reference, self.active)
        self.dual = (self.dual + self.rho * (js / self.limit - 1)).clamp_min(0)
        if not bool(torch.isfinite(self.dual).all()):
            raise FloatingPointError('Nonfinite JS multiplier')

    def metrics(self, index):
        return dict(js_dual_before=float(self.dual_before[index]), js_dual_after=float(self.dual[index]),
                    js_before_update=float(self.js_before[index].detach()),
                    js_optimization_penalty=float(self.penalty_before[index].detach()))
