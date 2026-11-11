# Fixed-discrete Ritz step certificate (prototype)

`problems.certify_discrete_ritz_step(problem, u, phi, ...)` is an independent
verifier for one finite-dimensional quadratic step. It requires an existing
`DiscreteRitzProblem` with `problem.feasible is None`; from its stored `K` and
source vector and the supplied state/direction, it interprets finite
float32/float64 values as exact dyadic rationals and recomputes

\[
J(u)=\tfrac12 u^T K u-f^T u,\qquad
c=(f-Ku)^T\phi,\qquad q=\phi^T K\phi.
\]

It requires exact rational symmetry and positive definiteness of `K`, `q>0`,
nonnegative radii, `q_hat > epsilon`, and exact containment
`|c-c_hat| <= delta`, `|q-q_hat| <= epsilon`. The SPD check uses exact
rational LDL^T pivots. It then forms

\[
s=(|\widehat c|-\delta)_+,
\quad \alpha=\operatorname{sgn}(\widehat c)\frac{s}{\widehat q+\epsilon},
\quad \underline\Delta=\frac{s^2}{2(\widehat q+\epsilon)}.
\]

The trial `u + alpha*phi`, energy gain, and comparison with the lower bound
are all evaluated using `fractions.Fraction`. The exact-rational LDL^T check is
limited to dimension 32, at most 6,000 LDL summation products, and 16,384 bits
per intermediate Fraction numerator/denominator; larger inputs return
`unsupported_resource_limit` before hashing or factorization where the input
shape exposes the excess. This explicit cap contains the O(n^3) factorization
work and rational numerator/denominator growth. A positive lower bound returns
`verified_discrete_lower_bound`; a zero lower bound returns
`zero_certified_gain`; malformed inputs, failed enclosures, unsupported
constraints, or failed model checks return `invalid_input` with a reason.

The first version supports only an unconstrained trial space. It reads the
problem's actual `feasible` property, so callers cannot bypass the constraint
gate by omitting the callback argument. With no
constraints, every exact trial is feasible by definition; the record's
`feasibility_basis="unconstrained_domain"` means feasibility follows from the
unconstrained domain, not from callback evaluation. A non-`None` Python feasibility callback is rejected as
`unsupported_feasibility_callback`; a callback result is not a proof protocol.
No state is updated. Enclosure
radii are checked as declared dyadic values, but their provenance or
mathematical derivation is not verified here. The module does not certify
continuous PDE integration/error, candidate-search quality, weak-greedy
selection, or solver convergence, and makes no novelty claim. Its exactness
claim is limited to the finite IEEE-754 inputs and rational operations in this
verifier. This Python prototype is not a machine-certified proof artifact.

`to_json`/`from_json` provide deterministic records, not proofs by themselves.
Use `verify_discrete_ritz_certificate(record, problem, u, phi, enclosures...)`
to recompute every field against the supplied problem/inputs and independently compare the claimed gain
with the exact direct difference `J(u)-J(trial)`. A deserialized record whose
fields or input hash do not match is rejected by that verifier.

## Evidence boundary

The energy projection/fully-corrective identity in research-plan §4.2 and the
P1 scalar lower-bound algebra in §5 are standard exact-Ritz/quadratic
identities; this software does not claim them as new theory. Ern and Vohralík's
equilibrated-flux a posteriori estimates concern deriving reliable error
bounds for finite-element discretizations; this prototype does not implement
such bound construction. Temlyakov's weak-greedy theory and Xu–Xu's randomized
weak-greedy results concern selection and convergence guarantees; this
prototype has no dictionary search, selector, or convergence claim. The only
software object checked here is a fixed discrete SPD quadratic form, declared
dyadic enclosures, and the exact rational step/trial.
