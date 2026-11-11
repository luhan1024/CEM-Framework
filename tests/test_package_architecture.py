"""Structural checks for the top-level responsibility packages."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
import unittest

import core
import enrichment
import estimators_1d
import problems
import solver
import strategies


PACKAGE_EXPORTS = {
    "core": (
        "CostEvent", "WorkLedger", "CorrectionLedger", "CorrectionRecord",
        "PoolAuditRecord", "SolverState", "StateTransaction",
    ),
    "problems": (
        "Operator", "TransposeOperator", "NumpyMatrixOperator", "CallableOperator",
        "LinearPDEProblem", "DiscreteRitzProblem", "MatrixFreeRitzProblem",
        "NonlinearPDEProblem", "DiscreteRitzStepCertificate",
        "certify_discrete_ritz_step", "verify_discrete_ritz_certificate",
    ),
    "enrichment": (
        "Candidate", "WeakLearner", "AnalyticLearner", "LocalBasisLearner",
        "CandidatePool", "PoolCandidateResult", "BestPredictedGainSelector",
        "SelectionResult", "candidate_to_json", "candidate_from_json",
    ),
    "strategies": (
        "Score", "LinearCandidateScorer", "RitzCandidateScorer",
        "NonlinearCandidateScorer", "AcceptancePolicy", "MeritDecreasePolicy",
        "StagewisePolicy", "FullyCorrectivePolicy",
    ),
}

PUBLIC_OWNERS = {
    "SolverState": "core.state",
    "StateTransaction": "core.state",
    "CorrectionLedger": "core.ledger",
    "CorrectionRecord": "core.ledger",
    "PoolAuditRecord": "core.ledger",
    "CostEvent": "core.cost",
    "WorkLedger": "core.cost",
    "Operator": "problems.operators",
    "TransposeOperator": "problems.operators",
    "NumpyMatrixOperator": "problems.operators",
    "CallableOperator": "problems.operators",
    "LinearPDEProblem": "problems.problem",
    "DiscreteRitzProblem": "problems.problem",
    "MatrixFreeRitzProblem": "problems.problem",
    "NonlinearPDEProblem": "problems.problem",
    "DiscreteRitzStepCertificate": "problems.ritz_certificate",
    "certify_discrete_ritz_step": "problems.ritz_certificate",
    "verify_discrete_ritz_certificate": "problems.ritz_certificate",
    "Candidate": "enrichment.learner",
    "WeakLearner": "enrichment.learner",
    "AnalyticLearner": "enrichment.learner",
    "LocalBasisLearner": "enrichment.learner",
    "CandidatePool": "enrichment.pool",
    "PoolCandidateResult": "enrichment.pool",
    "BestPredictedGainSelector": "enrichment.pool",
    "SelectionResult": "enrichment.pool",
    "candidate_to_json": "enrichment.serialization",
    "candidate_from_json": "enrichment.serialization",
    "Score": "strategies.scorer",
    "LinearCandidateScorer": "strategies.scorer",
    "RitzCandidateScorer": "strategies.scorer",
    "NonlinearCandidateScorer": "strategies.scorer",
    "AcceptancePolicy": "strategies.policy",
    "MeritDecreasePolicy": "strategies.policy",
    "StagewisePolicy": "strategies.policy",
    "FullyCorrectivePolicy": "strategies.policy",
    "CEMSolver": "solver",
    "P1Poisson1DData": "estimators_1d",
    "EstimatorRecord": "estimators_1d",
    "EquilibratedFlux1DEstimator": "estimators_1d",
}

ROOT_MODULES = {"solver", "estimators_1d"}
TOP_LEVEL_IMPORTS = {"core", "problems", "enrichment", "strategies", *ROOT_MODULES}
ALLOWED_LAYER_IMPORTS = {
    "core": {"core"},
    "problems": {"problems"},
    "enrichment": {"enrichment"},
    "strategies": {"strategies", "problems"},
    "solver": {"core", "problems", "enrichment", "strategies"},
    "estimators_1d": set(),
}


def _module_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _from_import_targets(node: ast.ImportFrom, current_module: str, is_package_init: bool) -> set[str]:
    if node.level:
        package_parts = (
            current_module.split(".")
            if is_package_init
            else current_module.split(".")[:-1]
        )
        if node.level > 1:
            package_parts = package_parts[: -(node.level - 1)]
        base = ".".join(package_parts)
        target = f"{base}.{node.module}" if node.module else base
        if node.module is None:
            return {f"{target}.{alias.name}" for alias in node.names if alias.name != "*"}
        return {target} if target.split(".", 1)[0] in TOP_LEVEL_IMPORTS else set()
    if node.module and node.module.split(".", 1)[0] in TOP_LEVEL_IMPORTS:
        return {node.module}
    return set()


class PackageArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.source_root = Path(solver.__file__).resolve().parent
        self.repo_root = self.source_root.parent

    def test_source_root_has_only_responsibility_packages_and_coordinators(self):
        expected_packages = set(PACKAGE_EXPORTS)
        actual_packages = {
            path.name for path in self.source_root.iterdir()
            if (path.is_dir() and not path.name.startswith("__")
                and not path.name.endswith(".egg-info"))
        }
        self.assertEqual(actual_packages, expected_packages)
        for package_name in expected_packages:
            self.assertTrue((self.source_root / package_name / "__init__.py").is_file())

        root_modules = {path.stem for path in self.source_root.glob("*.py")}
        self.assertEqual(root_modules, ROOT_MODULES)
        self.assertTrue((self.source_root / "solver.py").is_file())
        self.assertTrue((self.source_root / "estimators_1d.py").is_file())

    def test_responsibility_package_exports_and_public_symbols_are_stable(self):
        package_objects = {
            "core": core,
            "problems": problems,
            "enrichment": enrichment,
            "strategies": strategies,
        }
        for package_name, expected_names in PACKAGE_EXPORTS.items():
            package = package_objects[package_name]
            with self.subTest(package=package_name):
                self.assertEqual(tuple(package.__all__), expected_names)
                namespace = {}
                exec(f"from {package_name} import *", namespace)
                self.assertEqual(
                    {name for name in namespace if name != "__builtins__"},
                    set(expected_names),
                )

        for public_name, owner_name in PUBLIC_OWNERS.items():
            owner = importlib.import_module(owner_name)
            with self.subTest(public_name=public_name):
                if public_name == "CEMSolver":
                    exported = solver.CEMSolver
                elif public_name in {"P1Poisson1DData", "EstimatorRecord", "EquilibratedFlux1DEstimator"}:
                    exported = getattr(estimators_1d, public_name)
                else:
                    package_name = owner_name.split(".", 1)[0]
                    package = package_objects[package_name]
                    exported = getattr(package, public_name)
                self.assertIs(exported, getattr(owner, public_name))

    def test_internal_import_graph_is_acyclic_and_respects_package_boundaries(self):
        graph = {}
        for path in sorted(self.source_root.rglob("*.py")):
            module_name = _module_name(path, self.source_root)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            targets = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    targets.update(_from_import_targets(
                        node, module_name, path.name == "__init__.py"
                    ))
                elif isinstance(node, ast.Import):
                    targets.update(
                        alias.name for alias in node.names
                        if alias.name.split(".", 1)[0] in TOP_LEVEL_IMPORTS
                    )
            graph[module_name] = targets
            source_layer = module_name.split(".", 1)[0]
            for target in targets:
                target_layer = target.split(".", 1)[0]
                self.assertIn(
                    target_layer,
                    ALLOWED_LAYER_IMPORTS[source_layer],
                    f"forbidden dependency: {module_name} -> {target}",
                )

        visiting = set()
        visited = set()

        def visit(module_name):
            if module_name in visiting:
                self.fail(f"internal import cycle reaches {module_name}")
            if module_name in visited:
                return
            visiting.add(module_name)
            for dependency in graph.get(module_name, ()):
                if dependency in graph:
                    visit(dependency)
            visiting.remove(module_name)
            visited.add(module_name)

        for module_name in graph:
            visit(module_name)

    def test_examples_and_tests_import_through_selected_entrypoints(self):
        for directory in (self.repo_root / "tests", self.repo_root / "examples"):
            for path in sorted(directory.rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    roots = set()
                    if isinstance(node, ast.ImportFrom) and node.module:
                        roots.add(node.module.split(".", 1)[0])
                    elif isinstance(node, ast.Import):
                        roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                    for root in roots:
                        if (root not in TOP_LEVEL_IMPORTS
                                and root not in sys.stdlib_module_names
                                and root != "numpy"):
                            self.fail(f"unexpected import root {root!r} in {path}")


if __name__ == "__main__":
    unittest.main()
