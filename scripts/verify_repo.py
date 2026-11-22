#!/usr/bin/env python3
"""Fast structural and Git-hygiene verification gate for CD-Agent.

Run before committing or submitting Slurm jobs:
    python -B scripts/verify_repo.py
"""
from __future__ import annotations

import argparse
import ast
import importlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FORBIDDEN_TRACKED_EXTS = frozenset({
    '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npz', '.npy',
    '.pth', '.pt', '.ckpt', '.safetensors', '.so', '.out', '.err', '.log', '.zip',
})
FORBIDDEN_TRACKED_PREFIXES = ('outputs/', '.runtime/', 'third_party/', 'models/', 'datasets/')
ALLOWED_TRACKED_UNDER_PREFIX = frozenset({
    'outputs/README.md',
    'models/README.md',
    'datasets/README.md',
})

# Minimal allowlist of bootstrap/env-fallback files permitted to mention /guisongxia01
ABS_PATH_ALLOWLIST = frozenset({
    'src/cd_agent/paths.py',
    'src/cd_agent/_compat/common.py',
    'src/cd_agent/_compat/sam3_tools.py',
    'src/cd_agent/analysis/second_labels.py',
    'experiments/verifier_init_masks/initial_masks/verifier_v1_dataset.py',
    'experiments/verifier_init_masks/initial_masks/verifier_v1_ovcd_methods.py',
    'experiments/verifier_init_masks/initial_masks/verifier_v1_pilot_generate.py',
})

CORE_MODULES = (
    'cd_agent',
    'cd_agent.paths',
    'cd_agent.verify',
    'cd_agent.evidence',
    'cd_agent.diagnostic',
    'cd_agent.verifier',
    'cd_agent.state',
    'cd_agent.tasks',
    'cd_agent.evaluation',
    'cd_agent.planner',
    'cd_agent.recorder',
    'cd_agent.backends',
    'cd_agent.tools',
    'cd_agent.loop_json',
    'cd_agent.t32_dev32',
)


def check_single_package() -> list[str]:
    src_dirs = sorted(
        p.name for p in (REPO / 'src').iterdir()
        if p.is_dir() and not p.name.startswith('__') and not p.name.endswith('.egg-info')
    )
    if src_dirs != ['cd_agent']:
        return [f'Expected single package [cd_agent] in src/, found: {src_dirs}']
    return []


def check_symlinks() -> list[str]:
    errors: list[str] = []
    for folder in ('models', 'datasets'):
        base = REPO / folder
        if not base.is_dir():
            errors.append(f'Missing directory: {folder}/')
            continue
        links = [p for p in base.iterdir() if p.is_symlink()]
        if not links:
            errors.append(f'No symlinks found under {folder}/')
        for link in sorted(links):
            if not link.exists():
                errors.append(f'Broken symlink: {folder}/{link.name} -> {link.readlink()}')
    return errors


def check_git_hygiene() -> list[str]:
    errors: list[str] = []
    gitignore = REPO / '.gitignore'
    if not gitignore.is_file():
        return ['Missing .gitignore at repository root']
    gi_lines = {ln.strip() for ln in gitignore.read_text(encoding='utf-8').splitlines()}
    for required in ('outputs/*', '.runtime/', 'third_party/', 'models/*', 'datasets/*'):
        if required not in gi_lines:
            errors.append(f'.gitignore missing required guard: {required}')

    tracked = subprocess.check_output(['git', '-C', str(REPO), 'ls-files'], text=True).splitlines()
    staged = subprocess.check_output(['git', '-C', str(REPO), 'diff', '--cached', '--name-only'], text=True).splitlines()
    for rel in sorted(set(tracked) | set(staged)):
        if rel in ALLOWED_TRACKED_UNDER_PREFIX:
            continue
        if rel.startswith(FORBIDDEN_TRACKED_PREFIXES):
            errors.append(f'Forbidden tracked/staged path under ignored dir: {rel}')
        elif Path(rel).suffix.lower() in FORBIDDEN_TRACKED_EXTS:
            errors.append(f'Forbidden binary/artifact extension tracked/staged: {rel}')
    return errors


def check_paths_and_syntax() -> list[str]:
    errors: list[str] = []
    py_files = [
        p for d in ('src', 'experiments', 'scripts', 'tests')
        for p in (REPO / d).rglob('*.py')
        if '__pycache__' not in p.parts
    ]
    for path in sorted(py_files):
        rel = str(path.relative_to(REPO))
        text = path.read_text(encoding='utf-8')
        try:
            ast.parse(text, filename=rel)
        except SyntaxError as exc:
            errors.append(f'SyntaxError in {rel}:{exc.lineno}: {exc.msg}')
            continue
        if rel.startswith(('src/', 'experiments/')):
            if 'wyh/CD-Agent' in text or 'exp6_staging' in text:
                errors.append(f'Legacy workspace reference in {rel}')
            if '/guisongxia01/' in text and rel not in ABS_PATH_ALLOWLIST:
                errors.append(f'Hardcoded /guisongxia01/ path in {rel} (use cd_agent.paths instead)')
    return errors


def check_imports() -> list[str]:
    sys.dont_write_bytecode = True
    src = str(REPO / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    errors: list[str] = []
    for mod in CORE_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            errors.append(f'Failed to import {mod}: {type(exc).__name__}: {exc}')
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-imports', action='store_true', help='Skip runtime Python import check')
    args = parser.parse_args()

    checks = [
        ('single_package', check_single_package),
        ('symlinks', check_symlinks),
        ('git_hygiene', check_git_hygiene),
        ('paths_and_syntax', check_paths_and_syntax),
    ]
    if not args.skip_imports:
        checks.append(('core_imports', check_imports))

    all_errors: list[str] = []
    for label, fn in checks:
        errs = fn()
        status = 'PASS' if not errs else f'FAIL ({len(errs)})'
        print(f'[{status}] {label}')
        for err in errs:
            print(f'  - {err}')
        all_errors.extend(errs)

    if all_errors:
        raise SystemExit(1)
    print('REPO_VERIFY_PASSED')


if __name__ == '__main__':
    main()
