# LTC_BTE
End-to-end cost of lattice thermal conductivity prediction: machine-learned potentials versus DFT.
CSCE 585 · Machine Learning Systems · Fall 2026 · University of South Carolina
## What this is
A benchmark study comparing two independent ways to reduce the cost of computing lattice thermal conductivity (κ_L): replacing DFT force evaluations with a machine-learned interatomic potential (MLIP), and reducing the number of evaluations required through regression-based force-constant extraction. We measure wall-clock time, compute consumed, and κ_L accuracy across all four combinations, on a common axis nobody has yet reported.
The full write-up is in proposal.md.
## Team
Ramm Chevva (PhD, CSE) — project lead, DFT references, integration
Thrinadh Nimmagadda (MS, CS) — systems and throughput
Divyashanu Swain (MS, CS) — evaluation and reproducibility
## Status
Proposal	Submitted Sep 4, 2026
Environment	Not yet pinned
Baseline pipeline	Not yet running
Final report	Due week 15
Progress is tracked in GitHub Issues, one per research question.
## Repository layout
Planned structure. Directories appear as work begins.
proposal.md — written project proposal
env/ — pinned conda environment and model checkpoints
harness/ — benchmark code: profiler, batching, sharding, precision
configs/ — one YAML per swept configuration
scripts/ — sweep drivers, SLURM templates, reproduction scripts
results/raw/ — one JSON per run with environment header
results/processed/ — aggregated CSVs behind every figure
analysis/ — one script per figure
figures/ — figures and tables (PDF and PNG)
docs/ — plain-language project primer
report/ — final report and presentation
## Reproducing a result
Once the harness is available, one principal result will regenerate from a fresh clone with bash scripts/reproduce_figure1.sh. The script reports the relative per-stage timing breakdown; absolute timings differ across hardware. Expected runtime and tolerance are documented alongside the script.
## Dependencies
Pinned in env/environment.yml. Key packages: phono3py 4.4.0, phonopy 4.4.0, hiphive, NequIP-OAM-L (pretrained, checkpoint hash pinned in env/models.lock), ASE, pymatgen, VASP. External code is referenced by release tag or commit hash, never by branch.
## Use of AI assistants
An LLM assistant (Claude) was used during proposal preparation for literature search, drafting, and running preliminary calculations. See the Reproducibility Plan in proposal.md for details. Materially AI-assisted components will be noted in source file headers during implementation, consistent with the course academic-integrity policy.
## License
MIT. See LICENSE.
