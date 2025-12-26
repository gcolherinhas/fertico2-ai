# FertiCO₂-AI

FertiCO₂-AI is an open, modular, and fully reproducible computational framework
designed to classify, predict, and interpret the CO₂ absorption potential of
amino acid ionic liquids (AAILs) using molecular dynamics simulations and
artificial intelligence.

The project integrates physically grounded molecular descriptors extracted
from GROMACS simulations with machine learning models capable of learning
ordinal absorption regimes and quantifying predictive uncertainty.

---

## Project Scope

This repository provides:

- A complete molecular dynamics (MD) simulation pipeline for AAIL and AAIL–CO₂ systems
- Automated extraction of structural, energetic, dynamical, and hydrogen-bond descriptors
- Physically ordered unsupervised classification of CO₂ absorption regimes
- Supervised ordinal neural network models with uncertainty quantification
- Full reproducibility, transparency, and open-science compliance

---

## Main Technical Documentation

The full and authoritative description of the Molecular Dynamics pipeline,
including system preparation, simulation protocols, descriptor extraction, and
physical interpretation, is provided in the following document:

Complete MD Pipeline Documentation (English)  
`FERTICO2AI-AI_PIPELINE_EN.docx`

This document should be considered the primary technical reference for:
- Reproducing the simulations
- Understanding the physical meaning of descriptors
- Interpreting the AI predictions
- Extending or adapting the workflow

---

## Artificial Intelligence Pipeline

The AI pipeline (Stages 1–3), including clustering, neural network training,
ordinal analysis, and uncertainty quantification, is implemented directly in
the repository source code and documented inline.

---

## Reproducibility and Open Science

- All scripts are fully reproducible
- Models, scalers, and physical mappings are explicitly saved
- Descriptor definitions are physically interpretable
- The project follows principles of open science, responsible AI, and
  methodological transparency

---

## License

This project is released under the MIT License.  
See the `LICENSE` file for details.

