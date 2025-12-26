# FertiCO₂-AI

**FertiCO₂-AI** is an open, modular, and fully reproducible computational framework
designed to **classify and predict the CO₂ absorption potential of amino acid ionic
liquids (AAILs)** based exclusively on data generated from **molecular dynamics (MD)
simulations** and **machine learning (ML)**.

The core objective of the project is to develop a **physically grounded artificial
intelligence model** capable of learning structure–property relationships from
first-principles molecular simulations, enabling **data-driven screening and design
of novel AAIL-based systems** for CO₂ capture and sustainable fertilizer technologies.

---

## Scientific Scope and Methodology

The FertiCO₂-AI framework integrates three tightly coupled computational stages:

### 1. Molecular Dynamics Simulations (MD)
Classical molecular dynamics simulations are performed using **GROMACS (open-source)**
to generate physically consistent trajectories of pure and mixed AAIL systems,
with and without explicit CO₂. From these simulations, structural, energetic,
dynamic, and thermodynamic properties are extracted in a fully reproducible manner.

### 2. Physically Ordered Unsupervised Classification
A **K-Means clustering** approach is applied to direct LI–CO₂ interaction descriptors
(e.g., Coulomb and Lennard-Jones energies, hydrogen bonding statistics).  
Clusters are **deterministically reordered based on physical affinity to CO₂**,
yielding **ordinal classes** that reflect increasing absorption strength without
imposing arbitrary labels.

### 3. Supervised Machine Learning and Uncertainty-Aware Prediction
A **Multilayer Perceptron (MLP)** neural network is trained using only **intrinsic
physicochemical descriptors of the AAILs** (composition, density, diffusion,
IL–IL interactions, hydrogen bonding).  
The model predicts the **ordinal CO₂ absorption class**, incorporates **class
imbalance mitigation**, and provides **probabilistic outputs** that enable uncertainty
quantification and physically interpretable confidence metrics.

---

## Repository Contents

This repository includes:

- **MD simulation pipelines** based on GROMACS (input files, protocols, automation)
- **Descriptor extraction workflows** for energetic, structural, and dynamic properties
- **Physically ordered clustering scripts** (Stage 1)
- **Machine learning models and training pipelines** (Stage 2)
- **Cluster-based interpretation and uncertainty analysis tools** (Stage 3)
- **Fully reproducible scripts**, including preprocessing, validation, and inference
- **Documentation and protocols** enabling complete regeneration of the datasets

All components rely exclusively on **open-source software** and standard scientific
libraries.

---

## Reproducibility and Open Science

FertiCO₂-AI was designed from its inception to comply with the principles of
**open science and reproducible research**:

- All code is written in **Python** using open-source libraries
- Molecular simulations can be fully regenerated using **GROMACS (GNU GPL)**
- Models, scalers, and physical mappings are explicitly saved and versioned
- No proprietary software or closed datasets are required
- The full pipeline can be independently audited, replicated, or extended

The project adheres to the **FAIR principles** (Findable, Accessible, Interoperable,
Reusable) and to best practices in **responsible and transparent AI**.

---

## Ethical and Legal Considerations

This project does **not process any personal, sensitive, or identifiable data**.
All datasets consist exclusively of **molecular descriptors and simulation outputs**,
ensuring full compliance with data protection regulations, including the Brazilian
LGPD.

---

## Authors and Contributors

FertiCO₂-AI is developed collaboratively by researchers affiliated with the
**Universidade Federal de Goiás (UFG)**:

- Prof. Dr. Guilherme Colherinhas de Oliveira  
- Prof. Dr. Wesley Bueno Cardoso  
- Prof. Dr. Tertius Lima da Fonseca  
- Prof. Dr. Leonardo Bruno Assis Oliveira  
- Ma. Karinna Mendanha Soares  
- Me. Lucas de Sousa Silva  
- Me. Henrique de Araujo Chagas  

All contributors hold equal intellectual participation in the project.

---

## License

This project is released under the **MIT License**.

Copyright (c) 2025  
**FertiCO₂-AI Project Contributors**  
Universidade Federal de Goiás (UFG)

See the `LICENSE` file for full terms.
