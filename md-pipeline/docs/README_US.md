# FertiCO₂-AI — Molecular Dynamics Pipeline

---

## English Version

### FertiCO₂-AI — Molecular Dynamics (MD) Pipeline

FertiCO₂-AI is an open, modular, and fully reproducible computational framework  
designed to study and predict the CO₂ absorption potential of amino acid ionic liquids 
(AAILs).

This repository focuses exclusively on the Molecular Dynamics (MD) pipeline used to 
generate  
the physicochemical data that feed the project’s Artificial Intelligence models.

---

## Purpose of the MD Pipeline

The Molecular Dynamics pipeline is designed to:

- Simulate pure AAILs and AAIL–CO₂ systems
- Obtain physicochemical, structural, dynamical, and energetic descriptors
- Ensure physical consistency, reproducibility, and traceability
- Provide reliable data for:
  - Unsupervised physical classification
  - Training of supervised AI models

---

## Official Technical Documentation

This repository is accompanied by a comprehensive technical document, which MUST be 
consulted before running the pipeline:

### Main document

- `docs/FERTICO2AI-MD_PIPELINE_BR.pdf`

This document provides a detailed description of:

- Construction of initial configurations using Packmol
- Topology definition and force field specification (CHARMM36)
- Energy minimization and equilibration protocols
- Automated insertion of CO₂ molecules
- Production simulations
- Descriptor extraction using GROMACS commands
- Best practices for reproducibility

This README serves as an operational guide.  
The technical document is the complete methodological reference.

---

## Molecular Dynamics Pipeline Stages

### 1 Initial System Construction

- Construction of pure AAILs or AAIL mixtures
- Guaranteed electrical neutrality
- Typical system sizes:
  - 300 ionic pairs (pure AAIL)
  - Mixtures ranging from 50–250 to 250–50 ionic pairs

---

### 2 Topology and Force Field

- Force field: CHARMM36
- Main topology file: `topol.top`
- Explicit definition of:
  - Atom types
  - Partial charges
  - Bonded and non-bonded interactions

---

### 3 Minimization and Equilibration of Pure AAIL

- Energy minimization
- NVT equilibration
- NPT equilibration
- Monitoring of:
  - Energy
  - Temperature
  - Pressure
  - Density

---

### 4 Automated CO₂ Insertion

After equilibration of the pure AAIL:

- Expansion of the simulation box along the Z-axis
- Explicit insertion of 300 CO₂ molecules
- Preservation of the original ionic liquid configuration

---

### 5 AAIL–CO₂ Production Simulations

- Long production runs in NPT or NVT ensembles
- Generation of trajectory files (`.xtc` or `.trr`)
- Generation of energy files (`.edr`)

---

### 6 Descriptor Extraction

The main extracted descriptors include:

- Coulomb and Lennard-Jones interaction energies (AAIL–CO₂)
- Density
- Diffusion coefficient (MSD)
- Hydrogen bonds:
  - Number
  - Lifetime
  - Effective dissociation energy

---

##  License

This project is released under the MIT License.

---

##  Citation

If you use this pipeline in academic work, please cite the associated publication  
and acknowledge the FertiCO₂-AI project.

---

