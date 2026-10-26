# Cantilever beam vibration sensitivity study

An original, self-contained mechanical engineering research-code example. It uses a small Euler–Bernoulli finite-element model to study how reducing flexural stiffness in one beam element changes the first five natural frequencies of a cantilever.

The example uses a synthetic uniform steel beam: 1.0 m long, 30 mm wide, and 10 mm deep, with 24 elements. The assumed material properties are Young's modulus 210 GPa and density 7850 kg/m³. A local element stiffness reduction from 0% to 50% represents a simplified damage scenario at the beam midpoint. No external files, prior project materials, or measured data are used.

## Run

Install the dependency and run:

```bash
python -m pip install -r requirements.txt
python beam_crack_modal_study.py
```

The script writes `results/frequency_sweep.csv`, `results/study_summary.json`, and `results/frequency_shift.svg`.

## Method and limits

Each two-node beam element has transverse-displacement and rotation degrees of freedom, an Euler–Bernoulli stiffness matrix, and a consistent mass matrix. The root displacement and rotation are fixed. The generalized eigenproblem `K φ = ω² M φ` gives the natural frequencies. Damage is represented by multiplying one element's flexural rigidity by `1 − loss_fraction`; mass stays unchanged.

This is a reproducible sensitivity demonstration, not a validated crack-detection method. It omits crack breathing, damping, shear deformation, rotary inertia, uncertainty in material properties, and measurement noise. The frequency shifts should not be interpreted as field thresholds.
