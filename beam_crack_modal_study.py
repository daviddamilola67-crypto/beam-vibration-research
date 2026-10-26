#!/usr/bin/env python3
"""Synthetic Euler-Bernoulli beam modal sensitivity study.

Localized stiffness reduction is a simplified damage proxy, not a calibrated
crack model. Inputs are illustrative, not measured material properties.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
LENGTH, WIDTH, HEIGHT = 1.0, 0.03, 0.01
YOUNG, DENSITY = 210e9, 7850.0
ELEMENTS, SOFT_ELEMENT = 24, 11
LOSSES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def element_matrices(length, stiffness_factor):
    area = WIDTH * HEIGHT
    inertia = WIDTH * HEIGHT**3 / 12.0
    ell = length
    k = YOUNG * inertia * stiffness_factor / ell**3 * np.array([
        [12, 6*ell, -12, 6*ell], [6*ell, 4*ell**2, -6*ell, 2*ell**2],
        [-12, -6*ell, 12, -6*ell], [6*ell, 2*ell**2, -6*ell, 4*ell**2]])
    m = DENSITY * area * ell / 420.0 * np.array([
        [156, 22*ell, 54, -13*ell], [22*ell, 4*ell**2, 13*ell, -3*ell**2],
        [54, 13*ell, 156, -22*ell], [-13*ell, -3*ell**2, -22*ell, 4*ell**2]])
    return k, m


def assemble(loss):
    size = 2 * (ELEMENTS + 1)
    stiffness, mass = np.zeros((size, size)), np.zeros((size, size))
    ell = LENGTH / ELEMENTS
    for e in range(ELEMENTS):
        factor = 1.0 - loss if e == SOFT_ELEMENT else 1.0
        ke, me = element_matrices(ell, factor)
        ids = [2*e, 2*e+1, 2*e+2, 2*e+3]
        stiffness[np.ix_(ids, ids)] += ke
        mass[np.ix_(ids, ids)] += me
    free = np.arange(2, size)  # clamp displacement and rotation at beam root
    return stiffness[np.ix_(free, free)], mass[np.ix_(free, free)]


def frequencies_hz(loss):
    k, m = assemble(loss)
    lower = np.linalg.cholesky(m)
    inv_lower = np.linalg.solve(lower, np.eye(lower.shape[0]))
    a = inv_lower @ k @ inv_lower.T
    eigenvalues = np.linalg.eigvalsh((a + a.T) / 2.0)[:5]
    return np.sqrt(np.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)


def write_svg(path, shifts):
    w, h, left, top, right, bottom = 900, 520, 90, 55, 30, 80
    pw, ph = w-left-right, h-top-bottom
    ymin = min(-20.0, float(shifts.min())-2.0)
    x = lambda pct: left + pct/50.0*pw
    y = lambda val: top + (2.0-val)/(2.0-ymin)*ph
    colors = ("#1769aa", "#d04a35", "#26845b", "#8b5db7", "#d18a00")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{w/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">Natural-frequency shift from local EI reduction</text>']
    for tick in range(-20, 1, 5):
        yy = y(tick)
        svg.extend([f'<line x1="{left}" y1="{yy:.1f}" x2="{w-right}" y2="{yy:.1f}" stroke="#d9dee5"/>',
                    f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{tick}</text>'])
    for tick in range(0, 51, 10):
        xx = x(tick)
        svg.extend([f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{h-bottom}" stroke="#edf0f3"/>',
                    f'<text x="{xx:.1f}" y="{h-bottom+22}" text-anchor="middle" font-family="Arial" font-size="13">{tick}</text>'])
    for mode, color in enumerate(colors):
        points = " ".join(f'{x(loss*100):.1f},{y(value):.1f}' for loss, value in zip(LOSSES, shifts[:, mode]))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lx, ly = w-135, top+20+mode*22
        svg.extend([f'<line x1="{lx}" y1="{ly-4}" x2="{lx+18}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{lx+24}" y="{ly}" font-family="Arial" font-size="13">Mode {mode+1}</text>'])
    svg.extend([f'<line x1="{left}" y1="{y(0):.1f}" x2="{w-right}" y2="{y(0):.1f}" stroke="#555"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#333"/>',
                f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>',
                f'<text x="{w/2}" y="{h-22}" text-anchor="middle" font-family="Arial" font-size="15">Local element EI reduction (%)</text>',
                f'<text x="22" y="{h/2}" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 22 {h/2})">Frequency shift from intact beam (%)</text>',
                '</svg>'])
    path.write_text(chr(10).join(svg) + chr(10), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    freqs = np.array([frequencies_hz(loss) for loss in LOSSES])
    shifts = (freqs / freqs[0] - 1.0) * 100.0
    with (OUT / "frequency_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["local_EI_loss_fraction", *[f"mode_{i}_hz" for i in range(1, 6)],
                         *[f"mode_{i}_shift_percent" for i in range(1, 6)]])
        for i, loss in enumerate(LOSSES):
            writer.writerow([loss, *freqs[i], *shifts[i]])
    summary = {"study": "Localized stiffness-loss sensitivity of cantilever beam modes",
               "synthetic_beam": {"length_m": LENGTH, "width_m": WIDTH, "height_m": HEIGHT,
                                  "youngs_modulus_pa": YOUNG, "density_kg_m3": DENSITY,
                                  "elements": ELEMENTS, "soft_element_index_zero_based": SOFT_ELEMENT},
               "intact_natural_frequencies_hz": freqs[0].tolist(),
               "sweep": [{"local_EI_loss_fraction": loss, "frequencies_hz": freqs[i].tolist(),
                          "shift_from_intact_percent": shifts[i].tolist()} for i, loss in enumerate(LOSSES)],
               "limitations": ["Synthetic inputs, not measured data.",
                               "Damage is represented by reducing one element's EI.",
                               "No crack breathing, damping, shear deformation, or rotary inertia."]}
    (OUT / "study_summary.json").write_text(json.dumps(summary, indent=2) + chr(10), encoding="utf-8")
    write_svg(OUT / "frequency_shift.svg", shifts)
    print("Intact first five frequencies (Hz):", ", ".join(f"{v:.3f}" for v in freqs[0]))
    print(f"Results written to {OUT}")


if __name__ == "__main__":
    main()
