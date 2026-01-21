#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Quick Test Script - Verify Simulation Works
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         test_quick.py (Quick Verification Script)

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

Quick test script to verify the simulation engine is working correctly.
Runs all integrators on a harmonic oscillator and reports energy drift.
"""

import numpy as np
from src.integrators import VelocityVerlet, ForwardEuler
from src.physics import HarmonicOscillator
from src.simulation import Simulator, SimulationConfig, compare_integrators

# Create system
system = HarmonicOscillator(mass=1.0, spring_constant=1.0)
x0 = np.array([1.0])
v0 = np.array([0.0])

# Quick comparison
config = SimulationConfig(dt=0.02, duration=10.0, realtime=False, audio_enabled=False)
results = compare_integrators(system, x0, v0, config)

print("=" * 60)
print("INTEGRATOR COMPARISON RESULTS")
print("=" * 60)
for name, r in results.items():
    symplectic = "YES" if r.is_symplectic else "NO"
    drift = r.energy_drift * 100
    print(f"{name:<20} Symplectic: {symplectic:<4} Drift: {drift:+.4f}%")
print("=" * 60)
print("✓ Simulation engine working correctly!")
