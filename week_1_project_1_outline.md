# Week 1 - Project 1: The "Audible Integrator" – Sonification of Energy Drift

## Overview

**Week:** 1 (Jan 20 – Jan 27)  
**Theme:** Numerical Integration, Conservation Laws, and Inter-Atomic Potentials  
**Goal:** Prove you understand *how* biophysics simulations calculate movement and *where* they fail.

---

## Project Details

### The "Gap" It Fills
Mastery of **Classical Mechanics** (Hamiltonian dynamics) and **Numerical Analysis**. It proves you understand why specific algorithms (like Velocity Verlet) are required for molecular dynamics versus standard algorithms (like Runge-Kutta) which are often used in other engineering fields.

### The Concept
Most students just plot a graph to show that energy is not conserved in a bad simulation. You will create a simulation of a Harmonic Oscillator or a Double Pendulum that produces **sound**.
- The "volume" or "pitch" of the background drone will be tied to the **System Energy**.
- As the simulation runs using an unstable integrator (like Forward Euler), the energy will drift, causing the sound to distort or rise in pitch, giving immediate auditory feedback on the "health" of the physics engine.

### Novelty/Creative Angle
Instead of visual graphs, this is an **auditory debugger** for physics engines. It turns an abstract mathematical error (global truncation error) into a visceral sensory experience.

### Technical Implementation
- **Language:** Python (NumPy for math, PyAudio or sounddevice for real-time audio synthesis).
- **Simulation:** Implement four integrators from scratch: Forward Euler, Runge-Kutta 4 (RK4), Velocity Verlet, and Leapfrog.
- **Output:** A Python script that generates a real-time audio stream.

### The "Paper" & Interactive Element
- *Interactive:* A simple GUI (Tkinter or browser-based via Streamlit) where a user can toggle between integrators mid-simulation and "hear" the chaos ensue.
- *Paper Focus:* "Auditory Feedback Mechanisms for Monitoring Symplectic Conservation in Molecular Dynamics Simulations."

---

## Progress Tracking

- [x] Initial research and planning
- [x] Core implementation
- [x] Testing and validation
- [x] Documentation and paper draft
- [x] Interactive demo creation

## Implementation Complete! 🎉

**Files created:**
- `src/integrators.py` - Four numerical integrators (Euler, RK4, Verlet, Leapfrog)
- `src/physics.py` - Harmonic oscillator and double pendulum systems
- `src/sonification.py` - Energy-to-audio mapping engine
- `src/simulation.py` - Main simulation framework
- `src/visualization.py` - Plotting utilities
- `app.py` - Interactive Streamlit web application
- `main.py` - Command-line interface
- `tests/test_integrators.py` - Comprehensive test suite
- `README.md` - Full documentation

**To run:**
```bash
pip install -r requirements.txt
streamlit run app.py          # Web interface
python main.py                 # Command line
python main.py --audio         # With sound!
```
