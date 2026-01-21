# 🔊 The Audible Integrator

**Week 1 - Project 1: Sonification of Energy Drift in Numerical Integration**

---

| **Author** | Ryan Kamp |
|------------|-------------------|
| **Affiliation** | University of Cincinnati Department of Computer Science |
| **Email** | kamprj@mail.uc.edu |
| **GitHub** | [github.com/ryanjosephkamp](https://github.com/ryanjosephkamp) |
| **Created** | January 21, 2026 |
| **Last Updated** | January 21, 2026 |
| **License** | MIT |

---

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project transforms abstract numerical errors into **audible feedback**, creating an "auditory debugger" for physics simulations. By mapping system energy to sound parameters (pitch, volume, distortion), you can *hear* when a simulation is becoming unstable.

### The Problem

In molecular dynamics simulations, we integrate Newton's equations millions of times. Most integration algorithms introduce systematic errors that cause **energy drift**—the total energy of the system gradually increases or decreases. Over long simulations, this leads to:

- 🔥 "Explosions" (energy → ∞)
- ❄️ "Freezing" (energy → 0)  
- 📈 Physically meaningless results

### The Solution: Symplectic Integrators

**Symplectic integrators** (like Velocity Verlet and Leapfrog) preserve the geometric structure of phase space, keeping energy bounded for arbitrarily long simulations. This project demonstrates why they're essential for molecular dynamics.

## 🎯 Learning Objectives

This project teaches you to master:

1. **Classical Mechanics** - Hamiltonian dynamics, conservation laws
2. **Numerical Analysis** - Truncation error, stability, convergence
3. **Symplectic Geometry** - Why phase space volume matters
4. **Audio Synthesis** - Real-time sound generation from data

## 🏗️ Project Structure

```
week_1_project_1/
├── app.py                 # Interactive Streamlit web application
├── main.py                # Command-line interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── src/
    ├── __init__.py       # Package initialization
    ├── integrators.py    # Four numerical integrators
    ├── physics.py        # Physical systems (oscillator, pendulum)
    ├── sonification.py   # Energy-to-audio mapping
    ├── simulation.py     # Main simulation engine
    └── visualization.py  # Plotting utilities
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to the project directory
cd week_1_project_1

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Interactive App

```bash
streamlit run app.py
```

This opens a web browser with an interactive GUI where you can:
- Toggle between integrators
- Adjust time step and duration
- Watch real-time energy plots
- See phase space trajectories

### Command Line Usage

```bash
# Compare all four integrators
python main.py

# Run a specific integrator
python main.py --integrator euler

# Enable audio sonification (hear the energy!)
python main.py --integrator euler --audio

# Save plots to disk
python main.py --save-plots --output-dir figures

# Double pendulum (chaotic system)
python main.py --system pendulum

# Custom parameters
python main.py --dt 0.05 --duration 20
```

## 🔬 The Four Integrators

| Integrator | Order | Symplectic | Energy Behavior |
|------------|-------|------------|-----------------|
| Forward Euler | 1 | ❌ No | Exponential growth (UNSTABLE) |
| Runge-Kutta 4 | 4 | ❌ No | Slow drift |
| Velocity Verlet | 2 | ✅ Yes | Bounded oscillation (STABLE) |
| Leapfrog | 2 | ✅ Yes | Bounded oscillation (STABLE) |

### Forward Euler (The Villain)

```
x_{n+1} = x_n + v_n · dt
v_{n+1} = v_n + a_n · dt
```

Simple but catastrophically unstable for oscillatory systems. Energy grows exponentially.

### Velocity Verlet (The Hero)

```
x_{n+1} = x_n + v_n · dt + ½ a_n · dt²
a_{n+1} = acceleration(x_{n+1})
v_{n+1} = v_n + ½ (a_n + a_{n+1}) · dt
```

Preserves phase space volume, keeping energy bounded forever. This is why it's the default in every MD package.

## 🔊 Audio Sonification

The sonification maps energy to sound:

| Energy State | Audio Response |
|--------------|----------------|
| E = E₀ (stable) | Steady tone at 220 Hz |
| E > E₀ (growing) | Rising pitch, louder volume |
| E fluctuating | Distorted, harsh timbre |

When running with `--audio`, you'll *hear* the difference:
- **Velocity Verlet**: Pleasant, steady drone
- **Forward Euler**: Increasingly high-pitched screech!

## 📊 Sample Output

Running `python main.py` produces:

```
════════════════════════════════════════════════════════════════════════════════
INTEGRATOR COMPARISON RESULTS
════════════════════════════════════════════════════════════════════════════════
Integrator           Symplectic    Energy Drift    Max Deviation  
────────────────────────────────────────────────────────────────────────────────
Forward Euler        NO            +0.234567       0.234567       
Runge-Kutta 4        NO            +0.000234       0.000456       
Velocity Verlet      YES           +0.000001       0.000123       
Leapfrog             YES           +0.000001       0.000123       
════════════════════════════════════════════════════════════════════════════════
```

## 📝 The Paper

This project includes the foundation for a paper:

**"Auditory Feedback Mechanisms for Monitoring Symplectic Conservation in Molecular Dynamics Simulations"**

Key points:
- Sonification as a debugging tool for scientific computing
- Real-time monitoring of conservation law violations
- Accessible demonstration of abstract mathematical concepts

## 🎓 Educational Value

This project demonstrates understanding of:

1. **Why MD uses Verlet, not RK4**: Despite RK4's higher order accuracy, Verlet's symplectic nature makes it essential for long simulations.

2. **The cost of simplicity**: Forward Euler is easy to implement but physically wrong for oscillatory systems.

3. **Conservation laws as diagnostics**: Energy should be constant—deviations reveal numerical errors.

4. **Phase space geometry**: Symplectic integrators preserve the manifold structure of Hamiltonian dynamics.

## 🔧 Technical Details

### Physical Systems

**Harmonic Oscillator**:
- Hamiltonian: H = ½mv² + ½kx²
- Analytical solution known (for validation)
- Perfect for demonstrating energy conservation

**Double Pendulum**:
- Chaotic dynamics
- Tests integrator robustness
- Beautiful phase space trajectories

### Audio Implementation

- Sample rate: 44.1 kHz (CD quality)
- Real-time synthesis using `sounddevice`
- Additive synthesis with harmonics
- Soft clipping distortion for instability

## 📚 References

1. Verlet, L. (1967). "Computer 'Experiments' on Classical Fluids." *Physical Review*.
2. Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge.
3. Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration*. Springer.

## 📄 License

MIT License - Feel free to use for educational purposes.

---

**Part of the Biophysics Portfolio**  
*Week 1: Numerical Integration, Conservation Laws, and Inter-Atomic Potentials*
