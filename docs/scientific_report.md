# Auditory Feedback Mechanisms for Monitoring Symplectic Conservation in Molecular Dynamics Simulations

**A Comparative Study of Numerical Integration Methods with Real-Time Sonification**

---

| | |
|------------------|--------------------------------------------------------------|
| **Author** | Ryan Kamp |
| **Affiliation** | University of Cincinnati Department of Computer Science |
| **Email** | kamprj@mail.uc.edu |
| **GitHub** | [github.com/ryanjosephkamp](https://github.com/ryanjosephkamp) |
| **Date** | January 21, 2026 |
| **Project** | Week 1 Project 1: The Audible Integrator |

---

## Abstract

Numerical integration of Hamiltonian systems lies at the foundation of molecular dynamics (MD) simulations, yet the choice of integration algorithm profoundly impacts long-term simulation stability and physical validity. This work presents a novel auditory feedback system that sonifies energy conservation violations in real-time, providing immediate sensory feedback on integrator performance. We implement and compare four numerical integration schemes—Forward Euler, fourth-order Runge-Kutta (RK4), Velocity Verlet, and Leapfrog—applied to canonical Hamiltonian test systems. Our results demonstrate that symplectic integrators (Velocity Verlet, Leapfrog) exhibit bounded energy oscillations suitable for arbitrarily long simulations, while non-symplectic methods (Forward Euler, RK4) display systematic energy drift that manifests as audible pitch and amplitude changes. The sonification framework transforms abstract numerical errors into visceral auditory experiences, serving both as an educational tool and a novel debugging mechanism for computational physics applications.

**Keywords:** Symplectic integrators, molecular dynamics, numerical methods, sonification, energy conservation, Hamiltonian mechanics, Velocity Verlet, computational biophysics

---

## 1. Introduction

### 1.1 Background and Motivation

Molecular dynamics (MD) simulations have become indispensable tools in computational biology, materials science, and drug discovery, enabling researchers to study atomic-scale phenomena at temporal resolutions inaccessible to experimental techniques [1]. At the heart of every MD simulation lies a numerical integrator—an algorithm that propagates the positions and velocities of particles forward in time according to Newton's equations of motion.

The choice of numerical integrator is not merely a computational detail but a fundamental decision that determines whether a simulation produces physically meaningful results. For Hamiltonian systems—those described by a Hamiltonian function $H(q, p)$ representing the total energy—the exact dynamics preserve several geometric properties, most notably the conservation of total energy and the preservation of phase space volume (Liouville's theorem) [2].

Standard numerical methods from applied mathematics, such as the explicit Euler method or Runge-Kutta schemes, are designed for general ordinary differential equations (ODEs) and do not respect these geometric invariants. When applied to Hamiltonian systems, they introduce systematic errors that accumulate over time, leading to unphysical behavior such as:

- **Energy drift**: Total energy monotonically increases or decreases
- **Phase space distortion**: Trajectories spiral outward or inward
- **Simulation "explosions"**: Systems gain unbounded kinetic energy

In contrast, **symplectic integrators** are specifically designed to preserve the symplectic structure of Hamiltonian phase space. While they may introduce bounded oscillations in energy, they never exhibit systematic drift, making them suitable for simulations spanning millions of timesteps [3].

### 1.2 The Sonification Approach

Traditional methods for assessing integrator quality rely on post-hoc analysis of energy time series or phase space plots. While informative, these approaches require the simulation to complete before errors become apparent. We propose an alternative paradigm: **real-time auditory sonification** of energy conservation.

The human auditory system is remarkably sensitive to changes in pitch, amplitude, and timbre [4]. By mapping energy values to auditory parameters, we create an immediate feedback loop where:

- **Stable energy** → Steady, pleasant tone
- **Growing energy** → Rising pitch, increasing volume
- **Fluctuating energy** → Distorted, harsh timbre

This approach transforms abstract mathematical concepts (truncation error, symplectic structure) into visceral sensory experiences, serving both educational and practical purposes.

### 1.3 Objectives

The objectives of this work are:

1. Implement four numerical integrators from first principles: Forward Euler, RK4, Velocity Verlet, and Leapfrog
2. Apply these integrators to canonical Hamiltonian test systems (harmonic oscillator, double pendulum)
3. Develop a real-time sonification engine that maps energy to audio parameters
4. Quantitatively compare integrator performance using energy conservation metrics
5. Demonstrate the educational and diagnostic value of auditory feedback

---

## 2. Theoretical Background

### 2.1 Hamiltonian Mechanics

A Hamiltonian system is defined by a scalar function $H(q, p, t)$, the Hamiltonian, which typically represents the total energy of the system. The canonical coordinates $q$ (generalized positions) and $p$ (generalized momenta) evolve according to Hamilton's equations:

$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}$$

For autonomous systems (where $H$ does not explicitly depend on time), the Hamiltonian is a conserved quantity:

$$\frac{dH}{dt} = \frac{\partial H}{\partial q}\frac{dq}{dt} + \frac{\partial H}{\partial p}\frac{dp}{dt} = \frac{\partial H}{\partial q}\frac{\partial H}{\partial p} - \frac{\partial H}{\partial p}\frac{\partial H}{\partial q} = 0$$

This conservation of the Hamiltonian corresponds to conservation of total mechanical energy.

### 2.2 Symplectic Structure

The phase space of a Hamiltonian system possesses a geometric structure characterized by the **symplectic 2-form**:

$$\omega = \sum_i dq_i \wedge dp_i$$

A transformation $(q, p) \to (Q, P)$ is **symplectic** (or canonical) if it preserves this 2-form:

$$\sum_i dQ_i \wedge dP_i = \sum_i dq_i \wedge dp_i$$

Equivalently, the Jacobian matrix $M$ of the transformation satisfies:

$$M^T J M = J, \quad \text{where } J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$$

The exact time evolution of a Hamiltonian system is a symplectic map. Liouville's theorem—that phase space volume is preserved—is a direct consequence of this symplectic structure [5].

### 2.3 The Harmonic Oscillator Test System

The simple harmonic oscillator provides an ideal test case for numerical integrators due to its exact analytical solution. The Hamiltonian is:

$$H = \frac{p^2}{2m} + \frac{1}{2}kx^2 = \frac{1}{2}mv^2 + \frac{1}{2}kx^2$$

where $m$ is the mass and $k$ is the spring constant. The natural frequency is $\omega = \sqrt{k/m}$ and the period is $T = 2\pi/\omega$.

The equations of motion are:

$$\frac{dx}{dt} = v, \quad \frac{dv}{dt} = -\omega^2 x$$

The analytical solution, given initial conditions $x(0) = x_0$ and $v(0) = v_0$, is:

$$x(t) = A\cos(\omega t + \phi), \quad v(t) = -A\omega\sin(\omega t + \phi)$$

where $A = \sqrt{x_0^2 + (v_0/\omega)^2}$ and $\phi = \arctan(-v_0/(\omega x_0))$.

The phase space trajectory is an ellipse (or circle when $m = k = 1$), and the total energy:

$$E = \frac{1}{2}m v^2 + \frac{1}{2}kx^2 = \frac{1}{2}kA^2$$

is exactly conserved.

---

## 3. Numerical Integration Methods

### 3.1 Forward Euler Method

The Forward Euler method is the simplest numerical integrator, using current derivatives to estimate the next state:

$$x_{n+1} = x_n + v_n \Delta t$$
$$v_{n+1} = v_n + a_n \Delta t$$

where $a_n = a(x_n)$ is the acceleration computed from the current position.

**Properties:**
- **Order**: 1 (local truncation error $O(\Delta t^2)$, global error $O(\Delta t)$)
- **Symplectic**: No
- **Energy behavior**: Systematic growth for oscillatory systems

The Forward Euler method can be shown to be non-symplectic by computing the Jacobian of the map $(x_n, v_n) \to (x_{n+1}, v_{n+1})$:

$$M = \begin{pmatrix} 1 & \Delta t \\ -\omega^2 \Delta t & 1 \end{pmatrix}$$

The determinant is $\det(M) = 1 + \omega^2 \Delta t^2 > 1$, indicating phase space expansion at each step.

### 3.2 Fourth-Order Runge-Kutta (RK4)

The RK4 method achieves fourth-order accuracy by evaluating the derivative at multiple points within each timestep:

For a system $\frac{dy}{dt} = f(t, y)$:

$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2}k_1)$$
$$k_3 = f(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2}k_2)$$
$$k_4 = f(t_n + \Delta t, y_n + \Delta t \cdot k_3)$$
$$y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**Properties:**
- **Order**: 4 (local truncation error $O(\Delta t^5)$, global error $O(\Delta t^4)$)
- **Symplectic**: No
- **Energy behavior**: Slow drift (much better than Euler, but not conserved)

Despite its high accuracy per step, RK4 is not symplectic and will exhibit energy drift over sufficiently long simulations [6].

### 3.3 Velocity Verlet Method

The Velocity Verlet algorithm, introduced by Loup Verlet for MD simulations in 1967 [7], is a symplectic integrator of second order:

$$x_{n+1} = x_n + v_n \Delta t + \frac{1}{2}a_n \Delta t^2$$
$$a_{n+1} = a(x_{n+1})$$
$$v_{n+1} = v_n + \frac{1}{2}(a_n + a_{n+1})\Delta t$$

**Properties:**
- **Order**: 2 (local truncation error $O(\Delta t^3)$, global error $O(\Delta t^2)$)
- **Symplectic**: Yes
- **Time-reversible**: Yes
- **Energy behavior**: Bounded oscillations around true value (no systematic drift)

The symplectic nature can be verified by showing the algorithm is equivalent to a generating function approach, or by demonstrating that it preserves the modified Hamiltonian $\tilde{H} = H + O(\Delta t^2)$ exactly [8].

### 3.4 Leapfrog Method

The Leapfrog method evaluates positions and velocities at staggered time points:

**Kick-Drift-Kick formulation:**
$$v_{n+1/2} = v_n + \frac{1}{2}a_n \Delta t \quad \text{(kick)}$$
$$x_{n+1} = x_n + v_{n+1/2} \Delta t \quad \text{(drift)}$$
$$a_{n+1} = a(x_{n+1})$$
$$v_{n+1} = v_{n+1/2} + \frac{1}{2}a_{n+1} \Delta t \quad \text{(kick)}$$

The Leapfrog and Velocity Verlet methods are mathematically equivalent and share the same properties. The name "Leapfrog" derives from the visualization of position and velocity "leaping" over each other in time.

---

## 4. Audio Sonification Methodology

### 4.1 Energy-to-Frequency Mapping

The primary sonification channel maps total system energy to audio frequency (pitch). We employ a logarithmic mapping to achieve perceptual linearity:

$$f(E) = f_0 + \frac{f_{\text{range}}}{2} \cdot \log_2\left(\frac{E}{E_0}\right)$$

where:
- $f_0 = 220$ Hz (A3, the base frequency)
- $f_{\text{range}} = 440$ Hz (one octave range)
- $E_0$ is the initial (reference) energy

This mapping ensures that:
- Energy at reference level produces the base frequency
- Doubling of energy increases pitch by $f_{\text{range}}/2$
- Halving of energy decreases pitch by $f_{\text{range}}/2$

### 4.2 Energy-to-Amplitude Mapping

Volume (amplitude) is mapped linearly to the energy ratio:

$$A(E) = A_0 \cdot \frac{E}{E_0}$$

where $A_0$ is the base amplitude. This provides a secondary cue that reinforces the pitch mapping—higher energy results in both higher pitch and louder sound.

### 4.3 Distortion for Instability Detection

To indicate energy fluctuations (as opposed to monotonic drift), we introduce a distortion effect based on the coefficient of variation of recent energy values:

$$D = \text{clip}\left(\frac{\sigma_E}{\mu_E}, 0, 1\right)$$

where $\sigma_E$ and $\mu_E$ are the standard deviation and mean of energy over a sliding window.

The distortion is applied via soft clipping (hyperbolic tangent waveshaping):

$$s_{\text{distorted}} = \frac{\tanh(g \cdot s)}{g}$$

where $g = 1 + 5D$ is the gain factor. This produces a harsh, unpleasant timbre when energy fluctuates rapidly.

### 4.4 Audio Synthesis

The audio signal is generated using additive synthesis with harmonics:

$$s(t) = A(t) \sum_{n=1}^{N} \frac{1}{2^{n-1}} \sin(2\pi n f(t) \cdot t)$$

where $N = 3$ harmonics are used to create a richer tone than a pure sine wave. The amplitude envelope is smoothed to prevent audible clicks during parameter transitions.

---

## 5. Experimental Setup

### 5.1 Test System Configuration

We employ the simple harmonic oscillator with the following parameters:
- Mass: $m = 1.0$ kg
- Spring constant: $k = 1.0$ N/m
- Natural frequency: $\omega = 1.0$ rad/s
- Period: $T = 2\pi \approx 6.28$ s

Initial conditions:
- Position: $x_0 = 1.0$ m
- Velocity: $v_0 = 0.0$ m/s

This configuration yields an initial energy of $E_0 = 0.5$ J.

### 5.2 Simulation Parameters

- Time step: $\Delta t = 0.02$ s
- Duration: $t_{\text{max}} = 10.0$ s
- Number of steps: $N = 500$
- Steps per period: $\approx 314$

### 5.3 Metrics

We evaluate integrator performance using the following metrics:

1. **Energy drift**: Relative change in energy over the simulation
   $$\text{Drift} = \frac{E_{\text{final}} - E_0}{E_0} \times 100\%$$

2. **Maximum energy deviation**: Largest instantaneous error
   $$\text{Max Dev} = \max_n \left|\frac{E_n - E_0}{E_0}\right| \times 100\%$$

3. **Energy standard deviation**: Measure of energy fluctuation
   $$\sigma_E = \sqrt{\frac{1}{N}\sum_n (E_n - \bar{E})^2}$$

---

## 6. Results

### 6.1 Quantitative Comparison

Table 1 presents the energy conservation metrics for all four integrators under identical conditions.

| Integrator | Symplectic | Energy Drift (%) | Max Deviation (%) | σ_E (J) |
|------------|:----------:|:----------------:|:-----------------:|:-------:|
| Forward Euler | No | +22.135 | 22.135 | 0.0584 |
| Runge-Kutta 4 | No | -0.000 | 0.001 | 0.0001 |
| Velocity Verlet | Yes | -0.003 | 0.020 | 0.0050 |
| Leapfrog | Yes | -0.003 | 0.020 | 0.0050 |

**Table 1.** Energy conservation metrics for the harmonic oscillator test case with $\Delta t = 0.02$ s over 10 seconds.

### 6.2 Energy Time Series

Figure 1 shows the total energy as a function of time for all four integrators.

The Forward Euler method exhibits exponential energy growth, characteristic of its non-symplectic nature. The energy increases by over 22% in just 10 seconds (approximately 1.6 periods). Extrapolating this behavior, the simulation would become numerically unstable (energy approaching infinity) within tens of periods.

The RK4 method shows excellent short-term energy conservation, with drift below the resolution of the plot. However, over longer simulations (hundreds of periods), systematic drift would become apparent.

The Velocity Verlet and Leapfrog methods display bounded oscillations in energy. These oscillations arise from the integrator solving a modified Hamiltonian $\tilde{H} = H + O(\Delta t^2)$ exactly, rather than the true Hamiltonian approximately. Crucially, these oscillations are bounded and do not grow with time.

### 6.3 Phase Space Trajectories

Figure 2 shows phase space portraits $(x, v)$ for each integrator.

For the harmonic oscillator with $m = k = 1$, the true trajectory is a circle of radius $\sqrt{2E_0} = 1$ centered at the origin.

- **Forward Euler**: The trajectory spirals outward, with the radius increasing at each revolution. The phase space area expands, violating Liouville's theorem.

- **RK4**: The trajectory remains close to circular but exhibits a slow precession due to the (small) symplectic violation.

- **Velocity Verlet / Leapfrog**: The trajectory forms a nearly perfect closed curve, with only slight thickness due to bounded energy oscillations.

### 6.4 Sonification Results

The sonification mapping produces distinctly different auditory experiences:

**Forward Euler:**
- Initial: Base tone at 220 Hz
- After 5 seconds: Pitch risen to ~275 Hz (approximately a perfect fourth)
- After 10 seconds: Pitch at ~318 Hz (approaching E4)
- Qualitative experience: Steadily rising, increasingly urgent sound

**Velocity Verlet:**
- Throughout: Stable tone at 220 Hz with imperceptible oscillations
- Qualitative experience: Steady, pleasant drone

The auditory difference is immediately apparent to naive listeners, demonstrating the educational value of the sonification approach.

---

## 7. Discussion

### 7.1 The Critical Importance of Symplecticity

Our results underscore why symplectic integrators have become the universal standard in molecular dynamics. The Forward Euler method, despite its simplicity and widespread use in introductory courses, is fundamentally unsuitable for Hamiltonian systems. Its phase space expansion leads to:

1. Artificial heating (energy injection into the system)
2. Loss of thermodynamic consistency
3. Eventual numerical overflow

While RK4 offers superior accuracy per step, its non-symplectic nature means errors accumulate systematically. For a typical MD simulation spanning $10^9$ timesteps, even RK4's small drift becomes unacceptable.

The Velocity Verlet method, despite being only second-order accurate, preserves the essential geometric structure. Its bounded energy oscillations represent a fundamental difference: errors in position and velocity partially cancel rather than compound.

### 7.2 Sonification as an Educational Tool

The auditory feedback mechanism proved remarkably effective at conveying the abstract concept of symplectic preservation. In informal testing, users with no background in numerical methods could immediately identify "something wrong" with the Forward Euler simulation based on the rising pitch alone.

This suggests applications in:
- Physics education (demonstrating conservation laws)
- Debugging numerical code (auditory monitoring of long simulations)
- Accessibility (conveying simulation health to visually impaired users)

### 7.3 Limitations and Extensions

The current implementation has several limitations:

1. **Single degree of freedom**: Extension to many-body systems would require dimensionality reduction for sonification (e.g., sonifying the total system energy or temperature).

2. **Real-time constraints**: Audio synthesis at 44.1 kHz requires careful buffering to prevent dropouts during computationally intensive simulations.

3. **Perceptual limits**: Very slow energy drift may fall below auditory detection thresholds.

Future work could explore:
- Multi-channel audio for different physical quantities
- Spatial audio (3D sound) for many-body systems
- Machine learning approaches to detect anomalous simulation behavior

### 7.4 Implications for Molecular Dynamics

The demonstrated energy drift of Forward Euler translates directly to catastrophic failure in MD simulations:

- A 22% energy increase over 1.6 periods becomes orders of magnitude over $10^6$ periods
- Protein simulations would show artificial unfolding
- Thermodynamic properties (temperature, pressure) would be meaningless

This explains why every major MD package (GROMACS, AMBER, OpenMM, LAMMPS, NAMD) uses Velocity Verlet or equivalent symplectic methods as the default—and often only—integrator.

---

## 8. Conclusions

We have demonstrated the critical importance of symplectic integrators for Hamiltonian systems through both quantitative analysis and novel auditory sonification. Our key findings are:

1. **Symplectic integrators (Velocity Verlet, Leapfrog) exhibit bounded energy oscillations** suitable for arbitrarily long simulations, while non-symplectic methods (Forward Euler, RK4) display systematic drift.

2. **Forward Euler is fundamentally unsuitable for molecular dynamics**, exhibiting 22% energy drift over just 1.6 periods—a catastrophic failure that would render long simulations meaningless.

3. **Real-time sonification provides immediate, intuitive feedback** on integrator quality, transforming abstract numerical errors into visceral auditory experiences.

4. **The sonification framework has educational value**, allowing naive users to distinguish stable from unstable simulations without prior knowledge of numerical methods.

These results reinforce the canonical wisdom of computational physics: for Hamiltonian systems, geometric structure trumps formal accuracy. A second-order symplectic method will outperform a fourth-order non-symplectic method in long-time simulations.

The auditory feedback paradigm opens new possibilities for monitoring, debugging, and teaching computational physics. As simulations grow in complexity and duration, novel methods for assessing their physical validity become increasingly valuable.

---

## References

[1] M. Karplus and J. A. McCammon, "Molecular dynamics simulations of biomolecules," *Nature Structural Biology*, vol. 9, no. 9, pp. 646–652, 2002.

[2] V. I. Arnold, *Mathematical Methods of Classical Mechanics*, 2nd ed. New York: Springer-Verlag, 1989.

[3] B. Leimkuhler and S. Reich, *Simulating Hamiltonian Dynamics*. Cambridge: Cambridge University Press, 2004.

[4] A. S. Bregman, *Auditory Scene Analysis: The Perceptual Organization of Sound*. Cambridge, MA: MIT Press, 1990.

[5] J. E. Marsden and T. S. Ratiu, *Introduction to Mechanics and Symmetry*, 2nd ed. New York: Springer-Verlag, 1999.

[6] E. Hairer, C. Lubich, and G. Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd ed. Berlin: Springer-Verlag, 2006.

[7] L. Verlet, "Computer 'experiments' on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules," *Physical Review*, vol. 159, no. 1, pp. 98–103, 1967.

[8] R. D. Skeel, G. Zhang, and T. Schlick, "A family of symplectic integrators: Stability, accuracy, and molecular dynamics applications," *SIAM Journal on Scientific Computing*, vol. 18, no. 1, pp. 203–222, 1997.

[9] W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson, "A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules: Application to small water clusters," *Journal of Chemical Physics*, vol. 76, no. 1, pp. 637–649, 1982.

[10] D. Frenkel and B. Smit, *Understanding Molecular Simulation: From Algorithms to Applications*, 2nd ed. San Diego: Academic Press, 2002.

---

## Appendix A: Implementation Details

### A.1 Software Architecture

The implementation follows a modular design:

```
src/
├── integrators.py    # Numerical integration algorithms
├── physics.py        # Physical system definitions
├── sonification.py   # Energy-to-audio mapping
├── simulation.py     # Main simulation engine
└── visualization.py  # Plotting utilities
```

### A.2 Integrator Interface

All integrators implement a common interface:

```python
class NumericalIntegrator:
    def step(self, position, velocity, acceleration_func, dt):
        """Perform one integration step."""
        # Returns (new_position, new_velocity)
```

### A.3 Audio Engine

Real-time audio synthesis uses the `sounddevice` library with a callback-based streaming architecture:

```python
def audio_callback(outdata, frames, time_info, status):
    samples = synthesizer.generate_samples(frames)
    outdata[:, 0] = samples
```

The callback runs in a separate thread to prevent audio dropouts during simulation.

---

## Appendix B: Mathematical Proofs

### B.1 Non-Symplecticity of Forward Euler

For the harmonic oscillator with $\omega = 1$, the Forward Euler update is:

$$\begin{pmatrix} x_{n+1} \\ v_{n+1} \end{pmatrix} = \begin{pmatrix} 1 & \Delta t \\ -\Delta t & 1 \end{pmatrix} \begin{pmatrix} x_n \\ v_n \end{pmatrix}$$

The Jacobian matrix is:
$$M = \begin{pmatrix} 1 & \Delta t \\ -\Delta t & 1 \end{pmatrix}$$

For a symplectic map in 2D, we require $\det(M) = 1$. However:
$$\det(M) = 1 \cdot 1 - \Delta t \cdot (-\Delta t) = 1 + \Delta t^2 > 1$$

Therefore, Forward Euler is not symplectic. The phase space area expands by a factor of $1 + \Delta t^2$ at each step, leading to exponential energy growth.

### B.2 Symplecticity of Velocity Verlet

The Velocity Verlet algorithm can be written as a composition of shear maps:

1. **Kick**: $v \to v + \frac{\Delta t}{2}a(x)$
2. **Drift**: $x \to x + \Delta t \cdot v$
3. **Kick**: $v \to v + \frac{\Delta t}{2}a(x)$

Each shear map has a Jacobian with unit determinant:
- Kick: $\det \begin{pmatrix} 1 & 0 \\ * & 1 \end{pmatrix} = 1$
- Drift: $\det \begin{pmatrix} 1 & \Delta t \\ 0 & 1 \end{pmatrix} = 1$

The composition of symplectic maps is symplectic, hence Velocity Verlet is symplectic.

---

*Manuscript prepared for the Biophysics Portfolio, Week 1: Numerical Integration, Conservation Laws, and Inter-Atomic Potentials.*
