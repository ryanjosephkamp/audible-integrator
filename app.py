#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
The Audible Integrator - Interactive Streamlit Application
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         app.py (Streamlit Web Application)

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

A web-based GUI for exploring numerical integration methods through
audio sonification. Users can:
- Toggle between integrators mid-simulation
- Hear the difference between stable and unstable algorithms
- Watch real-time phase space and energy plots
- Understand why symplectic integrators matter for molecular dynamics
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
from typing import Optional
import sys
from pathlib import Path
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integrators import (
    ForwardEuler, RungeKutta4, VelocityVerlet, Leapfrog,
    get_integrator, INTEGRATOR_COMPARISON
)
from src.physics import HarmonicOscillator, DoublePendulum
from src.simulation import Simulator, SimulationConfig, compare_integrators
from src.sonification import SonificationConfig, EnergyToAudio


def generate_sonification_audio(energies: np.ndarray, times: np.ndarray, 
                                 initial_energy: float, sample_rate: int = 44100) -> bytes:
    """
    Generate WAV audio bytes from energy time series for web playback.
    
    Maps energy values to pitch (frequency) and amplitude, creating an auditory
    representation of the simulation's energy conservation.
    
    Args:
        energies: Array of energy values over time
        times: Array of time values
        initial_energy: Reference energy for mapping
        sample_rate: Audio sample rate (Hz)
        
    Returns:
        WAV file as bytes
    """
    import struct
    import wave
    
    # Audio parameters
    base_freq = 220.0  # A3
    freq_range = 440.0
    base_amplitude = 0.4
    
    # Calculate duration and samples
    duration = times[-1] - times[0]
    num_samples = int(duration * sample_rate)
    
    # Interpolate energy to audio sample rate
    audio_times = np.linspace(times[0], times[-1], num_samples)
    audio_energies = np.interp(audio_times, times, energies)
    
    # Generate audio samples
    samples = np.zeros(num_samples)
    phase = 0.0
    
    # Energy-to-audio mapping
    energy_mapper = EnergyToAudio(SonificationConfig(), initial_energy)
    
    for i in range(num_samples):
        energy = audio_energies[i]
        energy_mapper.update_energy(energy)
        
        # Map energy to frequency (logarithmic)
        ratio = max(energy / initial_energy, 0.01)
        log_ratio = np.log2(ratio)
        freq = base_freq + log_ratio * freq_range / 2
        freq = np.clip(freq, 80.0, 2000.0)
        
        # Map energy to amplitude
        amplitude = base_amplitude * min(ratio, 2.0)
        amplitude = np.clip(amplitude, 0.1, 0.8)
        
        # Get distortion amount
        distortion = energy_mapper.get_distortion_amount()
        
        # Generate sample with harmonics
        sample = np.sin(phase)
        sample += 0.5 * np.sin(phase * 2)  # 2nd harmonic
        sample += 0.25 * np.sin(phase * 3)  # 3rd harmonic
        
        # Apply distortion (soft clipping)
        if distortion > 0.01:
            gain = 1.0 + distortion * 5.0
            sample = np.tanh(sample * gain) / gain
            sample += distortion * 0.05 * (np.random.random() - 0.5)
        
        sample *= amplitude
        samples[i] = sample
        
        # Update phase
        phase += 2.0 * np.pi * freq / sample_rate
        if phase > 2.0 * np.pi:
            phase -= 2.0 * np.pi
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val * 0.9
    
    # Convert to 16-bit PCM
    samples_int = (samples * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples_int.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="The Audible Integrator",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stable {
        color: #27ae60;
        font-weight: bold;
    }
    .unstable {
        color: #e74c3c;
        font-weight: bold;
    }
    .author-info {
        font-size: 0.85rem;
        color: #888;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .author-info a {
        color: #1E88E5;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Header
# ============================================================================

st.markdown('<p class="main-header">🔊 The Audible Integrator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sonification of Energy Drift in Numerical Integration</p>', 
            unsafe_allow_html=True)

# Author information
st.markdown('''
<p class="author-info">
    <strong>Author:</strong> Ryan Kamp | 
    <strong>Affiliation:</strong> University of Cincinnati, Department of Computer Science<br>
    <strong>Email:</strong> <a href="mailto:kamprj@mail.uc.edu">kamprj@mail.uc.edu</a> | 
    <strong>Last Updated:</strong> January 22, 2026
</p>
''', unsafe_allow_html=True)

# Detailed description with examples
with st.expander("ℹ️ **What does this simulation do?** (Click to learn more)", expanded=False):
    st.markdown("""
    ### Understanding the Simulation
    
    This interactive tool demonstrates a fundamental concept in computational physics: 
    **how the choice of numerical algorithm affects the accuracy of physics simulations**.
    
    #### The Core Problem
    
    When scientists simulate physical systems on computers—whether modeling protein folding, 
    predicting planetary orbits, or designing new materials—they must break continuous motion 
    into discrete time steps. The algorithm used to calculate each step is called a 
    **numerical integrator**.
    
    **The critical insight:** Not all integrators are created equal. Some preserve the 
    fundamental physics (like energy conservation), while others introduce errors that 
    accumulate over time, leading to physically meaningless results.
    
    #### What You'll See (and Hear!)
    
    This simulation lets you:
    - **Compare four different integration algorithms** on the same physical system
    - **Watch energy conservation** (or lack thereof) in real-time
    - **Listen to the simulation**—stable algorithms produce a steady tone, while 
      unstable ones create rising pitch and distorted sounds
    
    #### Real-World Applications
    
    | Field | Application | Why Integrator Choice Matters |
    |-------|-------------|-------------------------------|
    | 🧬 **Drug Discovery** | Molecular dynamics of proteins | Simulations run for billions of steps; small errors compound |
    | 🚀 **Space Exploration** | Satellite trajectory planning | Missions span years; drift causes missed targets |
    | 🌡️ **Climate Science** | Long-term weather models | Century-scale predictions need stable algorithms |
    | ⚛️ **Particle Physics** | Accelerator beam dynamics | Particles circulate millions of times |
    | 🎮 **Video Games** | Physics engines for realism | Objects shouldn't spontaneously gain energy! |
    
    #### The Key Takeaway
    
    **Symplectic integrators** (like Velocity Verlet and Leapfrog) are specifically designed 
    to preserve the geometric structure of physics. They may be less "accurate" on paper, 
    but they maintain energy bounds forever—making them essential for long simulations.
    
    **Non-symplectic integrators** (like Forward Euler) can show dramatic energy drift 
    in just seconds, which you'll both see on the plots and *hear* as rising pitch!
    """)

st.markdown("""
This interactive demo shows how different numerical integration methods preserve (or don't preserve)
energy in classical mechanical systems. Run a simulation and **listen to the generated audio**—
stable integrators produce a steady tone, while unstable ones create rising, distorted sounds.
""")


# ============================================================================
# Sidebar Controls
# ============================================================================

st.sidebar.header("⚙️ Simulation Parameters")

# System selection
system_type = st.sidebar.selectbox(
    "Physical System",
    ["Harmonic Oscillator", "Double Pendulum"],
    help="Choose the physical system to simulate"
)

# Integrator selection
INTEGRATOR_MAP = {
    "Forward Euler": "euler",
    "Runge-Kutta 4": "rk4",
    "Velocity Verlet": "verlet",
    "Leapfrog": "leapfrog"
}

integrator_name = st.sidebar.selectbox(
    "Numerical Integrator",
    list(INTEGRATOR_MAP.keys()),
    index=0,
    help="Choose the integration algorithm"
)

# Time step
dt = st.sidebar.slider(
    "Time Step (dt)",
    min_value=0.001,
    max_value=0.1,
    value=0.02,
    step=0.001,
    format="%.3f",
    help="Larger time steps make errors more visible"
)

# Duration
duration = st.sidebar.slider(
    "Simulation Duration (s)",
    min_value=1.0,
    max_value=30.0,
    value=10.0,
    step=1.0
)

# Initial conditions
st.sidebar.header("🎯 Initial Conditions")

if system_type == "Harmonic Oscillator":
    x0 = st.sidebar.slider("Initial Position", -2.0, 2.0, 1.0, 0.1)
    v0 = st.sidebar.slider("Initial Velocity", -2.0, 2.0, 0.0, 0.1)
    initial_position = np.array([x0])
    initial_velocity = np.array([v0])
else:
    theta1 = st.sidebar.slider("θ₁ (degrees)", -180, 180, 90, 5)
    theta2 = st.sidebar.slider("θ₂ (degrees)", -180, 180, 90, 5)
    initial_position = np.array([np.radians(theta1), np.radians(theta2)])
    initial_velocity = np.array([0.0, 0.0])


# ============================================================================
# Run Simulation
# ============================================================================

@st.cache_data
def run_simulation(system_type, integrator_key, dt, duration, x0_tuple, v0_tuple):
    """Run simulation with caching for performance."""
    x0 = np.array(x0_tuple)
    v0 = np.array(v0_tuple)
    
    # Create system
    if system_type == "Harmonic Oscillator":
        system = HarmonicOscillator(mass=1.0, spring_constant=1.0)
    else:
        system = DoublePendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0)
    
    # Create integrator using the key directly
    integrator = get_integrator(integrator_key)
    
    # Run simulation
    config = SimulationConfig(dt=dt, duration=duration, realtime=False, audio_enabled=False)
    sim = Simulator(system, integrator, config)
    result = sim.run_fast(x0, v0)
    
    return result


# Run button
if st.sidebar.button("▶️ Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        # Map display name to integrator key
        integrator_key = INTEGRATOR_MAP[integrator_name]
        result = run_simulation(
            system_type,
            integrator_key,
            dt,
            duration,
            tuple(initial_position.tolist()),
            tuple(initial_velocity.tolist())
        )
        st.session_state['result'] = result
        st.session_state['integrator_name'] = integrator_name
        st.session_state['system_type'] = system_type  # Store system type used for simulation


# ============================================================================
# Display Results
# ============================================================================

if 'result' in st.session_state:
    result = st.session_state['result']
    # Use the system type that was used when the simulation was run
    result_system_type = st.session_state.get('system_type', 'Harmonic Oscillator')
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Integrator",
            result.integrator_name,
            delta="Symplectic ✓" if result.is_symplectic else "Non-Symplectic ✗"
        )
    
    with col2:
        drift_pct = result.energy_drift * 100
        st.metric(
            "Energy Drift",
            f"{drift_pct:+.4f}%",
            delta="Stable" if abs(drift_pct) < 1 else "Drifting!",
            delta_color="normal" if abs(drift_pct) < 1 else "inverse"
        )
    
    with col3:
        st.metric(
            "Max Deviation",
            f"{result.max_energy_deviation*100:.4f}%"
        )
    
    with col4:
        st.metric(
            "Energy Std Dev",
            f"{result.energy_std:.6f} J"
        )
    
    # Audio Sonification Section
    st.markdown("---")
    st.subheader("🔊 Listen to the Simulation")
    
    audio_col1, audio_col2 = st.columns([2, 1])
    
    with audio_col1:
        st.markdown("""
        **How to interpret the audio:**
        - 🎵 **Steady pitch** = Stable energy (good!)
        - 📈 **Rising pitch** = Energy is increasing (unstable!)
        - 🔊 **Increasing volume** = Energy growing
        - 😖 **Distorted sound** = Wildly fluctuating energy
        """)
    
    with audio_col2:
        if st.button("🎧 Generate Audio", type="secondary"):
            with st.spinner("Generating sonification..."):
                audio_bytes = generate_sonification_audio(
                    result.energies, 
                    result.times, 
                    result.initial_energy
                )
                st.session_state['audio_bytes'] = audio_bytes
                st.session_state['audio_integrator'] = result.integrator_name
    
    if 'audio_bytes' in st.session_state:
        st.audio(st.session_state['audio_bytes'], format='audio/wav')
        st.caption(f"🎵 Sonification of {st.session_state.get('audio_integrator', 'simulation')} - "
                   f"Listen for pitch changes indicating energy drift!")
    
    # Plots
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📈 Energy Plot", "🌀 Phase Space", "🔊 Audio Mapping"])
    
    with tab1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Absolute energy
        color = '#e74c3c' if not result.is_symplectic and abs(result.energy_drift) > 0.01 else '#27ae60'
        ax1.plot(result.times, result.energies, color=color, linewidth=1.5)
        ax1.axhline(y=result.initial_energy, color='black', linestyle='--', alpha=0.5, label='Initial')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Total Energy (J)')
        ax1.set_title(f'{result.integrator_name}: Energy vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative error
        relative_error = (result.energies - result.initial_energy) / result.initial_energy * 100
        ax2.plot(result.times, relative_error, color=color, linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Energy Drift')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if result_system_type == "Harmonic Oscillator":
            positions = result.positions.flatten()
            velocities = result.velocities.flatten()
            
            # Color by time
            scatter = ax.scatter(positions, velocities, c=result.times, cmap='viridis', 
                               s=2, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Time (s)')
            
            # Start and end markers
            ax.plot(positions[0], velocities[0], 'go', markersize=12, label='Start', zorder=5)
            ax.plot(positions[-1], velocities[-1], 'ro', markersize=12, label='End', zorder=5)
            
            # True orbit (circle for m=k=1)
            r = np.sqrt(2 * result.initial_energy)
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.3, label='True orbit')
            
            ax.set_xlabel('Position', fontsize=12)
            ax.set_ylabel('Velocity', fontsize=12)
            ax.set_aspect('equal')
        else:
            # Double pendulum: plot both angles
            ax.plot(result.positions[:, 0], result.positions[:, 1], 
                   'b-', linewidth=0.5, alpha=0.7)
            ax.plot(result.positions[0, 0], result.positions[0, 1], 
                   'go', markersize=12, label='Start')
            ax.plot(result.positions[-1, 0], result.positions[-1, 1], 
                   'ro', markersize=12, label='End')
            ax.set_xlabel('θ₁ (rad)', fontsize=12)
            ax.set_ylabel('θ₂ (rad)', fontsize=12)
        
        ax.set_title(f'{result.integrator_name}: Phase Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.markdown("""
        ### 🔊 How Energy Maps to Sound
        
        In real-time audio mode, the simulation energy would be converted to sound:
        """)
        
        # Calculate audio parameters
        config = SonificationConfig()
        mapper = EnergyToAudio(config, result.initial_energy)
        
        frequencies = []
        amplitudes = []
        
        for e in result.energies:
            frequencies.append(mapper.energy_to_frequency(e))
            amplitudes.append(mapper.energy_to_amplitude(e))
        
        frequencies = np.array(frequencies)
        amplitudes = np.array(amplitudes)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        ax1.plot(result.times, frequencies, 'b-', linewidth=1.5)
        ax1.axhline(y=config.base_frequency, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Base: {config.base_frequency} Hz')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('Mapped Audio Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(result.times, 0, amplitudes, color='purple', alpha=0.3)
        ax2.plot(result.times, amplitudes, 'purple', linewidth=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Mapped Audio Amplitude')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Interpretation
        freq_change = frequencies[-1] - frequencies[0]
        if abs(freq_change) < 10:
            st.success("🎵 **Stable tone** - The frequency stays nearly constant, producing a steady, pleasant sound.")
        elif freq_change > 0:
            st.error(f"📈 **Rising pitch** - Frequency increased by {freq_change:.1f} Hz. You would hear an increasingly high-pitched, alarming sound!")
        else:
            st.warning(f"📉 **Falling pitch** - Frequency decreased by {abs(freq_change):.1f} Hz.")


# ============================================================================
# Comparison Mode
# ============================================================================

st.markdown("---")
st.header("🔬 Compare All Integrators")

if st.button("📊 Run Comparison", type="secondary"):
    with st.spinner("Running all four integrators..."):
        # Create system based on current dropdown selection
        if system_type == "Harmonic Oscillator":
            system = HarmonicOscillator(mass=1.0, spring_constant=1.0)
        else:
            system = DoublePendulum(m1=1.0, m2=1.0, L1=1.0, L2=1.0)
        st.session_state['comparison_system_type'] = system_type
        
        config = SimulationConfig(dt=dt, duration=duration, realtime=False, audio_enabled=False)
        results = compare_integrators(system, initial_position, initial_velocity, config)
        
        st.session_state['comparison'] = results

if 'comparison' in st.session_state:
    results = st.session_state['comparison']
    
    # Summary table
    st.markdown("### Results Summary")
    
    summary_data = []
    for name, r in results.items():
        summary_data.append({
            "Integrator": name,
            "Symplectic": "✅ Yes" if r.is_symplectic else "❌ No",
            "Energy Drift (%)": f"{r.energy_drift*100:+.6f}",
            "Max Deviation (%)": f"{r.max_energy_deviation*100:.6f}",
            "Verdict": "🟢 Stable" if abs(r.energy_drift) < 0.01 else "🔴 Unstable"
        })
    
    st.table(summary_data)
    
    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'Forward Euler': '#e74c3c',
        'Runge-Kutta 4': '#f39c12',
        'Velocity Verlet': '#27ae60',
        'Leapfrog': '#3498db',
    }
    
    # Energy comparison
    ax1 = axes[0, 0]
    for name, r in results.items():
        ax1.plot(r.times, r.energies, label=name, color=colors.get(name, 'gray'), linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy (J)')
    ax1.set_title('Energy Conservation Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-scale error
    ax2 = axes[0, 1]
    for name, r in results.items():
        error = np.abs(r.energies - r.initial_energy) / r.initial_energy + 1e-16
        ax2.semilogy(r.times, error, label=name, color=colors.get(name, 'gray'), linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('|ΔE|/E₀ (log scale)')
    ax2.set_title('Energy Error (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Phase space for Euler (worst)
    ax3 = axes[1, 0]
    euler_result = results.get('Forward Euler')
    if euler_result:
        pos = euler_result.positions.flatten()
        vel = euler_result.velocities.flatten()
        ax3.plot(pos, vel, color=colors['Forward Euler'], linewidth=0.5, alpha=0.7)
        ax3.plot(pos[0], vel[0], 'go', markersize=10)
        ax3.plot(pos[-1], vel[-1], 'ro', markersize=10)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Forward Euler: Spiraling Out (Unstable!)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Phase space for Verlet (best)
    ax4 = axes[1, 1]
    verlet_result = results.get('Velocity Verlet')
    if verlet_result:
        pos = verlet_result.positions.flatten()
        vel = verlet_result.velocities.flatten()
        ax4.plot(pos, vel, color=colors['Velocity Verlet'], linewidth=0.5, alpha=0.7)
        ax4.plot(pos[0], vel[0], 'go', markersize=10)
        ax4.plot(pos[-1], vel[-1], 'ro', markersize=10)
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Velocity')
    ax4.set_title('Velocity Verlet: Closed Orbit (Stable!)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Audio comparison section
    st.markdown("### 🔊 Listen to the Difference")
    st.markdown("Generate audio for each integrator to *hear* the energy drift!")
    
    audio_cols = st.columns(4)
    integrator_names = ['Forward Euler', 'Runge-Kutta 4', 'Velocity Verlet', 'Leapfrog']
    
    for idx, int_name in enumerate(integrator_names):
        with audio_cols[idx]:
            if int_name in results:
                r = results[int_name]
                emoji = "🔴" if not r.is_symplectic else "🟢"
                st.markdown(f"**{emoji} {int_name}**")
                
                button_key = f"audio_{int_name.replace(' ', '_').replace('-', '_')}"
                if st.button(f"🎧 Play", key=button_key):
                    with st.spinner("Generating..."):
                        audio_bytes = generate_sonification_audio(
                            r.energies, r.times, r.initial_energy
                        )
                        st.session_state[f'comparison_audio_{int_name}'] = audio_bytes
                
                if f'comparison_audio_{int_name}' in st.session_state:
                    st.audio(st.session_state[f'comparison_audio_{int_name}'], format='audio/wav')


# ============================================================================
# Educational Content
# ============================================================================

st.markdown("---")

with st.expander("📚 Learn: Why Symplectic Integrators Matter"):
    st.markdown("""
    ## The Physics Behind the Audio
    
    ### What is a Hamiltonian System?
    
    A Hamiltonian system is described by:
    - **Position** $q$ and **Momentum** $p$
    - The **Hamiltonian** $H(q, p) = $ Total Energy = Kinetic + Potential
    
    For our harmonic oscillator:
    $$H = \\frac{1}{2}mv^2 + \\frac{1}{2}kx^2$$
    
    **Key Property:** In a closed system, $H$ is constant (conservation of energy).
    
    ### Why Do Some Integrators Fail?
    
    **Forward Euler** makes a simple approximation that doesn't respect the geometry
    of phase space. Each step introduces a small error that **accumulates systematically**,
    causing the energy to grow exponentially.
    
    **RK4** is more accurate per step, but still doesn't preserve the symplectic structure.
    Over long simulations, energy still drifts.
    
    ### What Makes Symplectic Integrators Special?
    
    **Symplectic integrators** (Verlet, Leapfrog) preserve the **phase space volume**.
    This is Liouville's theorem in action!
    
    While they have local errors (the orbit precesses slightly), the errors are **bounded**
    and don't accumulate. The energy oscillates around the true value but never drifts away.
    
    ### Why This Matters for Molecular Dynamics
    
    In MD simulations of proteins:
    - We integrate for **millions** of timesteps
    - Small systematic errors would cause the protein to "explode" (gain infinite energy)
    - Symplectic integrators keep the simulation stable for arbitrarily long times
    
    This is why **Velocity Verlet** is the default algorithm in every MD package
    (GROMACS, AMBER, OpenMM, etc.)!
    """)

with st.expander("📖 The Mathematics"):
    st.markdown("""
    ## Integrator Formulas
    
    ### Forward Euler (Order 1)
    $$x_{n+1} = x_n + v_n \\cdot dt$$
    $$v_{n+1} = v_n + a_n \\cdot dt$$
    
    ### Velocity Verlet (Order 2, Symplectic)
    $$x_{n+1} = x_n + v_n \\cdot dt + \\frac{1}{2} a_n \\cdot dt^2$$
    $$a_{n+1} = a(x_{n+1})$$
    $$v_{n+1} = v_n + \\frac{1}{2}(a_n + a_{n+1}) \\cdot dt$$
    
    ### Why Symplectic?
    
    A map $(q, p) \\to (Q, P)$ is **symplectic** if it preserves the symplectic 2-form:
    $$dq \\wedge dp = dQ \\wedge dP$$
    
    Geometrically, this means areas in phase space are preserved (Liouville's theorem).
    
    The Velocity Verlet algorithm can be shown to be a symplectic map, which is why
    it preserves the Hamiltonian structure and keeps energy bounded.
    """)


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Week 1 - Project 1: The Audible Integrator</strong></p>
    <p>Biophysics Portfolio | Computational Structural Biology Self-Study</p>
    <p><em>Turning abstract mathematical errors into visceral sensory experiences</em></p>
</div>
""", unsafe_allow_html=True)
