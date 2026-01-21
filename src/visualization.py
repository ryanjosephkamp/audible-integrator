#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Visualization Utilities
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         visualization.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

This module provides plotting functions for:
- Phase space trajectories
- Energy over time
- Integrator comparisons
- Audio parameter visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Tuple
import matplotlib.patches as mpatches

# Use a nice style
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme
COLORS = {
    'Forward Euler': '#e74c3c',      # Red (unstable)
    'Runge-Kutta 4': '#f39c12',      # Orange (drifts)
    'Velocity Verlet': '#27ae60',    # Green (stable)
    'Leapfrog': '#3498db',           # Blue (stable)
}


def plot_energy_comparison(
    results: dict,
    title: str = "Energy Conservation Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot energy vs time for multiple integrators.
    
    Args:
        results: Dict mapping integrator name to SimulationResult
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Absolute energy
    for name, result in results.items():
        color = COLORS.get(name, '#333333')
        ax1.plot(result.times, result.energies, label=name, color=color, linewidth=1.5)
    
    ax1.axhline(y=list(results.values())[0].initial_energy, 
                color='black', linestyle='--', alpha=0.5, label='Initial Energy')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Total Energy (J)', fontsize=12)
    ax1.set_title('Total Energy vs Time', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right: Relative energy error
    for name, result in results.items():
        color = COLORS.get(name, '#333333')
        relative_error = (result.energies - result.initial_energy) / result.initial_energy
        ax2.plot(result.times, relative_error * 100, label=name, color=color, linewidth=1.5)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Relative Energy Error (%)', fontsize=12)
    ax2.set_title('Energy Drift', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_phase_space(
    results: dict,
    title: str = "Phase Space Trajectories",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot phase space (position vs velocity) for multiple integrators.
    
    For a harmonic oscillator, the true trajectory is an ellipse.
    Deviations from the ellipse show integrator errors.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        color = COLORS.get(name, '#333333')
        
        # Plot trajectory
        positions = result.positions.flatten()
        velocities = result.velocities.flatten()
        
        # Color by time (early = dark, late = light)
        times = result.times
        scatter = ax.scatter(positions, velocities, c=times, cmap='viridis',
                           s=1, alpha=0.7)
        
        # Mark start and end
        ax.plot(positions[0], velocities[0], 'go', markersize=10, label='Start')
        ax.plot(positions[-1], velocities[-1], 'ro', markersize=10, label='End')
        
        # True ellipse for reference (harmonic oscillator)
        E0 = result.initial_energy
        # For m=1, k=1: E = 0.5*v^2 + 0.5*x^2, so x_max = v_max = sqrt(2E)
        r = np.sqrt(2 * E0)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.3, label='True orbit')
        
        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('Velocity', fontsize=10)
        ax.set_title(f'{name}\n(Drift: {result.energy_drift*100:.2f}%)', fontsize=12)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Time (s)')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_audio_parameters(
    result,  # SimulationResult
    title: str = "Audio Sonification Parameters",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize how energy maps to audio parameters.
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    times = result.times
    
    # Energy
    ax1 = axes[0]
    relative_energy = result.energies / result.initial_energy
    ax1.plot(times, relative_energy, 'b-', linewidth=1.5)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('E/E₀', fontsize=12)
    ax1.set_title(f'{result.integrator_name} - Energy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Frequency (if available)
    ax2 = axes[1]
    if result.frequencies is not None:
        ax2.plot(times[:len(result.frequencies)], result.frequencies, 'g-', linewidth=1.5)
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    ax2.set_title('Mapped Audio Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Amplitude (if available)
    ax3 = axes[2]
    if result.amplitudes is not None:
        ax3.plot(times[:len(result.amplitudes)], result.amplitudes, 'm-', linewidth=1.5)
    ax3.set_ylabel('Amplitude', fontsize=12)
    ax3.set_title('Mapped Audio Amplitude', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Distortion (if available)
    ax4 = axes[3]
    if result.distortions is not None:
        ax4.fill_between(times[:len(result.distortions)], 0, result.distortions,
                        color='red', alpha=0.5)
        ax4.plot(times[:len(result.distortions)], result.distortions, 'r-', linewidth=1.5)
    ax4.set_ylabel('Distortion', fontsize=12)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_title('Audio Distortion (Instability Indicator)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_figure(
    results: dict,
    title: str = "Audible Integrator: Summary",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary figure with all comparisons.
    """
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top left: Energy vs time
    ax1 = fig.add_subplot(gs[0, 0])
    for name, result in results.items():
        color = COLORS.get(name, '#333333')
        ax1.plot(result.times, result.energies, label=name, color=color, linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy (J)')
    ax1.set_title('Energy Conservation')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Top middle: Energy drift (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    for name, result in results.items():
        color = COLORS.get(name, '#333333')
        error = np.abs(result.energies - result.initial_energy) / result.initial_energy
        ax2.semilogy(result.times, error + 1e-16, label=name, color=color, linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('|ΔE|/E₀ (log scale)')
    ax2.set_title('Energy Error (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    # Top right: Bar chart of final drift
    ax3 = fig.add_subplot(gs[0, 2])
    names = list(results.keys())
    drifts = [abs(results[n].energy_drift) * 100 for n in names]
    colors = [COLORS.get(n, '#333333') for n in names]
    bars = ax3.bar(range(len(names)), drifts, color=colors)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Final Drift (%)')
    ax3.set_title('Total Energy Drift')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom: Phase space for each integrator (4 panels)
    for idx, (name, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx % 3])
        color = COLORS.get(name, '#333333')
        
        positions = result.positions.flatten()
        velocities = result.velocities.flatten()
        
        ax.plot(positions, velocities, color=color, linewidth=0.5, alpha=0.7)
        ax.plot(positions[0], velocities[0], 'go', markersize=8)
        ax.plot(positions[-1], velocities[-1], 'ro', markersize=8)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title(f'{name}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if idx >= 2:  # Only show 3 in bottom row
            break
    
    plt.suptitle(title, fontsize=18, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test with sample data
    print("Visualization module loaded successfully.")
    print("Run the main simulation to generate plots.")
