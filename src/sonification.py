#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Audio Sonification Engine for Energy Drift
================================================================================

Project:        Week 1 Project 1: The Audible Integrator
Module:         sonification.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        January 21, 2026
Last Updated:   January 21, 2026

License:        MIT License
================================================================================

This module converts simulation energy values into real-time audio,
creating an "auditory debugger" for physics simulations.

Key Concepts:
- Energy → Pitch: Higher energy = higher pitch
- Energy Stability → Timbre: Unstable energy = distorted sound
- Energy Drift → Volume: Growing energy = louder sound

The goal is to make abstract numerical errors (global truncation error,
energy non-conservation) into a visceral, auditory experience.
"""

import numpy as np
from typing import Optional, Callable, List
from dataclasses import dataclass
import threading
import queue
import time

# Try to import audio libraries
# Note: sounddevice requires PortAudio system library, which may not be available
# on cloud platforms like Streamlit Cloud. We catch both ImportError and OSError.
try:
    import sounddevice as sd
    # Test if PortAudio is actually available by querying devices
    sd.query_devices()
    AUDIO_AVAILABLE = True
except (ImportError, OSError) as e:
    AUDIO_AVAILABLE = False
    sd = None  # Define sd as None for type checking
    # Only print warning if not running in Streamlit Cloud
    import os
    if not os.environ.get('STREAMLIT_SERVER_HEADLESS'):
        print(f"Warning: Audio not available ({type(e).__name__}). Audio output disabled.")
        print("For local audio, install PortAudio: brew install portaudio && pip install sounddevice")


@dataclass
class SonificationConfig:
    """Configuration for energy sonification."""
    
    # Audio settings
    sample_rate: int = 44100  # CD quality
    buffer_size: int = 1024   # Samples per buffer
    
    # Frequency mapping
    base_frequency: float = 220.0      # A3 - base pitch when energy = reference
    frequency_range: float = 440.0     # How much pitch can vary (Hz)
    reference_energy: float = 1.0       # Energy value that maps to base frequency
    
    # Amplitude mapping
    base_amplitude: float = 0.3        # Base volume (0-1)
    max_amplitude: float = 0.8         # Maximum volume
    
    # Distortion settings (for unstable simulations)
    enable_distortion: bool = True
    distortion_threshold: float = 1.5  # Energy ratio that triggers distortion
    
    # Harmonics for richer sound
    num_harmonics: int = 3
    harmonic_decay: float = 0.5        # Each harmonic is this fraction of previous


class EnergyToAudio:
    """
    Converts energy values to audio parameters.
    
    The mapping is designed so that:
    - Stable energy (E ≈ E₀) → Pleasant, stable tone at base frequency
    - Growing energy (E > E₀) → Rising pitch, increasing volume
    - Shrinking energy (E < E₀) → Falling pitch (rarely happens with Euler)
    - Wildly varying energy → Distorted, unpleasant sound
    """
    
    def __init__(self, config: SonificationConfig, initial_energy: float):
        self.config = config
        self.reference_energy = initial_energy
        self.energy_history: List[float] = [initial_energy]
        self.max_history = 100  # For calculating variability
        
    def update_energy(self, energy: float):
        """Record a new energy value."""
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
    
    def energy_to_frequency(self, energy: float) -> float:
        """
        Map energy to frequency.
        
        Uses logarithmic mapping for perceptual linearity:
        - Energy = E₀ → Base frequency
        - Energy = 2×E₀ → Base + range/2
        - Energy = E₀/2 → Base - range/2
        """
        if energy <= 0:
            return self.config.base_frequency
        
        ratio = energy / self.reference_energy
        
        # Logarithmic mapping (each doubling = octave)
        # log2(ratio) gives number of doublings
        log_ratio = np.log2(max(ratio, 0.01))  # Clamp to avoid -inf
        
        # Map to frequency range
        freq_offset = log_ratio * self.config.frequency_range / 2
        frequency = self.config.base_frequency + freq_offset
        
        # Clamp to reasonable range (20 Hz - 4000 Hz)
        return np.clip(frequency, 20.0, 4000.0)
    
    def energy_to_amplitude(self, energy: float) -> float:
        """
        Map energy to amplitude (volume).
        
        Higher energy = louder (to emphasize the problem!)
        """
        ratio = energy / self.reference_energy
        
        # Linear mapping with clipping
        amplitude = self.config.base_amplitude * ratio
        return np.clip(amplitude, 0.05, self.config.max_amplitude)
    
    def get_distortion_amount(self) -> float:
        """
        Calculate distortion based on energy variability.
        
        Stable energy → No distortion
        Wildly fluctuating energy → Heavy distortion
        """
        if not self.config.enable_distortion or len(self.energy_history) < 10:
            return 0.0
        
        # Calculate coefficient of variation (std / mean)
        recent = np.array(self.energy_history[-50:])
        cv = np.std(recent) / (np.mean(recent) + 1e-10)
        
        # Also check absolute drift
        drift = abs(recent[-1] / self.reference_energy - 1.0)
        
        # Combine metrics
        instability = max(cv, drift / 2)
        
        return np.clip(instability, 0.0, 1.0)


class AudioSynthesizer:
    """
    Real-time audio synthesizer for sonification.
    
    Generates audio samples based on current frequency, amplitude,
    and distortion parameters.
    """
    
    def __init__(self, config: SonificationConfig):
        self.config = config
        self.phase = 0.0  # Current phase of the oscillator
        self.current_frequency = config.base_frequency
        self.current_amplitude = config.base_amplitude
        self.current_distortion = 0.0
        self.target_frequency = config.base_frequency
        self.target_amplitude = config.base_amplitude
        
        # Smoothing parameters (to avoid clicks)
        self.freq_smoothing = 0.1
        self.amp_smoothing = 0.1
    
    def set_parameters(self, frequency: float, amplitude: float, distortion: float = 0.0):
        """Set target audio parameters (will smooth towards these)."""
        self.target_frequency = frequency
        self.target_amplitude = amplitude
        self.current_distortion = distortion
    
    def generate_samples(self, num_samples: int) -> np.ndarray:
        """
        Generate audio samples.
        
        Uses additive synthesis with harmonics for a richer tone.
        Applies distortion when energy is unstable.
        """
        samples = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Smooth parameters
            self.current_frequency += self.freq_smoothing * (self.target_frequency - self.current_frequency)
            self.current_amplitude += self.amp_smoothing * (self.target_amplitude - self.current_amplitude)
            
            # Generate base tone with harmonics
            sample = 0.0
            harmonic_amp = 1.0
            
            for h in range(1, self.config.num_harmonics + 1):
                freq = self.current_frequency * h
                sample += harmonic_amp * np.sin(self.phase * h)
                harmonic_amp *= self.config.harmonic_decay
            
            # Apply distortion (soft clipping / waveshaping)
            if self.current_distortion > 0.01:
                # Overdrive-style distortion
                gain = 1.0 + self.current_distortion * 5.0
                sample = np.tanh(sample * gain) / gain
                
                # Add some noise for "broken" sound
                sample += self.current_distortion * 0.1 * (np.random.random() - 0.5)
            
            # Apply amplitude
            sample *= self.current_amplitude
            
            # Update phase
            phase_increment = 2.0 * np.pi * self.current_frequency / self.config.sample_rate
            self.phase += phase_increment
            if self.phase > 2.0 * np.pi:
                self.phase -= 2.0 * np.pi
            
            samples[i] = sample
        
        return samples.astype(np.float32)


class RealtimeAudioEngine:
    """
    Real-time audio streaming engine.
    
    Runs in a separate thread to ensure smooth audio output
    while the simulation runs.
    """
    
    def __init__(self, config: Optional[SonificationConfig] = None):
        self.config = config or SonificationConfig()
        self.synthesizer = AudioSynthesizer(self.config)
        self.energy_mapper: Optional[EnergyToAudio] = None
        
        self.running = False
        self.stream: Optional[sd.OutputStream] = None
        self.energy_queue: queue.Queue = queue.Queue()
        
    def initialize(self, initial_energy: float):
        """Initialize the audio engine with reference energy."""
        self.energy_mapper = EnergyToAudio(self.config, initial_energy)
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio status: {status}")
        
        # Get latest energy value if available
        try:
            while True:
                energy = self.energy_queue.get_nowait()
                if self.energy_mapper:
                    self.energy_mapper.update_energy(energy)
                    freq = self.energy_mapper.energy_to_frequency(energy)
                    amp = self.energy_mapper.energy_to_amplitude(energy)
                    dist = self.energy_mapper.get_distortion_amount()
                    self.synthesizer.set_parameters(freq, amp, dist)
        except queue.Empty:
            pass
        
        # Generate audio
        samples = self.synthesizer.generate_samples(frames)
        outdata[:, 0] = samples
    
    def start(self):
        """Start the audio stream."""
        if not AUDIO_AVAILABLE:
            print("Audio not available - running in silent mode")
            return
        
        self.running = True
        self.stream = sd.OutputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=self.config.buffer_size
        )
        self.stream.start()
    
    def stop(self):
        """Stop the audio stream."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def update_energy(self, energy: float):
        """Update the current energy value (thread-safe)."""
        self.energy_queue.put(energy)


class SilentAudioEngine:
    """
    Fallback audio engine that tracks parameters without producing sound.
    Useful for testing without audio hardware or for visualization only.
    """
    
    def __init__(self, config: Optional[SonificationConfig] = None):
        self.config = config or SonificationConfig()
        self.energy_mapper: Optional[EnergyToAudio] = None
        self.running = False
        
        # Track parameters for visualization
        self.frequency_history: List[float] = []
        self.amplitude_history: List[float] = []
        self.distortion_history: List[float] = []
    
    def initialize(self, initial_energy: float):
        """Initialize with reference energy."""
        self.energy_mapper = EnergyToAudio(self.config, initial_energy)
    
    def start(self):
        """Start tracking."""
        self.running = True
    
    def stop(self):
        """Stop tracking."""
        self.running = False
    
    def update_energy(self, energy: float):
        """Update energy and track audio parameters."""
        if self.energy_mapper:
            self.energy_mapper.update_energy(energy)
            
            freq = self.energy_mapper.energy_to_frequency(energy)
            amp = self.energy_mapper.energy_to_amplitude(energy)
            dist = self.energy_mapper.get_distortion_amount()
            
            self.frequency_history.append(freq)
            self.amplitude_history.append(amp)
            self.distortion_history.append(dist)


def create_audio_engine(config: Optional[SonificationConfig] = None, force_silent: bool = False):
    """
    Factory function to create appropriate audio engine.
    
    Returns RealtimeAudioEngine if audio is available,
    otherwise returns SilentAudioEngine.
    """
    if force_silent or not AUDIO_AVAILABLE:
        return SilentAudioEngine(config)
    return RealtimeAudioEngine(config)


if __name__ == "__main__":
    print("Audio Sonification Engine")
    print("=" * 40)
    
    # Test energy to frequency mapping
    config = SonificationConfig()
    mapper = EnergyToAudio(config, initial_energy=1.0)
    
    print("\nEnergy to Frequency Mapping:")
    print("-" * 30)
    for energy in [0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
        freq = mapper.energy_to_frequency(energy)
        amp = mapper.energy_to_amplitude(energy)
        print(f"E = {energy:.2f} E₀ → f = {freq:.1f} Hz, amp = {amp:.2f}")
