# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:06:26 2025

2 Scatters simulation

@author: ZAQ
"""

plot=True

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

# def simulate_source_spectrum(
#         L1 = 1015e-9,
#         L2 = 1090e-9,
#         N = 1028,
#         plot=False):
    
#     L_arr = np.linspace(L1,L2,N)
#     k1 = 2*np.pi/L_arr[0]
#     k2 = 2*np.pi/L_arr[-1]
#     k_arr = np.linspace(k1,k2,N)
#     k0 = (k1+k2)/2
    
#     half_width_k = 2e5
#     S_arr = np.ones(k_arr.shape)*1e-16 # small value to avoid div/0 errors
#     S_arr[np.where(np.abs(k_arr-k0)<half_width_k)] = 1
    
#     # sampling interval in k
#     dk = k_arr[1]-k_arr[0]
    
#     if plot:
#         plt.figure()
#         plt.plot(k_arr,S_arr)
#         plt.xlabel('k(rad/m)')
#         plt.ylabel('power (ADU)')
#         plt.title('source spectrum')
#     return S_arr, k_arr

# OCT function (Modified Eqn. 1 above)
# def OCT(z_S, R_S, S_arr, k_arr, z_R=0.0, R_R=1.0):
#     delta_z = z_S - z_R
#     # OCT_signal = S_arr*R_R + S_arr*R_S + 2*S_arr*np.sqrt(R_R*R_S)*np.real(np.cos(2*k_arr*delta_z))
#     OCT_signal = 2*S_arr*np.sqrt(R_R*R_S)*np.real(np.cos(2*k_arr*delta_z)) # No DC
#     return OCT_signal

# Complex OCT 
def OCT_complex(z_S, R_S, S_arr, k_arr, z_R=0.0, R_R=1.0):
    delta_z = z_S - z_R
    # Complex interference term (analytic in k)
    return 2*S_arr*np.sqrt(R_R*R_S) * np.exp(1j * 2 * k_arr * delta_z)

# def reconstruct_ascan_from_spectrum(spectrum):
#     """
#     Reconstruct complex A-scan from k-domain spectrum.
#     We assume uniform sampling in k. We use ifft with shift for centered spectra.
#     """
#     a = np.fft.fftshift(np.fft.ifft(spectrum))
#     return a

# No shift
def reconstruct_ascan_from_spectrum(spectrum):
    # If k_arr is already in ascending order with DC at index 0:
    return np.fft.ifft(spectrum)        # no shift
    # Only use fftshift/ifftshift if you also shift the spectrum to be centered.

def peak_phase(ascan):
    """Return index of the magnitude peak and its complex phase (in radians)."""
    mag = np.abs(ascan)
    idx = int(np.argmax(mag))
    ph = np.angle(ascan[idx])
    return idx, ph

def theoretical_phase_change(delta_z, lambda0):
    """Expected phase change for a sub-wavelength axial shift in OCT: Δφ ≈ 4π Δz / λ0."""
    return 4.0 * np.pi * delta_z / lambda0

def run_simulation(
    lambda0=1064e-9,          # central wavelength [m]
    fwhm_lambda=50e-9,       # source FWHM bandwidth [m]
    n_samples=1028,          # number of k-samples
    zA=1e-4,               # depth of scatterer A [m]
    zB=1.4e-4,               # initial depth of scatterer B [m]
    ampA=1.0e-2+0.0j,           # amplitude of A
    ampB=0.8e-2+0.0j,           # amplitude of B
    n_steps=10,              # number of incremental shifts to apply to B
    shift_per_step_fraction=1/50.0,  # each step is λ0 * this fraction (e.g., 1/50 -> λ/50)
    seed=None,
):
    """
    Execute the two-scatterer simulation and return a results dict.
    - Build uniform k-grid around k0 with Gaussian envelope.
    - Simulate spectra and reconstruct A-scans.
    - Compute phase(A) - phase(B) at their magnitude peaks.
    - Incrementally shift B and recompute, tracking phase evolution.
    """
    if seed is not None:
        np.random.seed(seed)

    # # Central wavenumber and spectral width (convert FWHM in wavelength to std in k-domain approximately)
    # k0 = 2.0 * np.pi / lambda0

    # # Approximate mapping: if source is Gaussian in wavelength with FWHM Δλ,
    # # then in k it's also roughly Gaussian with std sigma_k ≈ (2π/λ^2) * sigma_λ, where sigma_λ = FWHM / (2*sqrt(2*ln2))
    # sigma_lambda = fwhm_lambda / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # sigma_k = (2.0 * np.pi / (lambda0 ** 2)) * sigma_lambda

    # # k-grid (uniform around k0)
    # dk = 4.0 * sigma_k / n_samples  # span ~±2σ in k
    
    L1 = lambda0 - fwhm_lambda
    L2 = lambda0 + fwhm_lambda
    L_arr = np.linspace(L1,L2,n_samples)
    k1 = 2*np.pi/L_arr[0]
    k2 = 2*np.pi/L_arr[-1]
    k_arr = np.linspace(k1,k2,n_samples)
    k0 = (k1+k2)/2 
    
    half_width_k = 2e5
    S_arr = np.ones(k_arr.shape)*1e-16 # small value to avoid div/0 errors
    S_arr[np.where(np.abs(k_arr-k0)<half_width_k)] = 1
    
    # sampling interval in k
    dk = k_arr[1]-k_arr[0]
    
    # Helper to get phases from A and B
    def phases_for_positions(sA, sB):
        aA = reconstruct_ascan_from_spectrum(sA)
        aB = reconstruct_ascan_from_spectrum(sB)
        idxA, phiA = peak_phase(aA)
        idxB, phiB = peak_phase(aB)
        return aA, aB, idxA, idxB, phiA, phiB
    
    def plot_result(sA, sB, aA, aB, idxA, idxB, phiA, phiB, zA_cur, zB_cur, idx=0):
        save_dir = r"E:\Registration\Simulation\plots_minutes"
        os.makedirs(save_dir, exist_ok=True)
        
        # OCT Signal Spectrum
        fig, ax = plt.subplots(2, 2, figsize=(10,6))
        
        ax[0,0].plot(sA)
        ax[0,0].set_title(f"Scatter A OCT Signal Spectrum @ zA = {zA_cur*1e6:.0f}um", fontsize=10)
        
        ax[0,1].plot(sB)
        ax[0,1].set_title(f"Scatter B OCT Signal Spectrum @ zB = {zB_cur*1e6:.0f}um", fontsize=10)
        
        # # OCT Signal Spectrum
        # fig, ax = plt.subplots(2, 1)
        
        ax[1,0].plot(aA)
        ax[1,0].set_title(f"Scatter A Ascan, pk @ {idxA}\nphase = {phiA:.2e}", fontsize=10)
        
        ax[1,1].plot(aB)
        ax[1,1].set_title(f"Scatter B Ascan, pk @ {idxB}\nphase = {phiB:.2e}", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(rf"{save_dir}/Ascan_{idx}.png")
        plt.close(fig)
        return
    
    def windowed_complex(ascan, z0, hw=2, w=None):
        sl = slice(z0-hw, z0+hw+1)
        if w is None:
            w = np.hanning(2*hw+1)
        return np.sum(ascan[sl] * w)
    
    # sA = OCT(zA, a`mpA, S_arr, k_arr)
    # sB0 = OCT(zB, ampB, S_arr, k_arr)
    
    # # Initial phases
    # aA0, aB0, idxA0, idxB0, phiA0, phiB0 = phases_for_positions(sA, sB0)
    
    # if idxA0 > n_samples/2:
    #     idxA0 = n_samples - idxA0
    # if idxB0 > n_samples/2:
    #     idxB0 = n_samples - idxB0
        
    # base_phase_diff = np.angle(np.exp(1j * (phiA0 - phiB0)))  # wrapped into (-π, π)
    # plot_result(sA, sB0, aA0, aB0, idxA0, idxB0, phiA0, phiB0, zA, zB)
    
    # build once from first step to lock indices
    expected_offset = 0
    spec0 = OCT_complex(zA, ampA, S_arr, k_arr) + OCT_complex(zB, ampB, S_arr, k_arr)
    a0 = np.fft.ifft(spec0)
    zA_idx = np.argmax(np.abs(a0))                    # or pick known index for A
    zB_idx = np.argmax(np.abs(a0)) + expected_offset  # or locate B near expected
    
    # Perform incremental sub-wavelength shifts of B
    delta_per_step = lambda0 * shift_per_step_fraction  # meters per step
    shifts = []
    phase_diffs = []
    theo_phase = []
    phiA = []
    phiB = []
    zB_all = []
    sB_all = []

    # for i in range(n_steps + 1):
    #     zB_i = zB + i * delta_per_step
    #     sB_i = OCT(zB_i, ampB, S_arr, k_arr)
    #     zB_all.append(zB_i)
    #     # phiA_i, phiB_i = phases_for_positions(zA, zB_i)
    #     aAi, aBi, idxAi, idxBi, phiAi, phiBi = phases_for_positions(sA, sB_i)
    #     if idxAi > n_samples/2:
    #         idxAi = n_samples - idxAi
    #     if idxBi > n_samples/2:
    #         idxBi = n_samples - idxBi
    #     # plot_result(sA, sB_i, aAi, aBi, idxAi, idxBi, phiAi, phiBi, zA, zB_i, i)
    #     # phase difference between A and B' at their respective peaks
    #     dphi = np.angle(np.exp(1j * (phiAi - phiBi)))  # wrap to (-π, π]
    #     # dphi = np.angle(np.exp(1j * (phiA0 - phiBi)))  # wrap to (-π, π]
    #     shifts.append(i * delta_per_step)
    #     phiA.append(np.array(phiAi))
    #     phiB.append(np.array(phiBi))
    #     sB_all.append(np.array(sB_i))
    #     phase_diffs.append(dphi)
    #     theo_phase.append(np.mod(base_phase_diff - theoretical_phase_change(i * delta_per_step, lambda0) + np.pi, 2*np.pi) - np.pi)
        
    for i in range(n_steps+1):
        zB_i = zB + i*delta_per_step
        spec = OCT_complex(zA, ampA, S_arr, k_arr) + OCT_complex(zB_i, ampB, S_arr, k_arr)
        a = np.fft.ifft(spec)
    
        cA = windowed_complex(a, zA_idx, hw=2)
        cB = windowed_complex(a, zB_idx, hw=2)
        dphi = np.angle(cB * np.conj(cA))
        phase_diffs.append(dphi)
        shifts.append(i * delta_per_step)
        
    # Unwrap measured phase diffs for analysis/plotting
    phase_diffs_unwrapped = np.unwrap(phase_diffs)

    results = {
        "lambda0": lambda0,
        "fwhm_lambda": fwhm_lambda,
        "n_samples": n_samples,
        "k0": k0,
        "dk": dk,
        "zA": zA,
        "phiA": phiA,
        "zB": zB_all,
        "phiB": phiB,
        "sB": sB_all,
        "delta_per_step": delta_per_step,
        "shifts": np.array(shifts),
        "phase_diffs_wrapped": np.array(phase_diffs),
        "phase_diffs_unwrapped": np.array(phase_diffs_unwrapped),
        "theoretical_phase_wrapped": np.array(theo_phase),
        # "base_phase_diff": base_phase_diff,
    }
    return results

# def save_results(results, csv_path):
#     df = pd.DataFrame({
#         "shift_m": results["shifts"],
#         "phase_diff_wrapped_rad": results["phase_diffs_wrapped"],
#         "phase_diff_unwrapped_rad": results["phase_diffs_unwrapped"],
#         "theoretical_wrapped_rad": results["theoretical_phase_wrapped"],
#     })
#     df.to_csv(csv_path, index=False)
#     return df

def save_results(results, csv_path):
    df = pd.DataFrame({
        "zA": results["zA"],
        "zB": results["zB"],
        "shift_m": results["shifts"],
        "phase_diff_wrapped_rad": results["phase_diffs_wrapped"],
        "phase_diff_unwrapped_rad": results["phase_diffs_unwrapped"],
    })
    df.to_csv(csv_path, index=False)
    return df

def detect_jumps(x, y, k_sigma=3.5, window=3, min_drop=3.0, detrend=True):
    """
    x, y: 1D arrays (same length)
    k_sigma: outlier threshold for single-step changes (in MAD-based sigmas)
    window: number of points over which to look for multi-step drops
    min_drop: total drop y[i] - y[i+window] required to flag a multi-step jump
    detrend: remove global linear trend before differencing (helps on sloped data)
    """
    x = np.asarray(x); y = np.asarray(y)

    # Optional detrend so we detect deviations from the overall slope
    if detrend:
        p = np.polyfit(x, y, 1)
        y_use = y - np.polyval(p, x)
    else:
        y_use = y

    # ---------- 1) Single-step jumps ----------
    dy = np.diff(y_use)
    # # robust scale (MAD -> sigma)
    # mad = np.median(np.abs(dy - np.median(dy)))
    # sigma = 1.4826 * mad if mad > 0 else (np.std(dy) + 1e-12)
    # thr = k_sigma * sigma
    thr = 1
    single_step_idx = np.where(np.abs(dy) > thr)[0]        # jump between i and i+1

    # ---------- 2) Multi-step drops over a small window ----------
    multi_spans = []
    if window >= 1:
        for i in range(len(y) - window):
            total_drop = y[i] - y[i + window]
            if total_drop > min_drop:
                # ensure it’s mostly decreasing inside the window
                if np.all(np.diff(y[i:i+window+1]) < 0):
                    multi_spans.append((i, i + window))

    return single_step_idx, multi_spans



def plot_phase_vs_shift(results, show=True, save_path=None):
    shifts_um = results["shifts"] * 1e6
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(shifts_um, results["phase_diffs_unwrapped"], 'ks', label="Measured Δφ (unwrapped)")
    # ax.plot(shifts_um, results["theoretical_phase_wrapped"], linestyle="--", label="Theoretical Δφ (wrapped)")
    ax.set_xlabel("Axial shift of B (µm)")
    ax.set_ylabel("Phase difference Δφ (rad)")
    ax.set_title("Two-scatterer phase difference vs. sub-wavelength axial shift")
    ax.grid(True)
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def main():
    # Default run: 10 steps, each λ/50 shift
    results = run_simulation(
        lambda0=1064e-9,
        fwhm_lambda=50e-9,
        n_samples=1028,
        zA=0.1e-3,
        zB=0.3e-3,
        ampA=1.0e-2+0.0j,
        ampB=0.8e-2+0.0j,
        n_steps=200,
        shift_per_step_fraction=1/50.0,
    )
    
    # # Debug phase wrap
    # phase_diffs = results["phase_diffs_unwrapped"]
    # phase_diffs_rad = np.diff(phase_diffs)
    # idx = np.where(np.abs(phase_diffs_rad) > 2)[0]        # indices where diff > 2
    # target_phase_diffs = []
    # for i in idx:
    #     target_phase_diff = [phase_diffs[i], phase_diffs[i+1]]
    #     target_phase_diffs.append(target_phase_diff)
        
    # print(idx)                         # gives indices of the *first element* in the pair
    
    # # ---- Debug phase possible phase wrap ----
    # x = results["shifts"] * 1e6
    # y_wrapped = results["phase_diffs_wrapped"]
    # y = results["phase_diffs_unwrapped"]
    # phiA = results["phiA"]
    # phiB = results["phiB"]
    # zA = results["zA"]
    # zB = results["zB"]
    # sB = results["sB"]
    
    # #--- Visualize all parameters ---
    # fig, ax = plt.subplots(2,3)
    
    # # ax[0,0].plot(x, zA)
    # # ax[0,0].xlabel("Axial shift of B' (μm)")
    # # ax[0,0].ylabel("zA")
    
    # ax[0,1].plot(x, np.array(zB)*1e4)
    # ax[0,1].set_xlabel("Axial shift of B' (μm)", fontsize=8)
    # ax[0,1].set_ylabel("zB (μm)", fontsize=8)
    
    # ax[1,2].plot(x, y_wrapped)
    # ax[1,2].set_xlabel("Axial shift of B' (μm)", fontsize=8)
    # ax[1,2].set_ylabel("Δφ (wrapped)", fontsize=8)
    
    # ax[1,0].plot(x,phiA)
    # ax[1,0].set_xlabel("Axial shift of B' (μm)", fontsize=8)
    # ax[1,0].set_ylabel("φA (rad)", fontsize=8)
    
    # ax[1,1].plot(x, phiB)
    # ax[1,1].set_xlabel("Axial shift of B' (μm)", fontsize=8)
    # ax[1,1].set_ylabel("φB (rad)", fontsize=8)
    
    # ax[0,2].plot(x, y)
    # ax[0,2].set_xlabel("Axial shift of B' (μm)", fontsize=8)
    # ax[0,2].set_ylabel("Δφ (unwrapped)")
    
    # plt.tight_layout()
    # plt.show()
    
    # # single_idx gives "first element" in the pair
    # # spans give three idx where the big phase diff jump happended.
    # single_idx, spans = detect_jumps(x, y, k_sigma=0.2, window=2, min_drop=2) 

    # # visualize phase difference
    # fig, ax = plt.subplots()
    # data_line, = ax.plot(x, y, 'k.-', label="Δφ between A and B' (unwrapped)")
    # single_scatter = ax.scatter(x[single_idx+1], y[single_idx+1],
    #                             s=80, facecolors='none', edgecolors='r', label='Small phase jump')
    
    # # Add shaded spans
    # for i, j in spans:
    #     ax.axvspan(x[i], x[j], color='orange', alpha=0.25)
    
    # # Create a proxy patch for multi-step legend
    # multi_patch = mpatches.Patch(color='orange', alpha=0.25, label='Big phase drop')
    
    # # Combine all legend handles
    # handles = [data_line, single_scatter, multi_patch]
    # labels = [h.get_label() for h in handles]
    # ax.legend(handles, labels, loc='best')
    
    # ax.set_xlabel("Axial shift of B' (μm)")
    # ax.set_ylabel("Phase difference Δφ (rad)")
    # ax.set_title("Detected jumps in two-scatterer phase difference")
    # plt.show()
    
    # small_phase_jump_idx = []
    # small_phase_jump = []
    
    # for ii in single_idx:
    #     small_phase_jump_idx.append(ii)
    #     small_phase_jump.append(phiB[ii])
    
    
    # Save outputs
    csv_path = r"E:\Registration\Simulation/two_scatterer_results.csv"
    fig_path = r"E:\Registration\Simulation/two_scatterer_phase_vs_shift.png"
    # save_results(results, csv_path)
    plot_phase_vs_shift(results, show=True, save_path=fig_path)

    # Print a short report
    print("Saved CSV:", csv_path)
    print("Saved plot:", fig_path)
    print("Base phase diff (A vs. B) [rad]:", results["base_phase_diff"])
    print("Shift per step [nm]:", results["delta_per_step"] * 1e9)

if __name__ == "__main__":
    main()
