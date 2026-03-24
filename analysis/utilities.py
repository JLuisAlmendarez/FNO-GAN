import numpy as np

def espectro_angular(omega, dx=1.0, dy=1.0):
    Nx, Ny = omega.shape

    # FFT 2D normalizada
    omega_hat = np.fft.fft2(omega) / (Nx * Ny)

    # Frecuencias en cada dirección
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # [rad/m]
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi

    # Malla de frecuencias 2D
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)  # magnitud |k|

    # Potencia espectral
    poder = np.abs(omega_hat)**2

    # Agrupar por k discreto (anillos)
    k_vals = np.unique(np.round(K, 6))
    E_k = np.zeros(len(k_vals))

    for idx, k in enumerate(k_vals):
        mask = np.isclose(K, k, atol=1e-6)
        E_k[idx] = np.sum(poder[mask])

    return k_vals, E_k

def k_mean(k_vals, E_k):
    # Ignorar k=0 (modo DC, sin física relevante)
    mask = k_vals > 0
    k_m = np.sum(k_vals[mask] * E_k[mask]) / np.sum(E_k[mask])
    return k_m

