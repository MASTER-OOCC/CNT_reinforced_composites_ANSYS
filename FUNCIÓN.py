"""

Variables de entrada:

E_m     : Módulo de Elasticidad de la Matriz (GPa)
mu_m    : Coeficiente de Poisson de la Matriz

E_p     : Módulo de Elasticidad de la partícula (GPa)
mu_m    : Coeficiente de Poisson de la Partícula
D_p     : diámetro de la partícula (micrones)
f_p     : Fracción de volumen de la partícula (0-1)

E_i     : Módulo de Elasticidad del Interface (GPa)
mu_i    : Coeficiente de Poisson del Interface

G_w     : Módulo de Corte del Agua (GPa)
K_w     : Módulo de Compresibilidad del Agua (GPa)
S       : Saturación del Agua (0-1)

G_v     : Módulo de Corte del Vacío (GPa)
K_v     : Módulo de Compresibilidad del Vacío (GPa)
f_v     : Fracción de volumen del vacío (0-1)

Kappa_p : Relación de aspecto de la partícula
Kappa_v : Relación de aspecto del vacío

inter   : 0 -> No hay interface, 1 -> Interface
type_int: 1 -> Soft, 2 -> Hard


"""

globals().clear()#Limpia las variables

import numpy as np
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def f_int_mat(M_to_int, phi, theta):
        return simpson(simpson(M_to_int, theta, axis=-1), phi, axis=-1)


def f_isotropic(Ey, mu):
    # Cálculo del módulo de corte (G) a partir de Ey y Poisson
    G = Ey / (2 * (1 + mu)) #en teoria esto se calculó antes por lo que podría indicarse como variable de entrada pero ya está así
    
    # Inicialización de la matriz de flexibilidad S (6x6)
    S = np.zeros((6, 6))
    
    # Asignación de los elementos de la matriz S relacionados con el módulo de Young
    S[0, 0] = 1 / Ey  # Componente S11
    S[1, 1] = 1 / Ey  # Componente S22
    S[2, 2] = 1 / Ey  # Componente S33
    
    # Asignación de los elementos de la matriz S relacionados con el módulo de corte
    S[3, 3] = 1 / G   # Componente S44
    S[4, 4] = 1 / G   # Componente S55
    S[5, 5] = 1 / G   # Componente S66
    
    # Asignación de los elementos de la matriz S que representan los efectos de Poisson
    S[0, 1] = -mu / Ey  # Componente S12
    S[0, 2] = S[0, 1]  # Componente S13
    S[1, 2] = S[0, 1]  # Componente S23
    S[1, 0] = S[0, 1]  # Componente S21
    S[2, 0] = S[0, 2]  # Componente S31
    S[2, 1] = S[1, 2]  # Componente S32
    
    # Cálculo del tensor constitutivo C como la inversa de la matriz de flexibilidad S
    C = np.linalg.inv(S)  
    
    return C


def f_compute_eshelby_integrals(a):

    def FD(s): return np.sqrt((a[0]**2 + s) * (a[1]**2 + s) * (a[2]**2 + s))
    
    def F(i, s): return 1 / ((a[i]**2 + s) * FD(s))
    
    def Fij(i, j, s): return 1 / ((a[i]**2 + s) * (a[j]**2 + s) * FD(s))
    
    c = 2 * np.pi * np.prod(a)
    
    A = np.zeros(3)
    A[0] = c * quad(lambda s: F(0, s), 0, np.inf)[0]
    A[1] = c * quad(lambda s: F(1, s), 0, np.inf)[0]
    A[2] = 4 * np.pi - A[0] - A[1]  # A(3) por complementariedad
    
    B = np.zeros((3, 3))
    idx_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for i, j in idx_pairs:
        B[i, j] = c * quad(lambda s: Fij(i, j, s), 0, np.inf)[0]
        if i != j:
            B[j, i] = B[i, j]  # Simetría
    
    return A, B

def f_eshelby_int(a, nu):

    A, B = f_compute_eshelby_integrals(a)
    
    a2 = np.array(a)**2  # Cuadrados de las razones de aspecto
    c_offd = 1 / (8 * np.pi * (1 - nu))
    c_diag = 3 * c_offd
    c_maj = (1 - 2 * nu) * c_offd
    S = np.zeros((6, 6))
    
    # Componentes diagonales
    for i in range(3):
        S[i, i] = c_maj * A[i] + c_diag * a2[i] * B[i, i]
    
    # Componentes fuera de la diagonal
    idx_pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in idx_pairs:
        S[i, j] = -c_maj * A[i] + c_offd * a2[j] * B[i, j]
        S[j, i] = S[i, j]  # Simetría
    
    # Componentes de corte (4,4), (5,5), (6,6)
    c_offd_half = c_offd / 2
    c_maj_half = c_maj / 2
    S[3, 3] = (a2[1] + a2[2]) * B[1, 2] * c_offd_half + c_maj_half * (A[1] + A[2])
    S[4, 4] = (a2[0] + a2[2]) * B[0, 2] * c_offd_half + c_maj_half * (A[0] + A[2])
    S[5, 5] = (a2[0] + a2[1]) * B[0, 1] * c_offd_half + c_maj_half * (A[0] + A[1])
    
    # Ajustar términos de corte
    S[3:6, :] *= 2
    
    return S

def f_compute_engineering_constants_sqrt2(C):
    E = np.zeros(6)
    nu = np.zeros(6)
    
    if np.linalg.matrix_rank(C) < 6:
        return E, nu
    
    # Compute compliance matrix:
    S = np.linalg.inv(C)  # Inverse of a symmetric matrix is symmetric
    
    for i in range(6):
        E[i] = 1 / S[i, i]
    
    # Define nu: [nu_23, nu_13, nu_12, nu_32, nu_31, nu_21]
    # Better accuracy for larger Poisson ratio:
    nuscale = -np.diag(E[:3]) @ S[:3, :3].T
    
    nu[0] = nuscale[1, 2]
    nu[1] = nuscale[0, 2]
    nu[2] = nuscale[0, 1]
    nu[3] = nuscale[2, 1]
    nu[4] = nuscale[2, 0]
    nu[5] = nuscale[1, 0]
    
    return E, nu

def f_rottensor(Beta, Alpha, Psi, order, A, val):
    # Matrices W y invW
    W = np.diag([1, 1, 1, 2, 2, 2])
    invW = np.diag([1, 1, 1, 0.5, 0.5, 0.5])
    
    # Obtener matrices de rotación
    T1, T2 = rotmatrix(Beta, Alpha, Psi, order)
    
    # Cálculo del tensor rotado
    if val == 1:  # Tensor de rigidez
        A_rot = np.linalg.inv(T2.T) @ A @ T1.T
    elif val == 2:  # Tensor de cumplimiento
        A_rot = np.linalg.inv(T1.T) @ A @ T2.T
    elif val == 3:  # Tensor de Eshelby
        A_rot = T2 @ A @ np.linalg.inv(T2)
    elif val == 4:  # Tensor B
        A_rot = T1 @ A @ np.linalg.inv(T1)
    else:
        raise ValueError("El valor de 'val' debe estar entre 1 y 4")
    
    return A_rot

def rotmatrix(Beta, Alpha, Psi, order):
    # Convertir a grados
    Beta = np.degrees(Beta)
    Alpha = np.degrees(Alpha)
    Psi = np.degrees(Psi)
    
    # Matrices de rotación
    R = np.zeros((3, 3, 3))
    R[0] = np.array([[1, 0, 0], [0, np.cos(np.radians(Beta)), np.sin(np.radians(Beta))],
                      [0, -np.sin(np.radians(Beta)), np.cos(np.radians(Beta))]])
    R[1] = np.array([[np.cos(np.radians(Alpha)), 0, -np.sin(np.radians(Alpha))], [0, 1, 0],
                      [np.sin(np.radians(Alpha)), 0, np.cos(np.radians(Alpha))]])
    R[2] = np.array([[np.cos(np.radians(Psi)), np.sin(np.radians(Psi)), 0],
                      [-np.sin(np.radians(Psi)), np.cos(np.radians(Psi)), 0], [0, 0, 1]])
    
    # Calcular matriz de rotación final
    w = R[order[2] - 1] @ R[order[1] - 1] @ R[order[0] - 1]
    
    # Construcción de matrices K1, K2, K3, K4
    K1 = w**2
    K2 = np.array([[w[0,1]*w[0,2], w[0,2]*w[0,0], w[0,0]*w[0,1]],
                    [w[1,1]*w[1,2], w[1,2]*w[1,0], w[1,0]*w[1,1]],
                    [w[2,1]*w[2,2], w[2,2]*w[2,0], w[2,0]*w[2,1]]])
    K3 = np.array([[w[1,0]*w[2,0], w[1,1]*w[2,1], w[1,2]*w[2,2]],
                    [w[2,0]*w[0,0], w[2,1]*w[0,1], w[2,2]*w[0,2]],
                    [w[0,0]*w[1,0], w[0,1]*w[1,1], w[0,2]*w[1,2]]])
    K4 = np.array([[w[1,1]*w[2,2] + w[1,2]*w[2,1], w[1,2]*w[2,0] + w[1,0]*w[2,2], w[1,0]*w[2,1] + w[1,1]*w[2,0]],
                    [w[2,1]*w[0,2] + w[2,2]*w[0,1], w[2,2]*w[0,0] + w[2,0]*w[0,2], w[2,0]*w[0,1] + w[2,1]*w[0,0]],
                    [w[0,1]*w[1,2] + w[0,2]*w[1,1], w[0,2]*w[1,0] + w[0,0]*w[1,2], w[0,0]*w[1,1] + w[0,1]*w[1,0]]])
    
    # Matrices de transformación
    gchange3D1 = np.block([[K1, 2 * K2], [K3, K4]])  # Tensiones
    gchange3D2 = np.block([[K1, K2], [2 * K3, K4]])  # Deformaciones
    
    return gchange3D1, gchange3D2

def f_orientational_average(number_theta, number_phi, theta_max, phi_max, A, k, val):
    """
    Compute orientational average of tensor A.

    Parameters:
    number_theta : int
        Number of steps of integration for the variable theta.
    number_phi : int
        Number of steps of integration for the variable phi.
    theta_max : float
        Maximum value for variable theta.
    phi_max : float
        Maximum value for variable phi.
    A : numpy.ndarray
        Tensor to be averaged.
    k : float
        Parameter of the ODF according to Odegard et al.
    val : any
        Additional parameter for rottensor function.

    Returns:
    numpy.ndarray
        Orientational average of input tensor A.
    """

    if k > 1_000_000:  # Perfectly aligned case
        return A

    theta = np.linspace(0, theta_max, number_theta + 1)
    phi = np.linspace(0, phi_max, number_phi + 1)

    # Orientation Distribution Function (ODF)
    caso = 3  # General case
    ODF = np.zeros_like(theta)

    if caso == 1:
        ODF[:] = 1  # Random orientation
    elif caso == 2:
        ODF[0] = 1  # Aligned
    else:
        ODF = np.exp(-k * theta**2)  # General case

    # Initialize matrices
    A_phi_theta = np.zeros((6, 6, len(phi), len(theta)))
    denom_phi_theta = np.zeros((1, 1, len(phi), len(theta)))
    A_to_int = np.zeros_like(A_phi_theta)

    for t, theta_loop in enumerate(theta):
        for s, phi_loop in enumerate(phi):
            A_phi_theta[:, :, s, t] = f_rottensor(theta_loop, phi_loop, 0, [3, 1, 2], A, val)
        
        denom_phi_theta[0, 0, :, t] = ODF[t] * np.sin(theta_loop)
        A_to_int[:, :, :, t] = A_phi_theta[:, :, :, t] * (ODF[t] * np.sin(theta_loop))

    # Perform double integration
    denom = f_int_mat(denom_phi_theta, phi, theta)
    Amed = f_int_mat(A_to_int, phi, theta) / denom

    return Amed

def f_ellipsoidalinter_random(Cp, Ci, Cm, Si, Sp, vp, vi):
    # Parámetros de integración
    number_theta = 100
    number_phi = 100
    
    theta_max = np.pi / 2
    phi_max = 2 * np.pi
    
    if vp == 0:
        return Cm
    
    Id = np.eye(6)
    vm = 1 - vp - vi
    
    psiP = np.linalg.solve(Cp - Cm, Cm)
    psiI = np.linalg.solve(Ci - Cm, Cm)
    
    dS = Si - Sp
    
    Mp = -np.linalg.inv(Sp + psiP + dS @ np.linalg.inv(Sp + psiP - (vp / vi) * dS) @ (Sp + psiI - (vp / vi) * dS))
    Mi = -np.linalg.inv(dS + (Sp + psiP) @ np.linalg.inv(Sp + psiP - (vp / vi) * dS) @ (Sp + psiI - (vp / vi) * dS))
    
    Adil1 = Id + Sp @ Mp + dS @ Mi
    Adil2 = Id + Si @ Mi + (vp / vi) * dS @ (Mp - Mi)
    
    Bdil1 = Cp @ Adil1 @ np.linalg.inv(Cm)
    Bdil2 = Ci @ Adil2 @ np.linalg.inv(Cm)
    
    Amed1 = f_orientational_average(number_theta, number_phi, theta_max, phi_max, Adil1, 0, 3)
    Amed2 = f_orientational_average(number_theta, number_phi, theta_max, phi_max, Adil2, 0, 3)
    
    Bmed1 = f_orientational_average(number_theta, number_phi, theta_max, phi_max, Bdil1, 0, 4)
    Bmed2 = f_orientational_average(number_theta, number_phi, theta_max, phi_max, Bdil2, 0, 4)
    
    Ceff = (vm * Cm + vi * Bmed2 @ Cm + vp * Bmed1 @ Cm) @ np.linalg.inv(vm * Id + vi * Amed2 + vp * Amed1)
    
    return Ceff



def f_ellipsoidal_nointer_random(Ci, Cm, Seby, vi):
    vm = 1 - vi
    
    number_theta = 20
    number_phi = 20
    
    theta_max = np.pi / 2
    phi_max = 2 * np.pi
    
    Id = np.eye(6)
    A = np.linalg.inv(Id + Seby @ np.linalg.inv(Cm) @ (Ci - Cm))
    B = Ci @ A @ np.linalg.inv(Cm)
    
    Amed = f_orientational_average(number_theta, number_phi, theta_max, phi_max, A, 0, 3)
    Bmed = f_orientational_average(number_theta, number_phi, theta_max, phi_max, B, 0, 4)
    
    Ceff = (vm * Cm + vi * Bmed @ Cm) @ np.linalg.inv(vm * Id + vi * Amed)
    
    return Ceff



def f_MTinterparam(t, E_i, vp, E_m, mu_m, E_p, mu_p, mu_i, Kappa, d_p, typeinter):
    """
    Calcula el módulo efectivo Eeff basado en el modelo de interfase.

    Parámetros:
    t : float -> Espesor de la interfase.
    Ei : float -> Módulo de la interfase.
    vp : array -> Fracción volumétrica de partículas.
    Em : float -> Módulo de la matriz.
    poism : float -> Coeficiente de Poisson de la matriz.
    Ep : float -> Módulo de las partículas.
    poisp : float -> Coeficiente de Poisson de las partículas.
    poisi : float -> Coeficiente de Poisson de la interfase.
    Kappa : float -> Relación de aspecto de la interfase.
    d_p : float -> Diámetro equivalente de las partículas.
    typeinter : int -> Tipo de interfase (1 -> Soft, 2 -> Hard).

    Retorna:
    Eeff : array -> Módulo efectivo calculado.
    """

    if np.isscalar(vp):  # Asegura que `vp` sea un array
        vp = np.array([vp])
        
    Cm = f_isotropic(E_m, mu_m)  # Propiedades de la matriz
    Cp = f_isotropic(E_p, mu_p)  # Propiedades de las partículas
    Ci = f_isotropic(E_i, mu_i)  # Propiedades de la interfase

    Sp = f_eshelby_int([1, Kappa, 1], mu_m)
    Si = f_eshelby_int([1, Kappa, 1], mu_m)

    lambda_ratio = t / d_p  # Relación de espesor con diámetro

    vicont = np.zeros(len(vp))
    Eeff = np.zeros(len(vp))

    for i, vp_i in enumerate(vp):
        if typeinter == 1:
            vicont[i] = f_softinterphase(Kappa, lambda_ratio, vp_i)
        else:
            vicont[i] = f_hardinterphase(Kappa, lambda_ratio, vp_i)

        vi = vicont[i]

        # Homogeneización
        Ceff = f_ellipsoidalinter_random(Cp, Ci, Cm, Si, Sp, vp_i, vi)
        E, nu = f_compute_engineering_constants_sqrt2(Ceff)

        Eeff[i] = E[0]

    return Eeff


def f_softinterphase(Kappa, Lambda, vp):
    """
    Calculate the effective volume fraction of the soft interphase.

    Parameters:
    Kappa (float): Aspect ratio of particles.
    Lambda (float): Geometric size factor t/d_p.
    vp (float): Volume fraction of particles.

    Returns:
    float: Effective volume fraction of the interphase.
    """
    # Compute eta based on Kappa
    eta = 1 / Kappa if Kappa > 1 else Kappa

    # Compute phi
    phi = np.arccos(eta)

    # Compute nkappa based on Kappa
    if Kappa < 1:
        nkappa = (2 * Kappa ** (2/3) * np.sin(phi)) / (np.sin(phi) + Kappa**2 * np.arctanh(np.sin(phi)))
    elif Kappa == 1:
        nkappa = 1
    else:
        nkappa = (2 * Kappa ** (2/3) * np.tan(phi)) / (np.tan(phi) + Kappa**2 * phi)

    # m can be 0, 2, or 3
    m = 0

    # Compute vi
    vi = (1 - vp) * (1 - np.exp(-(6 * vp / (1 - vp)) * 
                                 (Lambda / nkappa + 
                                  (2 + 3 * vp / (nkappa**2 * (1 - vp))) * Lambda**2 + 
                                  (4 / 3) * (1 + 3 * vp / (nkappa * (1 - vp)**2) + 
                                             m * vp**2 / (nkappa**3 * (1 - vp)**2)) * Lambda**3)))
    
    return vi


def f_hardinterphase(Kappa, Lambda, vp):
    # Kappa -> Aspect ratio of particles
    # Lambda -> Geometric size factor t/d_p
    
    # Sphericity of an ellipsoidal particle with aspect ratio Kappa
    # Ratio of the surface area between a sphere and the ellipsoid with the same volume
    eta = 1 / Kappa if Kappa > 1 else Kappa
    phi = np.arccos(eta)
    
    if Kappa < 1:
        nkappa = (2 * Kappa**(2/3) * np.sin(phi)) / (np.sin(phi) + Kappa**2 * np.arctanh(np.sin(phi)))
    elif Kappa == 1:
        nkappa = 1
    else:
        nkappa = (2 * Kappa**(2/3) * np.tan(phi)) / (np.tan(phi) + Kappa**2 * phi)
    
    if Kappa > 1:
        omega = Kappa**(2/3) + np.log(Kappa + np.sqrt(Kappa**2 - 1)) / (Kappa**(1/3) * np.sqrt(Kappa**2 - 1))
    else:
        omega = Kappa**(2/3) + np.arcsin(np.sqrt(1 - Kappa**2)) / (Kappa**(1/3) * np.sqrt(1 - Kappa**2))
    
    if Kappa == 1:
        vi = vp * ((1 + 2 * Lambda)**3 - 1)
    else:
        vi = 6 * vp * (Lambda / nkappa + omega * Lambda**2 + 4 * Lambda**3 / 3)
    
    return vi





def MoriTanaka(
    f_p, f_v,
    E_m, mu_m,
    E_p, mu_p, D_p,
    E_i, mu_i, t_i,
    G_w, K_w, S,
    G_v, K_v,
    Kappa_p, Kappa_v,
    inter, type_int
):
    # Propiedades mecánicas de matriz y partícula
    C_m = f_isotropic(E_m, mu_m)
    C_p = f_isotropic(E_p, mu_p)
    S_p = f_eshelby_int([1, Kappa_p, 1], mu_m)

    # Propiedades del vacío con agua (medio poroso)
    K_agua_vacio = 1 / (S / K_w + (1 - S) / K_v)
    G_agua_vacio = G_v
    mu_vacio = (3 - 2 * G_agua_vacio / K_agua_vacio) / (2 * (3 + G_agua_vacio / K_agua_vacio))
    E_vacio = 2 * (1 + mu_vacio) * G_agua_vacio
    C_v = f_isotropic(E_vacio, mu_vacio)

    if inter == 1:
        # Cálculo con interfase
        lambda_ = t_i / D_p  # conversión micrones a milímetros si es necesario
        f_i = f_softinterphase(Kappa_p, lambda_, f_p) if type_int == 1 else f_hardinterphase(Kappa_p, lambda_, f_p)

        C_i = f_isotropic(E_i, mu_i)
        S_i = f_eshelby_int([1, Kappa_p, 1], mu_m)

        C_eff = f_ellipsoidalinter_random(C_p, C_i, C_m, S_i, S_p, f_p, f_i)
    else:
        # Cálculo sin interfase
        C_eff = f_ellipsoidal_nointer_random(C_p, C_m, S_p, f_p)

    E_, nu_ = f_compute_engineering_constants_sqrt2(C_eff)

    S_void = f_eshelby_int([1, Kappa_v, 1], nu_[0])
    C_eff_poro = f_ellipsoidal_nointer_random(C_v, C_eff, S_void, f_v)
    E_final, _ = f_compute_engineering_constants_sqrt2(C_eff_poro)
    return E_final[0], E_final[1], E_final[2]




# Supongamos que todas las funciones están ya definidas y disponibles

# Valores de entrada
E_m = 23.6077      # GPa
mu_m = 0.2048

E_p = 80           # GPa
mu_p = 0.21
D_p = 450          # micrones
f_p = 0.25         # Fracción de partícula

E_i = 9.4431       # GPa
mu_i = 0.2048
t_i = 0.1          # micrones #######################

G_w = 0.00001      # GPa
K_w = 2.3          # GPa
S = 1              # Saturación

G_v = 0.00001      # GPa
K_v = 0.013        # GPa
f_v = 0.1          # Fracción de poro

Kappa_p = 1        # Relación de aspecto partícula
Kappa_v = 1        # Relación de aspecto del vacío

inter = 1          # 1 -> considerar interphase
type_int = 1       # 1 -> Soft interphase

import time

# Inicio del temporizador
start_time = time.time()

# Cálculo
E1, E2, E3 = MoriTanaka(
    f_p, f_v,
    E_m, mu_m,
    E_p, mu_p, D_p,
    E_i, mu_i, t_i,
    G_w, K_w, S,
    G_v, K_v,
    Kappa_p, Kappa_v,
    inter, type_int
)

# Fin del temporizador
end_time = time.time()

# Tiempo transcurrido
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time:.6f} segundos")

# Mostrar resultados
print("Módulos de Young efectivos (GPa):")
print("E1 =", E1)
print("E2 =", E2)
print("E3 =", E3)
