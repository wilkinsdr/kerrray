"""
Prototype of a Mino-time symplectic integrator for Kerr null geodesics.

State: (u, p_u, theta, p_theta, t, phi) with r = 1 + b cosh u, b = sqrt(1 - a^2),
so Delta = b^2 sinh^2 u and p_u = sqrt(Delta) p_r.  Hamiltonian in Mino time
(d lambda_Mino = d lambda / rho^2):

  H_M = 1/2 p_u^2 + 1/2 p_theta^2 + V_r(u) + V_theta(theta)
  V_r     = -P(r)^2 / (2 Delta),          P = (r^2 + a^2) E - a L
  V_theta = (L - a E sin^2 theta)^2 / (2 sin^2 theta)

H_M = 0 on the null shell; H_r = 1/2 p_u^2 + V_r and H_theta = 1/2 p_theta^2 + V_theta
are separately conserved (H_theta = [Q + (L - aE)^2]/2).  t and phi are cyclic and
are advanced inside the kick (they depend only on r, theta).

Splitting: kick (V) / drift (kinetic).  Stormer-Verlet (2nd order), Yoshida 4th order
(3 Verlet substeps) and Yoshida 6th order (7 substeps).  Step in Mino time:
  h = min(h0, max_tstep / |dt/dlambda_M|)
which is the existing raytrace_cpu MAXDT cap expressed in Mino time (at large r it
becomes h ~ max_tstep / r^2, i.e. a constant affine step).

Reference solution: scipy DOP853 at rtol 1e-13 on the same Mino-time ODEs.
"""
import numpy as np, time, sys
from scipy.integrate import solve_ivp

a = 0.998; E = 1.0
b = np.sqrt(1 - a*a); r_plus = 1 + b

def r_of_u(u):      return 1 + b*np.cosh(u)
def u_of_r(r):      return np.arccosh(max((r-1)/b, 1.0))
def Delta_u(u):     return (b*np.sinh(u))**2
def P(r, L):        return (r*r + a*a)*E - a*L
def dVr_dr(r, L):
    D = r*r - 2*r + a*a; dD = 2*r - 2; Pr = P(r, L)
    return -(2*Pr*2*r*E*D - Pr*Pr*dD) / (2*D*D)
def dVth_dth(th, L):
    s = np.sin(th); c = np.cos(th); s2 = s*s; W = L - a*E*s2
    return (2*W*(-2*a*E*s*c)*s2 - W*W*2*s*c) / (2*s2*s2)
def tdot_M(r, th, L):   return (r*r + a*a)*P(r, L)/(r*r - 2*r + a*a) + a*(L - a*E*np.sin(th)**2)
def phidot_M(r, th, L): return a*P(r, L)/(r*r - 2*r + a*a) + L/np.sin(th)**2 - a*E
def H_r(u, pu, L):      return 0.5*pu*pu - P(r_of_u(u), L)**2/(2*Delta_u(u))
def H_th(th, pth, L):   s2 = np.sin(th)**2; return 0.5*pth*pth + (L - a*E*s2)**2/(2*s2)
def Carter_Q(th, pth, L): return pth*pth + np.cos(th)**2*(L*L/np.sin(th)**2 - a*a*E*E)

def kick(s, h, L):
    u, pu, th, pth, t, ph = s
    r = r_of_u(u)
    return (u, pu - h*dVr_dr(r, L)*b*np.sinh(u), th, pth - h*dVth_dth(th, L),
            t + h*tdot_M(r, th, L), ph + h*phidot_M(r, th, L))
def drift(s, h, L):
    u, pu, th, pth, t, ph = s
    return (u + h*pu, pu, th + h*pth, pth, t, ph)
def verlet(s, h, L):
    s = kick(s, h/2, L); s = drift(s, h, L); return kick(s, h/2, L)
_c = 2**(1/3); _w1 = 1/(2 - _c); _w0 = -_c/(2 - _c)
def yoshida4(s, h, L):
    s = verlet(s, _w1*h, L); s = verlet(s, _w0*h, L); return verlet(s, _w1*h, L)
_y6 = [0.784513610477560, 0.235573213359357, -1.17767998417887, 1.315186320683906]
_w6 = _y6[:3] + [_y6[3]] + _y6[2::-1]
def yoshida6(s, h, L):
    for w in _w6: s = verlet(s, w*h, L)
    return s
STEPPERS = {2: verlet, 4: yoshida4, 6: yoshida6}

def rhs(lam, y, L):
    u, pu, th, pth, t, ph = y; r = r_of_u(u)
    return [pu, -dVr_dr(r, L)*b*np.sinh(u), pth, -dVth_dth(th, L), tdot_M(r, th, L), phidot_M(r, th, L)]

def initial_state(r0, th0, L, Q, ingoing=True):
    pth0 = np.sqrt(Q - np.cos(th0)**2*(L*L/np.sin(th0)**2 - a*a*E*E))
    D = r0*r0 - 2*r0 + a*a
    R = P(r0, L)**2 - D*(Q + (L - a*E)**2)
    pr0 = (-1 if ingoing else 1)*np.sqrt(R)/D
    u0 = u_of_r(r0)
    return (u0, np.sqrt(D)*pr0, th0, pth0, 0.0, 0.0)

def integrate(s0, L, order, h0, max_tstep, r_lim=1000.0, steplim=5_000_000):
    stepper = STEPPERS[order]; s = s0; n = 0; lam = 0.0
    Hr0 = H_r(s[0], s[1], L); Ht0 = H_th(s[2], s[3], L); errs = [0.0, 0.0]; flips = 0
    while r_plus*1.001 < r_of_u(s[0]) < r_lim and n < steplim:
        h = min(h0, max_tstep/abs(tdot_M(r_of_u(s[0]), s[2], L)))
        pu_prev = s[1]
        s = stepper(s, h, L); lam += h; n += 1
        if pu_prev*s[1] < 0: flips += 1
        errs[0] = max(errs[0], abs(H_r(s[0], s[1], L) - Hr0))
        errs[1] = max(errs[1], abs(H_th(s[2], s[3], L) - Ht0))
    return s, lam, n, errs, flips

if __name__ == "__main__":
    # Near-critical rays for a = 0.998 (photon orbit radius r_ph = 2), impact parameters
    # offset from the critical curve by eps -> multi-orbit whirls before escape.
    rph = 2.0
    lam_c = -(rph**3 - 3*rph**2 + a*a*rph + a*a)/(a*(rph - 1))
    eta_c = -rph**3*(rph**3 - 6*rph**2 + 9*rph - 4*a*a)/(a*a*(rph - 1)**2)
    for eps in [1e-3, 1e-5]:
        L = lam_c*(1 + eps); Q = eta_c
        s0 = initial_state(1000.0, 1.2, L, Q)
        esc = lambda lam, y, L: r_of_u(y[0]) - 1000.0; esc.terminal = True; esc.direction = 1
        t0 = time.time()
        ref = solve_ivp(rhs, [0, 50], list(s0), args=(L,), method='DOP853', rtol=1e-13, atol=1e-14, events=[esc])
        yr = ref.y[:, -1]
        print(f"\n=== eps={eps}: reference exit theta={yr[2]:.6f} phi={yr[5]:.6f} ({yr[5]/2/np.pi:.2f} orbits) "
              f"t={yr[4]:.4f}  nfev={ref.nfev} ({time.time()-t0:.2f}s)")
        print(f"{'order':>5} {'h0':>7} {'maxdt':>6} {'steps':>7} {'dtheta':>9} {'dphi':>9} {'dt':>9} {'max|dHr|':>9} {'max|dHth|':>9} {'flips':>5} {'sec':>5}")
        for order in [2, 4, 6]:
            for h0, mdt in [(0.01, 1.0), (0.01, 0.25), (0.0025, 0.25), (0.001, 0.1)]:
                t0 = time.time()
                s, lam, n, errs, flips = integrate(s0, L, order, h0, mdt)
                print(f"{order:>5} {h0:>7} {mdt:>6} {n:>7} {abs(s[2]-yr[2]):9.1e} {abs(s[5]-yr[5]):9.1e} {abs(s[4]-yr[4]):9.1e} "
                      f"{errs[0]:9.1e} {errs[1]:9.1e} {flips:>5} {time.time()-t0:5.2f}")
