/*
 * kerr.h
 *
 *  Created on: 15 Oct 2013
 *      Author: drw
 */

#ifndef KERR_H_
#define KERR_H_

#include <cmath>

template<typename T>
T kerr_horizon(T a, int sign = 1)
{
	//
	// Calculate event horizon in Kerr geometry
	//
	return 1 + sign*sqrt( (1-a)*(1+a) );
}

template<typename T>
T kerr_isco(T a, int sign)
{
	//
	// Calculate radius of innermost stable circular orbit in Kerr geometry
	// for prograde (sign = +1) and retrograde (sign = -1) orbits
	//
	const float A = 1. + pow(1.-a*a , 1./3.) * ( pow(1.+a , 1./3.) + pow(1.-a , 1./3.) );
	const float B = sqrt(3.*a*a + A*A);
	return 3 + B - sign*sqrt( (3-A) * (3 + A + 2*B) );
}

template<typename T>
T disc_velocity(T r, T a, int sign)
{
	return 1 / (a + sign*pow(r, 3./2.));
}

template<typename T>
void cartesian(T& x, T& y, T& z, T r, T theta, T phi, T a)
{
  //
  // find the cartesian co-ordinates (aff,x,y,z) in the kerr spacetime
  //
  // result stored in first argument (T* x)
  //
  // arguments
  //   x      T*        stores the calcualted 'cartesian' co-ordinates
  //   r      T[4]      input Boyer-Lindquist co-ordinates
  //   a      T         BH spin parameter
  //
  x = sqrt(r*r + a*a)*sin(theta)*cos(phi);
  y = sqrt(r*r + a*a)*sin(theta)*sin(phi);
  z = r*cos(theta);
}

template<typename T>
T dot_product(T (*g)[4], T* u, T* v)
{
	//
	// returns the dot product of 4-vectors u and v under the metric g
	//
	T dotprod = 0;
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
		{
			dotprod += g[i][j] * u[i] * v[j];
		}

	return dotprod;
}

template<typename T>
void minkowski(T (*g)[4])
{
  //
  // return the Minkowski metric, eta = diag(1, -1, -1, -1)
  //
  // arguments:
  //   g      T* [4][4]      returned metric coefficients
  //
  for(int i=0; i<4; i++)
    for(int j=0; j<4; j++)
      g[i][j] = 0;

  g[0][0] = 1;
  g[1][1] = -1;
  g[2][2] = -1;
  g[3][3] = -1;
}

template<typename T>
void kerr_metric(T (*g)[4], T *r, T a)
{
  //
  // calculate Kerr metric coefficients at position r
  // (x0,x1,x2,x3) = (t,r,theta,phi)
  //
  // arguments:
  //   g      T* [4][4]      calculated metric coefficients
  //   r      T[4]           position
  //   a      T              BH spin parameter
  //
  const T rhosq = r[1]*r[1] + (a*cos(r[2]))*(a*cos(r[2]));
  const T delta = r[1]*r[1] - 2*r[1] + a*a;
  const T sigmasq = (r[1]*r[1] + a*a)*(r[1]*r[1] + a*a) - a*a*delta*sin(r[2])*sin(r[2]);

  // metric coefficients
  const T e2nu = rhosq * delta / sigmasq;
  const T e2psi = sigmasq * sin(r[2])*sin(r[2]) / rhosq;
  const T omega = 2*a*r[1] / sigmasq;

  for(int i=0; i<4; i++)
    for(int j=0; j<4; j++)
      g[i][j] = 0;

  g[0][0] = e2nu - omega*omega*e2psi;
  g[0][3] = omega*e2psi;
  g[3][0] = g[0][3];
  g[3][3] = -e2psi;
  g[1][1] = -rhosq/delta;
  g[2][2] = -rhosq;
}

template<typename T>
void tetrad(T* et, T* e1, T* e2, T* e3, T *r, T V, T a)
{
  //
  // calculate the tetrad basis vectors of an observer in the Kerr spacetime
  // at position r rotating at angular velocity (d(phi)/dt) V
  //
  // arguments:
  //   et      T* [4]      calculated time-like basis vector
  //   e1      T* [4]      calculated phi basis vector
  //   e2      T* [4]      calculated theta basis vector
  //   e3      T* [4]      calculated r basis vector
  //   r       T[4]        position of observer
  //   V       T           angular velocity (d(phi)/dt) of observer
  //   a       T           BH spin parameter
  //
  const T rhosq = r[1]*r[1] + (a*cos(r[2]))*(a*cos(r[2]));
  const T delta = r[1]*r[1] - 2*r[1] + a*a;
  const T sigmasq = (r[1]*r[1] + a*a)*(r[1]*r[1] + a*a) - a*a*delta*sin(r[2])*sin(r[2]);

  // metric coefficients
  const T e2nu = rhosq * delta / sigmasq;
  const T e2psi = sigmasq * sin(r[2])*sin(r[2]) / rhosq;
  const T omega = 2*a*r[1] / sigmasq;

  et[0] = (1/sqrt(e2nu))/sqrt(1 - (V - omega)*(V - omega)*e2psi/e2nu);
  et[1] = 0;
  et[2] = 0;
  et[3] = (1/sqrt(e2nu))*V / sqrt(1 - (V - omega)*(V - omega)*e2psi/e2nu);

  e1[0] = (V - omega)*sqrt(e2psi/e2nu) / sqrt(e2nu - (V-omega)*(V-omega)*e2psi);
  e1[1] = 0;
  e1[2] = 0;
  e1[3] = (1/sqrt(e2nu*e2psi))*(e2nu + V*omega*e2psi - omega*omega*e2psi) / sqrt(e2nu - (V-omega)*(V-omega)*e2psi);

  e2[0] = 0;
  e2[1] = 0;
  e2[2] = 1/sqrt(rhosq);
  e2[3] = 0;

  e3[0] = 0;
  e3[1] = sqrt(delta/rhosq);
  e3[2] = 0;
  e3[3] = 0;
}

template<typename T>
T lorentz(T* vel, T *v, T *r, T a)
{
  //
  // returns the Lorentz factor corresponding to a 4-velocity v
  // for a stationary observer at r
  //
  // arguments
  //   vel    T* [4]   stores 4-velocity as measured in observer's frame
  //   v      T[4]     input 4-velocity
  //   r      T[4]     position 4-vector of observer
  //   a      T        BH spin parameter
  //
  T g[4][4];
  T et[4], e1[4], e2[4], e3[4];
  T gvel[4];

  const T rhosq = r[1]*r[1] + (a*cos(r[2]))*(a*cos(r[2]));
  const T delta = r[1]*r[1] - 2*r[1] + a*a;
  const T sigmasq = (r[1]*r[1] + a*a)*(r[1]*r[1] + a*a) - a*a*delta*sin(r[2])*sin(r[2]);

  // observer rotating with frame-dragging
  const T omega = 2*a*r[1] / sigmasq;

  // metric coefficients
    kerr_metric(g, r, a);

  // tetrad basis vectors
    tetrad(et, e1, e2, e3, r, omega, a);

  // calculate components of v by dot product with basis vectors (gamma*c, gamma*vel)
  gvel[0] = g[0][0]*v[0]*et[0] + g[0][3]*v[0]*et[3] + g[3][0]*v[3]*et[0] + g[3][3]*v[3]*et[3];
  gvel[1] = g[0][0]*v[0]*e1[0] + g[0][3]*v[0]*e1[3] + g[3][0]*v[3]*e1[0] + g[3][3]*v[3]*e1[3];
  gvel[2] = g[2][2]*v[2]*e2[2];
  gvel[3] = g[1][1]*v[1]*e3[1];

  // calculate v from gamma*v
  for(int i=1; i<4; i++)
    vel[i] = gvel[i] / gvel[0];

  return gvel[0];   // v[0] is gamma (c=1)
}

template<typename T>
T disc_velocity_vector(T* v, T r, T a, int sign)
{
  //
  // calculate the 4-velocity of an element of the disc in stable circular orbit
  // in the equatorial plane of the Kerr geometry
  //
  // returns the angular velcocity (d(phi)/dt) measured in co-ordinate frame (T)
  //
  // arguments:
  //   v      T* [4]   calculated 4-velocity
  //   r      T        radius
  //   a      T        BH spin parameter
  //   sign   int           +1 for prograde or -1 for retrograde orbit
  //
  const T u = 1 / r;

  // constants of motion for stable circular orbit in Kerr spacetime
  const T k = ( 1 - 2*u + sign*a*sqrt(u*u*u) ) / sqrt( 1 - 3*u + sign*2*a*sqrt(u*u*u) );
  const T h = sign * ( 1 + a*a*u*u - sign*2*a*sqrt(u*u*u) ) / ( sqrt(u) * sqrt( 1 - 3*u + sign*2*a*sqrt(u*u*u) ) );

  v[0] = ( r*r*(r*r + a*a) + 2*a*a*r )*k - 2*a*r*h;
  v[0] = v[0] / ( r*r*(1 - (2/r))*(r*r + a*a) + 2*a*a*r);

  v[1] = 0;
  v[2] = 0;

  v[3] = 2*a*r*k + (r*r - 2*r)*h;
  v[3] = v[3] / ( r*r*(1-(2/r))*(r*r + a*a) + 2*a*a*r);

  // calculate and return angular velocity
  return v[3] / v[0];
}

template<typename T>
T disc_area(T r, T dr, T a)
{
  //
  // returns the proper area of an annulus in the equatorial plane
  // (T)
  //
  // arguments:
  //   r      T      radial co-ordinate of annulus
  //   dr     T      co-ordinate thickness of annulus
  //   a      T      BH spin parameter
  //
  const T rhosq = r*r;
  const T delta = r*r - 2*r + a*a;

  return sqrt( r*r + a*a + (2*a*a*r)/rhosq ) * sqrt(rhosq/delta)*dr;
}

template<typename T>
T rel_disc_area(T r, T dr, T a)
{
  //
  // return the effective area of a thin annulus in the equatorial plane (T)
  // taking into account relativistic effects on area
  //
  // arguments:
  //   r      T      r co-ordinate of annulus
  //   dr     T      thickness of annulus (<<r)
  //   a      T      BH spin parameter
  //
  T area;

  T g[4][4], et[4], e1[4], e2[4], e3[4], v[4], vel[3];
  T pos[] = {0,0,0,0};
  T V, gr_area, gamma;

  pos[1] = r;
    kerr_metric<T>(g, pos, a);
  V = disc_velocity_vector<T>(v, r, a, +1);
    tetrad(et, e1, e2, e3, pos, V, a);

  gr_area = disc_area<T>(r, dr, a);
  // Lorentz factor
  gamma = lorentz<T>(vel, v, pos, a);

  area = gr_area/gamma;

  return area;
}

template <typename T>
inline void momentum_from_consts(T& pt, T& pr, T& ptheta, T& pphi,
						T k, T h, T Q, int rdot_sign, int thetadot_sign,
						T r, T theta, T phi,
						const T a)
{
	//
	// Calculate photon momentum from constants of motion at a location
	//
	const T sin_theta = sin(theta);
	const T cos_theta = cos(theta);
	const T sin2theta = sin_theta * sin_theta;
	const T rhosq = r*r + (a*cos_theta)*(a*cos_theta);
	const T delta = r*r - 2*r + a*a;
	const T rhosq_delta = rhosq * delta;

	// tdot
	pt = (rhosq*(r*r + a*a) + 2*a*a*r*sin2theta)*k - 2*a*r*h;
	pt /= rhosq_delta;

	// phidot
	pphi = 2*a*r*sin2theta*k + (rhosq - 2*r)*h;
	pphi /= sin2theta * rhosq_delta;

	// thetadot
	T thetadotsq = Q + (k*a*cos_theta + h*cos_theta/sin_theta)*(k*a*cos_theta - h*cos_theta/sin_theta);
	thetadotsq = thetadotsq / (rhosq*rhosq);

	// take the square roots and get the right signs
	ptheta = sqrt(abs(thetadotsq)) * thetadot_sign;

	// rdot
	T rdotsq = k*pt - h*pphi - rhosq*ptheta*ptheta;
	rdotsq = rdotsq * delta/rhosq;

	pr = sqrt(abs(rdotsq)) * rdot_sign;
}



// =============================================================================
// Mino-time helpers for the symplectic integrator (Integrator::Symplectic)
// =============================================================================
//
// With Mino time dlambda_M = dlambda / rhosq, the Kerr null geodesic Hamiltonian
// multiplied by rhosq separates into independent radial and polar parts:
//
//   H_M = H_r(u, p_u) + H_theta(theta, p_theta) = 0
//   H_r     = 1/2 p_u^2     - P(r)^2 / (2 Delta)              P = (r^2 + a^2) k - a h
//   H_theta = 1/2 p_theta^2 + (h - a k sin^2theta)^2 / (2 sin^2theta)
//
// The radial coordinate u is defined by r = 1 + b cosh(u), b = sqrt(1 - a^2), so
// that Delta = b^2 sinh^2(u), and the conjugate momentum p_u = sqrt(Delta) p_r.
// This makes the radial kinetic term a plain 1/2 p_u^2 (trivial, exact drift)
// and places the event horizon at u = 0.
//
// Constants of motion use the same conventions as the rest of the code:
// k = photon energy (E), h = z angular momentum (L_z), Q = Carter constant.
// The cyclic coordinates t and phi advance by
//   dt/dlambda_M   = (r^2 + a^2) P / Delta + a (h - a k sin^2theta)    ( = rhosq * pt   )
//   dphi/dlambda_M = a P / Delta + h / sin^2theta - a k                ( = rhosq * pphi )
// where pt, pphi are the contravariant components returned by momentum_from_consts().
//
// Contravariant (per affine parameter) momenta as stored in Ray<T>:
//   pr = Delta p_r / rhosq = sqrt(Delta) p_u / rhosq,   ptheta = p_theta / rhosq
//

template<typename T>
inline T mino_b(T a)
{
	// b = sqrt(1 - a^2); clamp so that a -> 1 does not degenerate the u coordinate
	const T aa = (abs(a) > T(0.9999)) ? T(0.9999) : abs(a);
	return sqrt((1 - aa)*(1 + aa));
}

template<typename T>
inline T mino_u_from_r(T r, T a)
{
	// inverse of r = 1 + b cosh(u); returns 0 at (or inside) the horizon
	const T x = (r - 1) / mino_b(a);
	return (x <= 1) ? T(0) : acosh(x);
}

template<typename T>
inline void mino_r_from_u(T u, T a, T& r, T& sqrt_delta)
{
	const T b = mino_b(a);
	r = 1 + b*cosh(u);
	sqrt_delta = b*sinh(u);
}

template<typename T>
inline T mino_tdot(T r, T theta, T k, T h, T a)
{
	// dt/dlambda_M
	const T delta = r*r - 2*r + a*a;
	const T P = (r*r + a*a)*k - a*h;
	const T sin2theta = sin(theta)*sin(theta);
	return (r*r + a*a)*P/delta + a*(h - a*k*sin2theta);
}

template<typename T>
inline T mino_phidot(T r, T theta, T k, T h, T a)
{
	// dphi/dlambda_M
	const T delta = r*r - 2*r + a*a;
	const T P = (r*r + a*a)*k - a*h;
	const T sin2theta = sin(theta)*sin(theta);
	return a*P/delta + h/sin2theta - a*k;
}

template<typename T>
inline T mino_dVr_dr(T r, T k, T h, T a)
{
	// d/dr of the radial potential V_r = -P^2 / (2 Delta)
	const T delta = r*r - 2*r + a*a;
	const T ddelta = 2*r - 2;
	const T P = (r*r + a*a)*k - a*h;
	const T dP = 2*r*k;
	return -(2*P*dP*delta - P*P*ddelta) / (2*delta*delta);
}

template<typename T>
inline T mino_dVtheta_dtheta(T theta, T k, T h, T a)
{
	// d/dtheta of the polar potential V_theta = W^2 / (2 sin^2theta),  W = h - a k sin^2theta
	const T s = sin(theta);
	const T c = cos(theta);
	const T W = h - a*k*s*s;
	const T dW = -2*a*k*s*c;
	return W*(dW*s - W*c) / (s*s*s);
}

template<typename T>
inline void mino_init(T r, T theta, T k, T h, T Q, int rdot_sign, int thetadot_sign, T a,
                      T& u, T& pu, T& ptheta)
{
	//
	// Canonical (u, p_u, p_theta) from a ray's position, constants of motion and direction signs.
	// R(r) = P^2 - Delta (Q + (h - a k)^2),  Theta(theta) = Q + cos^2theta (a^2 k^2 - h^2 / sin^2theta)
	// p_r = +/- sqrt(R) / Delta  ->  p_u = sqrt(Delta) p_r = +/- sqrt(R) / sqrt(Delta)
	//
	const T delta = r*r - 2*r + a*a;
	const T P = (r*r + a*a)*k - a*h;
	const T c = cos(theta), s = sin(theta);

	T R = P*P - delta*(Q + (h - a*k)*(h - a*k));
	if (R < 0) R = 0;
	T Th = Q + c*c*(a*a*k*k - h*h/(s*s));
	if (Th < 0) Th = 0;

	u = mino_u_from_r(r, a);
	pu = rdot_sign * sqrt(R) / sqrt(delta);
	ptheta = thetadot_sign * sqrt(Th);
}

template<typename T>
inline void mino_to_bl(T u, T pu, T theta, T ptheta, T k, T h, T a,
                       T& r, T& pt, T& pr, T& pth, T& pphi)
{
	//
	// Convert the Mino-time canonical state back to Boyer-Lindquist position r and the
	// contravariant momenta (per affine parameter) stored in Ray<T>.
	//
	T sqrt_delta;
	mino_r_from_u(u, a, r, sqrt_delta);
	const T rhosq = r*r + (a*cos(theta))*(a*cos(theta));
	pt   = mino_tdot(r, theta, k, h, a) / rhosq;
	pphi = mino_phidot(r, theta, k, h, a) / rhosq;
	pr   = sqrt_delta*pu / rhosq;
	pth  = ptheta / rhosq;
}

// --- exact polar flow ---------------------------------------------------------
//
// The polar potential splits as V_theta = h^2 / (2 sin^2theta) + W(theta), with
//   W(theta) = 1/2 a^2 k^2 sin^2theta - a k h          (bounded, smooth)
// The part H_c = 1/2 p_theta^2 + h^2 / (2 sin^2theta) is the free particle on the
// unit sphere with z angular momentum h and total angular momentum J = sqrt(2 H_c):
// its flow is a great circle,  cos(theta) = A sin(psi),  A = sqrt(1 - h^2/J^2),
// psi = J lambda_M + psi_0, and the associated phi advance (dphi/dlambda_M = h/sin^2theta)
// integrates to sign(h) atan(|h|/J tan psi), continued across branches.
// Using this exact flow as the polar "drift" (with only W in the kick) keeps the
// integrator stable for rays that start or turn close to the polar axis, where
// the h^2/sin^2theta barrier would otherwise require a tiny step.
//

template<typename T>
inline T mino_dW_dtheta(T theta, T k, T a)
{
	// d/dtheta of W = 1/2 a^2 k^2 sin^2theta - a k h
	return a*a*k*k*sin(theta)*cos(theta);
}

template<typename T>
inline T mino_polar_phi_branch(T psi, T h, T J)
{
	// continuous antiderivative of h / sin^2theta along the great circle, as a function of psi
	const T n = floor((psi + T(M_PI_2)) / T(M_PI));
	const T sgn = (h >= 0) ? T(1) : T(-1);
	return sgn * (atan(abs(h)/J * tan(psi - n*T(M_PI))) + n*T(M_PI));
}

template<typename T>
inline void mino_polar_flow(T& theta, T& ptheta, T& phi, T h, T lam)
{
	//
	// Advance (theta, p_theta, phi) by Mino time lam under the exact flow of
	// H_c = 1/2 p_theta^2 + h^2 / (2 sin^2theta), including the h / sin^2theta part of dphi/dlambda_M.
	//
	const T s = sin(theta);
	const T J2 = ptheta*ptheta + h*h/(s*s);
	const T J = sqrt(J2);
	T A2 = 1 - h*h/J2;
	if (A2 <= T(1e-30) || J <= 0)
	{
		// motion confined to the equator (or no motion): theta and p_theta fixed, phi advances at h/sin^2theta
		phi += lam * h / (s*s);
		return;
	}
	const T A = sqrt(A2);
	const T psi0 = atan2(cos(theta)/A, -s*ptheta/(J*A));
	const T psi1 = psi0 + J*lam;
	const T z1 = A*sin(psi1);
	theta = acos(z1);
	const T s1 = sin(theta);
	ptheta = -J*A*cos(psi1)/s1;
	// (for h = 0 the branch function reduces to n pi, i.e. phi advances by pi each time the ray
	// passes over a pole, which is the correct h -> 0 limit)
	phi += mino_polar_phi_branch<T>(psi1, h, J) - mino_polar_phi_branch<T>(psi0, h, J);
}

// --- diagnostics: separately conserved pieces of the Mino-time Hamiltonian, and Carter's Q ---

template<typename T>
inline T mino_hamiltonian_r(T u, T pu, T k, T h, T a)
{
	T r, sqrt_delta;
	mino_r_from_u(u, a, r, sqrt_delta);
	const T P = (r*r + a*a)*k - a*h;
	return T(0.5)*pu*pu - P*P/(2*sqrt_delta*sqrt_delta);
}

template<typename T>
inline T mino_hamiltonian_theta(T theta, T ptheta, T k, T h, T a)
{
	const T s2 = sin(theta)*sin(theta);
	const T W = h - a*k*s2;
	return T(0.5)*ptheta*ptheta + W*W/(2*s2);
}

template<typename T>
inline T carter_Q(T theta, T ptheta, T k, T h, T a)
{
	// Carter constant from the covariant polar momentum p_theta
	const T c = cos(theta), s = sin(theta);
	return ptheta*ptheta + c*c*(h*h/(s*s) - a*a*k*k);
}


#endif /* KERR_H_ */
