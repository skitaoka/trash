float G(float const v, float const sigma)
{
  return v / (sigma - sigma * v + v);
}

float Z(float t, float sigma)
{
  float const sq_t = t * t;
  float const demZ = 1 / (1 + sigma * sq_t - sq_t);
  return sigma * demZ * demZ;
}

float A(float const w, float const psi)
{
  float sq_psi = psi * psi;
  float sq_w = w * w;
  return sqrtf( psi / ( sq_psi - sq_psi * sq_w + sq_w ) );
}

float S(float const u, float const f0)
{
  return f0 + (1 - f0) * pow(saturate(1 - u), 5);
}

// Schlick BRDF
float Schlick(float3 const omega, float3 const omega_dash, float3 const normal, float3 const tangent)
{
  float3 const H = normalize(omega + omega_dash);
  float const u = dot(omega, H);
  float const t = dot(normal, H);
  float const v = dot(omega, H);
  float const vd = dot(omega_dash, H);
  float const w = dot(tangent, normalize(H - t * normal));
  float const g = 4 * sigma * (1 - sigma);
  float const d = (sigma < 0.5) ? 0 : 1 - g;
  float const s = (sigma >= 0.5) ? 0 : 1 - g;
  float D = 0;
  if ((v > EPSILON) && (vd > EPSILON)) {
    float GG = G(v, sigma) * G(vd, sigma);
    float AZ = A(w, psi) * Z(t, sigma);
    D = (GG * (AZ - 1) + 1) / (4 * PI * v * vd);
  }
  return saturate(vd) * (d / PI + g * D + s * DiracDelta(x, omega, omega_dash));
}
