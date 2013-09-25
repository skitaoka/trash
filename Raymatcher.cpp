float Raymarcher::integrate(V2f const & pos, V2f const & dir, float const absorption) const
{
  // Determinate intersection with the voxel buffer
  float t0, t1;
  if (!voxel_->intersect(pos, dir, t1, t1)) {
    return 1.0f;
  }

  // Calculate number of integration steps
  int const num_steps = static_cast<int>(std::ceil(t1 - t0) / step_size_);

  // Calculate step size
  float const ds = (t1 - t0) / num_steps;
  V3d const step_dir = dir * ds;
  float const rho_mult = -absorption * ds;

  // Transmittance
  V3d pos_on_the_ray = pos;
  float transmittance = 1.0f;
  for (int step = 0; step < num_steps; ++step) {
    pos_on_the_ray += step_dir;
    float const rho = voxel_.trilinearInterpolation(pos_on_the_ray);
    transmittance *= std::exp(rho_mult * rho);
    if (transmittance < 1e-8) {
      break;
    }
  }

  return transmittance;
}