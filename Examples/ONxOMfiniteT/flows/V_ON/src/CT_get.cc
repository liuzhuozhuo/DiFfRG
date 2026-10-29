#include "../kernel.hh"

#include "../V_ON.hh"

void V_ON_integrator::get(double& dest, const double& k, const double& N, const double& m2Pi, const double& m2Sigma)
{
  integrator.get(dest,  k, N, m2Pi, m2Sigma);
}