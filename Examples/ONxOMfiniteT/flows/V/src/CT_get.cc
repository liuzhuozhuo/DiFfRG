#include "../kernel.hh"

#include "../V.hh"

void V_integrator::get(double& dest, const double& k, const double& N1, const double& N2, const double& U1, const double& U2, const double& M11, const double& M12, const double& M22)
{
  integrator.get(dest,  k, N1, N2, U1, U2, M11, M12, M22);
}