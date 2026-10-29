#include "../kernel.hh"

#include "../V.hh"

void V_integrator::get(autodiff::real& dest, const double& k, const double& N1, const double& N2, const autodiff::real& U1, const autodiff::real& U2, const autodiff::real& M11, const autodiff::real& M12, const autodiff::real& M22)
{
  integrator_AD.get(dest,  k, N1, N2, U1, U2, M11, M12, M22);
}
void V_integrator::get(autodiff::Real<2, double>& dest, const double& k, const double& N1, const double& N2, const autodiff::Real<2, double>& U1, const autodiff::Real<2, double>& U2, const autodiff::Real<2, double>& M11, const autodiff::Real<2, double>& M12, const autodiff::Real<2, double>& M22)
{
  integrator_AD2.get(dest,  k, N1, N2, U1, U2, M11, M12, M22);
}