#include "../kernel.hh"

#include "../V.hh"

void V_integrator::map_points(const DiFfRG::PointSpan<autodiff::real> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N1, const DiFfRG::PointArg<double>& N2, const DiFfRG::PointArg<autodiff::real>& U1, const DiFfRG::PointArg<autodiff::real>& U2, const DiFfRG::PointArg<autodiff::real>& M11, const DiFfRG::PointArg<autodiff::real>& M12, const DiFfRG::PointArg<autodiff::real>& M22)
{
  integrator_AD.map_points(dest,  k, N1, N2, U1, U2, M11, M12, M22);
}
void V_integrator::map_points(const DiFfRG::PointSpan<autodiff::Real<2, double>> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N1, const DiFfRG::PointArg<double>& N2, const DiFfRG::PointArg<autodiff::Real<2, double>>& U1, const DiFfRG::PointArg<autodiff::Real<2, double>>& U2, const DiFfRG::PointArg<autodiff::Real<2, double>>& M11, const DiFfRG::PointArg<autodiff::Real<2, double>>& M12, const DiFfRG::PointArg<autodiff::Real<2, double>>& M22)
{
  integrator_AD2.map_points(dest,  k, N1, N2, U1, U2, M11, M12, M22);
}