#include "../kernel.hh"

#include "../V.hh"

void V_integrator::map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N1, const DiFfRG::PointArg<double>& N2, const DiFfRG::PointArg<double>& U1, const DiFfRG::PointArg<double>& U2, const DiFfRG::PointArg<double>& M11, const DiFfRG::PointArg<double>& M12, const DiFfRG::PointArg<double>& M22)
{
  integrator.map_points(dest,  k, N1, N2, U1, U2, M11, M12, M22);
}