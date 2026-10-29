#include "../kernel.hh"

#include "../V_ON.hh"

void V_ON_integrator::map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& m2Pi, const DiFfRG::PointArg<double>& m2Sigma)
{
  integrator.map_points(dest,  k, N, m2Pi, m2Sigma);
}