#include "./flows.hh"

ONxOMFlows::ONxOMFlows(const DiFfRG::ConfigTree& config) : quadrature_provider(config), V(quadrature_provider, config), V_ON(quadrature_provider, config)
{
}
void ONxOMFlows::set_k(const double k)
{
  DiFfRG::all_set_k(V, k);DiFfRG::all_set_k(V_ON, k);
}
void ONxOMFlows::set_T(const double T)
{
  DiFfRG::all_set_T(V, T);DiFfRG::all_set_T(V_ON, T);
}
void ONxOMFlows::set_typical_E(const double E)
{
  DiFfRG::all_set_typical_E(V, E);DiFfRG::all_set_typical_E(V_ON, E);
}
void ONxOMFlows::set_x_extent(const double x_extent)
{
  DiFfRG::all_set_x_extent(V, x_extent);DiFfRG::all_set_x_extent(V_ON, x_extent);
}