#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG {
  template<typename _Regulator>
  class V_kernel
  {
    public:
    using Regulator = _Regulator;

    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& l0, const auto& k, const auto& N1, const auto& N2, const auto& U1, const auto& U2, const auto& M11, const auto& M12, const auto& M22)
    {
      using namespace DiFfRG;using namespace DiFfRG::compute;
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _den1 = powr<-1>(_interp1 + powr<2>(l0) + powr<2>(l1) + U1);
      const auto _den2 = powr<-1>(_interp1 + powr<2>(l0) + powr<2>(l1) + U2);
      const auto _den3 = powr<-1>(-powr<2>(M12) + (_interp1 + powr<2>(l0) + powr<2>(l1) + M11) * (_interp1 + powr<2>(l0) + powr<2>(l1) + M22));
      return 0.5 * fma(-1., _den1 * _interp2, fma(-1., _den2 * _interp2, fma(2., _den3 * _interp1 * _interp2, fma(2., _den3 * _interp2 * powr<2>(l0), fma(2., _den3 * _interp2 * powr<2>(l1), fma(_den3, _interp2 * M11, fma(_den3, _interp2 * M22, fma(_den1, _interp2 * N1, fma(_den2, _interp2 * N2, 0.)))))))));
    }

    static KOKKOS_INLINE_FUNCTION auto constant(const auto& k, const auto& N1, const auto& N2, const auto& U1, const auto& U2, const auto& M11, const auto& M12, const auto& M22)
    {
      using namespace DiFfRG;using namespace DiFfRG::compute;
      return 0.;
    }
    private:
    static KOKKOS_INLINE_FUNCTION auto RB(const auto& k2, const auto& p2)
    {
      return Regulator::RB(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto RF(const auto& k2, const auto& p2)
    {
      return Regulator::RF(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto RBdot(const auto& k2, const auto& p2)
    {
      return Regulator::RBdot(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto RFdot(const auto& k2, const auto& p2)
    {
      return Regulator::RFdot(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto dq2RB(const auto& k2, const auto& p2)
    {
      return Regulator::dq2RB(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto dq2RF(const auto& k2, const auto& p2)
    {
      return Regulator::dq2RF(k2, p2);
    }
  };
} using DiFfRG::V_kernel;