#pragma once

// The KT headers come before DiFfRG.hh, see ../ONfiniteT/KT.cc.
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/model/fv_boundaries.hh>

#include <DiFfRG/DiFfRG.hh>
#include <DiFfRG/discretization/FV/coordinates/invariant_coordinates.hh>
#include <DiFfRG/discretization/FV/coordinates/polar_invariants.hh>
#include <DiFfRG/model/gradient_potential.hh>
using namespace DiFfRG;

#include "flows/flows.hh"

#include <cmath>
#include <type_traits>
#include <vector>

/**
 * The O(N1) x O(N2) model at finite temperature in the LPA: a potential U(rho1, rho2) of the two invariants
 * rho_a = phi_a^2 / 2 of an O(N1) and an O(N2) multiplet, evolved in gradient form with def::GradientPotential on a
 * coordinate map (polar or Cartesian, FV::PolarInvariants / FV::CartesianInvariants). The flow is the kernel V of
 * ONxOM.wl, evaluated for all points of a KT batch in one map_points() call.
 *
 * Initial condition: U_Lambda = U_O(N)(rho1 + rho2) + delta rho2, with U_O(N)' = lambda2 + lambda4 rho + lambda6 rho^2
 * as in ../ONfiniteT; delta > 0 breaks O(N1 + N2) explicitly and favours the first multiplet. Model1D is the
 * O(N1 + N2) model in the one invariant rho (kernel V_ON), the reference for delta = 0.
 */
namespace ONxOM
{
  struct Parameters {
    Parameters(const ConfigTree &config)
        : Lambda(config.get_double("/physical/Lambda")), N1(config.get_double("/physical/N1")),
          N2(config.get_double("/physical/N2")), T(config.get_double("/physical/T")),
          lambda2(config.get_double("/physical/lambda2")), lambda4(config.get_double("/physical/lambda4")),
          lambda6(config.get_double("/physical/lambda6")), delta(config.get_double("/physical/delta", 0.))
    {
    }
    double Lambda, N1, N2, T, lambda2, lambda4, lambda6, delta;

    /// U'_O(N)(rho) of the initial potential.
    double u_Lambda(const double rho) const { return lambda2 + lambda4 * rho + lambda6 * rho * rho; }
  };

  using Components = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u1">, Scalar<"u2">>>;
  using Components1D = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>;

  /// The radial 2x2 block M_ab = U_a delta_ab + 2 sqrt(rho_a rho_b) U_ab of the meson curvature.
  template <typename NT> struct RadialBlock {
    NT M11, M12, M22;
  };
  template <typename NT> RadialBlock<NT> radial_block(const FV::InvariantState<2, NT> &s)
  {
    return {s.U1[0] + 2. * s.rho[0] * s.U2[0][0], 2. * std::sqrt(s.rho[0] * s.rho[1]) * s.U2[0][1],
            s.U1[1] + 2. * s.rho[1] * s.U2[1][1]};
  }

  /**
   * @tparam Map FV::PolarInvariants or FV::CartesianInvariants<2>; the FE functions "u1", "u2" hold its two fields.
   */
  template <typename Map>
  class Model : public def::GradientPotential<Model<Map>, Components, Map, "u1", "u2">,
                public def::fRG,
                public def::FVDefaultBoundaries<Model<Map>>,
                public def::AD<Model<Map>>
  {
    using Base = def::GradientPotential<Model<Map>, Components, Map, "u1", "u2">;

  public:
    static constexpr uint dim = 2;
    static constexpr bool readouts_use_potential = false;

  protected:
    const Parameters prm;
    mutable ONxOMFlows flow_equations;

  public:
    Model(const ConfigTree &config)
        : Base(config), def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
    {
      flow_equations.set_k(Lambda);
      flow_equations.set_T(prm.T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      this->gradient_initial_condition(x, values, [&](double rho) { return prm.u_Lambda(rho); });
      // + delta rho2: U_2 += delta, i.e. v += s delta, w += delta (polar, w form), u2 += delta (Cartesian)
      if constexpr (std::is_same_v<Map, FV::PolarInvariants>) {
        values[0] += x[1] * prm.delta;
        values[1] += this->coordinates().parameters().form == FV::PolarForm::w ? prm.delta : x[0] * prm.delta;
      } else
        values[1] += prm.delta;
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * prm.Lambda;
      flow_equations.set_k(k);
    }

    /// d_t U = -F at one point (the per-point path of the assembler).
    template <typename NT, typename Solution>
    NT potential_flux(const FV::InvariantState<2, NT> &s, const Point<dim> & /*x*/, const Solution & /*sol*/) const
    {
      const auto M = radial_block(s);
      NT F;
      flow_equations.V.get(F, k, prm.N1, prm.N2, s.U1[0], s.U1[1], M.M11, M.M12, M.M22);
      return F;
    }

    /// F at all points of a batch in one map_points() call.
    template <typename NT, typename Batch>
    void potential_flux_batch(std::vector<NT> &F, const std::vector<FV::InvariantState<2, NT>> &states,
                              const Batch & /*batch*/) const
    {
      const size_t n = states.size();
      std::vector<NT> U1(n), U2(n), M11(n), M12(n), M22(n);
      for (size_t i = 0; i < n; ++i) {
        const auto M = radial_block(states[i]);
        U1[i] = states[i].U1[0];
        U2[i] = states[i].U1[1];
        M11[i] = M.M11;
        M12[i] = M.M12;
        M22[i] = M.M22;
      }
      flow_equations.V.map_points(F, k, prm.N1, prm.N2, U1, U2, M11, M12, M22);
    }

    /// At the minimum of U: the invariants, the Goldstone masses U_a and the radial masses (eigenvalues of M).
    template <int dim_, typename DataOut, typename Solutions>
    void readouts(DataOut &output, const Point<dim_> &x, const Solutions &sol) const
    {
      const auto s = this->template potential_state<double>(x, get<"fe_functions">(sol), get<"fe_derivatives">(sol));
      const auto M = radial_block(s);
      const double c = 0.5 * (M.M11 + M.M22), d = std::sqrt(0.25 * powr<2>(M.M11 - M.M22) + powr<2>(M.M12));

      auto out_file = output.table("data.csv");
      out_file.set_Lambda(Lambda);
      out_file.value("rho1", s.rho[0]);
      out_file.value("rho2", s.rho[1]);
      out_file.value("sigma [GeV]", std::sqrt(2. * (s.rho[0] + s.rho[1])));
      out_file.value("m^2_{pi,1} [GeV^2]", s.U1[0]);
      out_file.value("m^2_{pi,2} [GeV^2]", s.U1[1]);
      out_file.value("m^2_{radial,low} [GeV^2]", c - d);
      out_file.value("m^2_{radial,high} [GeV^2]", c + d);
    }
  };

  /// The O(N1 + N2) model in the one invariant rho = rho1 + rho2: the reference for delta = 0.
  class Model1D : public def::GradientPotential<Model1D, Components1D, FV::CartesianInvariants<1>, "u">,
                  public def::fRG,
                  public def::FVDefaultBoundaries<Model1D>,
                  public def::AD<Model1D>
  {
    using Base = def::GradientPotential<Model1D, Components1D, FV::CartesianInvariants<1>, "u">;

  public:
    static constexpr uint dim = 1;
    static constexpr bool readouts_use_potential = false;

  protected:
    const Parameters prm;
    mutable ONxOMFlows flow_equations;

  public:
    Model1D(const ConfigTree &config)
        : Base(config), def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
    {
      flow_equations.set_k(Lambda);
      flow_equations.set_T(prm.T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      this->gradient_initial_condition(x, values, [&](double rho) { return prm.u_Lambda(rho); });
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * prm.Lambda;
      flow_equations.set_k(k);
    }

    template <typename NT, typename Solution>
    NT potential_flux(const FV::InvariantState<1, NT> &s, const Point<dim> & /*x*/, const Solution & /*sol*/) const
    {
      NT F;
      flow_equations.V_ON.get(F, k, prm.N1 + prm.N2, s.U1[0], s.U1[0] + 2. * s.rho[0] * s.U2[0][0]);
      return F;
    }

    template <typename NT, typename Batch>
    void potential_flux_batch(std::vector<NT> &F, const std::vector<FV::InvariantState<1, NT>> &states,
                              const Batch & /*batch*/) const
    {
      const size_t n = states.size();
      std::vector<NT> m2Pi(n), m2Sigma(n);
      for (size_t i = 0; i < n; ++i) {
        m2Pi[i] = states[i].U1[0];
        m2Sigma[i] = states[i].U1[0] + 2. * states[i].rho[0] * states[i].U2[0][0];
      }
      flow_equations.V_ON.map_points(F, k, prm.N1 + prm.N2, m2Pi, m2Sigma);
    }

    template <int dim_, typename DataOut, typename Solutions>
    void readouts(DataOut &output, const Point<dim_> &x, const Solutions &sol) const
    {
      const auto s = this->template potential_state<double>(x, get<"fe_functions">(sol), get<"fe_derivatives">(sol));
      auto out_file = output.table("data.csv");
      out_file.set_Lambda(Lambda);
      out_file.value("rho", s.rho[0]);
      out_file.value("sigma [GeV]", std::sqrt(2. * s.rho[0]));
      out_file.value("m^2_{pi} [GeV^2]", s.U1[0]);
      out_file.value("m^2_{sigma} [GeV^2]", s.U1[0] + 2. * s.rho[0] * s.U2[0][0]);
    }
  };
} // namespace ONxOM
