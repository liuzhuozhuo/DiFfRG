// Gradient-form KT for a potential of two invariants (def::GradientPotential) on the polar and Cartesian maps,
// against the same flow in one invariant.
//
// Zero-dimensional O(N1) x O(N2) LPA, rho_a = phi_a^2 / 2 for the two multiplets, regulator r = k = Lambda e^{-t}:
//   d_t U = -F,  F = r/2 [ (N1 - 1)/(r + U_1) + (N2 - 1)/(r + U_2) + tr (r + M)^{-1} ],
//   M = [[U_1 + 2 rho_1 U_11, 2 sqrt(rho_1 rho_2) U_12], [., U_2 + 2 rho_2 U_22]],
// and in one invariant rho = rho_1 + rho_2, F = r/2 [ (N - 1)/(r + U') + 1/(r + U' + 2 rho U'') ], N = N1 + N2.
// From an O(N)-symmetric start the polar fields are v(r, s) = U'(r), w = 0 exactly, and the polar flux does not
// depend on s: v must follow the 1D flow to the time-stepping tolerance.

#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/timestepping/timestepping.hh"
#include "kt_regression_helpers.hh"

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/FV/coordinates/invariant_coordinates.hh>
#include <DiFfRG/discretization/FV/coordinates/polar_invariants.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/gradient_potential.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;

  constexpr double Lambda = 10.;
  constexpr double final_time = 3.5; // r = 0.30
  constexpr double N1 = 2., N2 = 2.;
  constexpr double m2 = -0.2, lambda = 1.;
  constexpr double rho_max = 2.;

  /// \f$U'_\Lambda(\rho)\f$.
  double u_Lambda(const double rho) { return m2 + lambda * rho; }

  template <int n> struct ComponentsOf;
  template <> struct ComponentsOf<1> {
    using type = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>;
  };
  template <> struct ComponentsOf<2> {
    using type = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u1">, Scalar<"u2">>>;
  };

  /// The 0D O(N1) x O(N2) LPA on a coordinate map; with `batched` through potential_flux_batch.
  template <typename Map, bool batched, FixedString... fields>
  class ONxOM : public def::GradientPotential<ONxOM<Map, batched, fields...>, typename ComponentsOf<Map::dim>::type,
                                              Map, fields...>,
                public def::fRG,
                public def::FVDefaultBoundaries<ONxOM<Map, batched, fields...>>,
                public def::AD<ONxOM<Map, batched, fields...>>
  {
    using Base = def::GradientPotential<ONxOM, typename ComponentsOf<Map::dim>::type, Map, fields...>;

  public:
    static constexpr int dim = Map::dim;
    static constexpr bool readouts_use_potential = false;

    explicit ONxOM(const ConfigTree &json, const double symmetry_breaking = 0.)
        : Base(json), def::fRG(json), delta(symmetry_breaking)
    {
    }

    /// \f$U_\Lambda = U_{O(N)}(\rho_1 + \rho_2) + \delta\rho_2\f$.
    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      this->gradient_initial_condition(x, values, u_Lambda);
      if constexpr (dim == 2) {
        if constexpr (std::is_same_v<Map, FV::PolarInvariants>) {
          // v += s delta, w (or W = r w) += delta
          values[0] += x[1] * delta;
          values[1] += this->coordinates().parameters().form == FV::PolarForm::w ? delta : std::max(x[0], 0.) * delta;
        } else
          values[1] += delta;
      }
    }

    template <typename NT, typename Solution>
    NT potential_flux(const FV::InvariantState<dim, NT> &s, const Point<dim> & /*x*/, const Solution & /*sol*/) const
    {
      return F(s);
    }

    template <typename NT, typename Batch>
    void potential_flux_batch(std::vector<NT> &J, const std::vector<FV::InvariantState<dim, NT>> &states,
                              const Batch & /*batch*/) const
      requires batched
    {
      for (size_t i = 0; i < states.size(); ++i)
        J[i] = F(states[i]);
    }

  private:
    template <typename NT> NT F(const FV::InvariantState<dim, NT> &s) const
    {
      const double r = k;
      if constexpr (dim == 1)
        return 0.5 * r * ((N1 + N2 - 1.) / (r + s.U1[0]) + 1. / (r + s.U1[0] + 2. * s.rho[0] * s.U2[0][0]));
      else {
        const NT a = r + s.U1[0] + 2. * s.rho[0] * s.U2[0][0];
        const NT d = r + s.U1[1] + 2. * s.rho[1] * s.U2[1][1];
        const NT b = 2. * std::sqrt(s.rho[0] * s.rho[1]) * s.U2[0][1];
        return 0.5 * r * ((N1 - 1.) / (r + s.U1[0]) + (N2 - 1.) / (r + s.U1[1]) + (a + d) / (a * d - b * b));
      }
    }

    double delta;
  };

  using Model1D = ONxOM<FV::CartesianInvariants<1>, false, "u">;
  template <bool batched> using ModelPolar = ONxOM<FV::PolarInvariants, batched, "u1", "u2">;
  using ModelCartesian = ONxOM<FV::CartesianInvariants<2>, false, "u1", "u2">;

  ConfigTree make_json(const std::string &x_grid, const std::string &y_grid, const std::string &polar_form = "w")
  {
    return json::value(
        {{"physical", {{"Lambda", Lambda}}},
         {"discretization",
          {{"fe_order", 0},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", x_grid}, {"y_grid", y_grid}, {"z_grid", "0:1:1"}, {"refine", 0}}},
           {"invariant_coordinates", {{"polar_form", polar_form}, {"damping", 20.}}}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", final_time},
           {"explicit",
            {{"dt", 1.0e-4}, {"minimal_dt", 1.0e-12}, {"maximal_dt", 1.0}, {"abs_tol", 1.0e-12}, {"rel_tol", 1.0e-10}}},
           {"implicit",
            {{"dt", 1.0e-4},
             {"minimal_dt", 1.0e-14},
             {"maximal_dt", 0.1},
             {"abs_tol", 1.0e-11},
             {"rel_tol", 1.0e-9}}}}},
         {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
  }

  /// The final solution, per cell centre: centre -> values of all components.
  using Profile = std::map<std::array<double, 2>, std::vector<double>>;

  template <typename Model, int dim> Profile run(const ConfigTree &json, Model model, const std::string &name)
  {
    using Components = typename Model::Components;
    using Mesh = RectangularMeshSerial<dim>;
    using Discretization = FV::Discretization<Components, Mesh, double>;
    using VectorType = typename Discretization::VectorType;
    using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
    using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;

    kt_regression::ensure_logger();
    Mesh mesh{Config::ConfigurationMesh<dim>(json)};
    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);
    auto path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, name, "output");
    OutputSession<Discretization> data_out(path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepperSUNDIALS_IDA<Discretization> time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());
    time_stepper.run(state, 0.0, final_time);

    const auto &dof_handler = discretization.get_dof_handler();
    const auto &fe = discretization.get_fe();
    const auto &solution = state.spatial_data();
    Profile profile;
    std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      cell->get_dof_indices(dofs);
      std::array<double, 2> key{{cell->center()[0], dim == 2 ? cell->center()[dim - 1] : 0.}};
      auto &values = profile[key];
      values.resize(fe.n_components());
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        values[fe.system_to_component_index(i).first] = solution[dofs[i]];
    }
    return profile;
  }

  /// Linear interpolation of the 1D profile at rho.
  double interpolate_1D(const Profile &p, const double rho)
  {
    auto hi = p.lower_bound({{rho, 0.}});
    if (hi == p.end()) --hi;
    if (hi == p.begin()) ++hi;
    auto lo = std::prev(hi);
    const double x0 = lo->first[0], x1 = hi->first[0];
    return lo->second[0] + (rho - x0) / (x1 - x0) * (hi->second[0] - lo->second[0]);
  }

  /// Bilinear interpolation of component c of a 2D profile on a uniform grid at (x, y).
  double interpolate_2D(const Profile &p, const double x, const double y, const unsigned int c, const double hx,
                        const double hy)
  {
    const auto centre = [](double z, double h) { return (std::floor(z / h - 0.5) + 0.5) * h; };
    const double x0 = centre(x, hx), y0 = centre(y, hy);
    const auto value = [&](double xi, double yi) {
      const auto it = p.find({{xi, yi}});
      if (it != p.end()) return it->second[c];
      // tolerant lookup (round-off in the cell centres)
      for (const auto &[key, v] : p)
        if (std::abs(key[0] - xi) < 1e-9 && std::abs(key[1] - yi) < 1e-9) return v[c];
      throw std::runtime_error("interpolate_2D: point outside the grid");
    };
    const double tx = (x - x0) / hx, ty = (y - y0) / hy;
    return (1. - tx) * (1. - ty) * value(x0, y0) + tx * (1. - ty) * value(x0 + hx, y0) +
           (1. - tx) * ty * value(x0, y0 + hy) + tx * ty * value(x0 + hx, y0 + hy);
  }

  const std::string r_grid = "0:0.01:2";
  const std::string s_grid = "0:0.0625:1";
  const std::string c_grid = "0:0.04:2";

  const Profile &reference_1D()
  {
    static const Profile p = [] {
      const auto json = make_json(r_grid, "0:1:1");
      return run<Model1D, 1>(json, Model1D(json), "invariant_kt_2D_1D");
    }();
    return p;
  }
} // namespace

TEST_CASE("Polar O(2)xO(2) from an O(4)-symmetric start follows the 1D O(4) flow", "[2d][FV][KT][invariants]")
{
  const auto &ref = reference_1D();
  for (const std::string form : {"w", "W"}) {
    CAPTURE(form);
    const auto json = make_json(r_grid, s_grid, form);
    const auto polar = run<ModelPolar<false>, 2>(json, ModelPolar<false>(json), "invariant_kt_2D_polar");
    double dv = 0., w = 0.;
    for (const auto &[x, values] : polar) {
      dv = std::max(dv, std::abs(values[0] - interpolate_1D(ref, x[0])));
      w = std::max(w, std::abs(values[1]));
    }
    std::clog << std::setprecision(6) << "[invariant_kt_2D] polar (" << form << ") vs 1D: max |v - u| = " << dv
              << ", max |w| = " << w << '\n';
    // to the time-stepping tolerance; W = r w is O(1) where w is O(1 / r)
    CHECK(dv < 1e-7);
    CHECK(w < (form == "w" ? 1e-10 : 1e-9));
  }
}

namespace
{
  /// Residual and jacobian of one polar KT assembly at a perturbed state, at RG time t = 2.
  template <bool batched> struct PolarAssembly {
    using Model = ModelPolar<batched>;
    using Discretization = FV::Discretization<typename Model::Components, RectangularMeshSerial<2>, double>;
    using Assembler =
        FV::KurganovTadmor::Assembler<Discretization, Model, def::TVDReconstructor<2, def::MinModLimiter, double>>;
    using VectorType = typename Discretization::VectorType;

    ConfigTree json;
    Model model;
    RectangularMeshSerial<2> mesh;
    Discretization discretization;
    Assembler assembler;
    VectorType u, u_dot;
    SparseMatrix<double> J;

    PolarAssembly()
        : json(make_json("0:0.05:2", s_grid)), model(json, 0.05), mesh(Config::ConfigurationMesh<2>(json)),
          discretization(mesh, json), assembler(discretization, model, json)
    {
      kt_regression::ensure_logger();
      FV::FlowingVariables<Discretization> state(discretization);
      state.interpolate(model);
      u = state.spatial_data();
      for (size_t i = 0; i < u.size(); ++i)
        u[i] += 0.01 * std::sin(1.7 * double(i));
      u_dot.reinit(u.size());
      J.reinit(assembler.get_sparsity_pattern_jacobian());
      assembler.set_time(2.);
    }

    VectorType residual()
    {
      VectorType r(u.size());
      assembler.residual(r, u, 1., u_dot, 0.);
      return r;
    }
    const SparseMatrix<double> &jacobian()
    {
      J = 0.;
      assembler.jacobian(J, u, 1., u_dot, 0., 0.);
      return J;
    }
  };
} // namespace

TEST_CASE("Polar potential_flux_batch assembles as the per-point path", "[2d][FV][KT][invariants]")
{
  PolarAssembly<true> batched;
  PolarAssembly<false> per_point;
  const auto r_b = batched.residual(), r_p = per_point.residual();
  auto dr = r_b;
  dr -= r_p;
  REQUIRE(r_p.linfty_norm() > 0.);
  CHECK(dr.linfty_norm() <= 1e-12 * r_p.linfty_norm());

  const auto &J_p = per_point.jacobian();
  const auto &J_b = batched.jacobian();
  double worst = 0., scale = 0.;
  for (auto it = J_p.begin(); it != J_p.end(); ++it) {
    worst = std::max(worst, std::abs(it->value() - J_b.el(it->row(), it->column())));
    scale = std::max(scale, std::abs(it->value()));
  }
  REQUIRE(scale > 0.);
  CHECK(worst <= 1e-12 * scale);
}

TEST_CASE("Cartesian and polar maps agree on an O(2)xO(2) flow", "[2d][FV][KT][invariants]")
{
  const auto &ref = reference_1D();
  const auto json_c = make_json(c_grid, c_grid);
  const auto json_p = make_json(r_grid, s_grid);

  // O(4)-symmetric: the Cartesian map against the 1D flow
  {
    const auto cart = run<ModelCartesian, 2>(json_c, ModelCartesian(json_c), "invariant_kt_2D_cart");
    double d = 0.;
    for (const auto &[x, values] : cart) {
      const double rho = x[0] + x[1];
      if (rho > 0.8 * rho_max) continue;
      const double u = interpolate_1D(ref, rho);
      d = std::max({d, std::abs(values[0] - u), std::abs(values[1] - u)});
    }
    std::clog << "[invariant_kt_2D] Cartesian vs 1D: max |u_a - u| = " << d << '\n';
    CHECK(d < 2e-2);
  }

  // explicitly broken, U_Lambda + delta rho_2: Cartesian against polar at the Cartesian cell centres
  {
    const double delta = 0.1;
    const auto cart = run<ModelCartesian, 2>(json_c, ModelCartesian(json_c, delta), "invariant_kt_2D_cart_b");
    const auto polar = run<ModelPolar<false>, 2>(json_p, ModelPolar<false>(json_p, delta), "invariant_kt_2D_polar_b");
    double d = 0.;
    for (const auto &[x, values] : cart) {
      const double r = x[0] + x[1], s = x[1] / r;
      if (r > 0.8 * rho_max || r < 0.05 || s < 0.04 || s > 0.96) continue;
      const double v = interpolate_2D(polar, r, s, 0, 0.01, 0.0625), w = interpolate_2D(polar, r, s, 1, 0.01, 0.0625);
      d = std::max({d, std::abs(values[0] - (v - s * w)), std::abs(values[1] - (v + (1. - s) * w))});
    }
    std::clog << "[invariant_kt_2D] broken: Cartesian vs polar: max |dU_a| = " << d << '\n';
    CHECK(d < 2e-2);
  }
}
