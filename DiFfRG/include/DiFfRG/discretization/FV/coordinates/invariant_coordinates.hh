#pragma once

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/FV/coordinates/axis_widths.hh>
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// standard library
#include <array>
#include <concepts>
#include <cstddef>

// clang-format off
/**
 * @file invariant_coordinates.hh
 * @brief Coordinate maps for potentials of several invariants on the Kurganov-Tadmor (KT) assembler.
 *
 * ### The problem
 * A potential \f$U(\rho_1, \dots, \rho_N)\f$ of \f$N\f$ invariants (e.g. \f$\rho_1 = \tfrac12(\phi_1^2 + \phi_2^2)\f$,
 * \f$\rho_2 = \tfrac12(\phi_3^2 + \phi_4^2)\f$ in an \f$O(2)\times O(2)\f$ model) flows as
 * \f[
 *   \partial_t U = -\mathcal F(\rho, U_a, U_{ab})\,,\qquad U_a = \partial_{\rho_a} U\,,\quad U_{ab} = \partial_{\rho_a}\partial_{\rho_b} U\,.
 * \f]
 * KT has no mixed second derivative in the source, so \f$U\f$ itself cannot be evolved. Its gradient can: in any grid
 * coordinates \f$x^i\f$ partial derivatives commute, so
 * \f[
 *   \partial_t u_i + \partial_i \mathcal F = 0\,,\qquad u_i = \partial_{x^i} U\,,
 * \f]
 * without metric factors. Each gradient component \f$u_i\f$ is one FE function carrying \f$\mathcal F\f$ along its own
 * axis \f$e_i\f$ only. KT needs no geometry; all of it sits in converting the evolved fields and their KT derivatives
 * to \f$\rho_a, U_a, U_{ab}\f$ for \f$\mathcal F\f$. This is what a coordinate map does. def::GradientPotential
 * (model/gradient_potential.hh) builds the KT fluxes of a model from one.
 *
 * The price: \f$N\f$ independent fields are not automatically a gradient. Their curl (or, in curvilinear
 * coordinates, the compatibility residual) is a mode the exact flow never excites, but the discretisation can; each
 * map provides a damping flux that vanishes on gradients and dissipates that mode.
 *
 * ### Maps
 * | Map | grid \f$x\f$ | fields | use |
 * |-----|--------------|--------|-----|
 * | CartesianInvariants<N> | \f$\rho_1, \dots, \rho_N\f$ | \f$U_a\f$ | any \f$N\f$; reference |
 * | PolarInvariants (polar_invariants.hh) | \f$r = \rho_1 + \rho_2\f$, \f$s = \rho_2 / r\f$ | \f$v = \partial_r U\f$ and \f$w\f$ or \f$W\f$ | \f$N = 2\f$ near \f$O(N_1 + N_2)\f$ symmetry |
 *
 * Near the symmetric point the curvature part of the flow is a diffusion along \f$\rho_1 + \rho_2\f$. On the
 * Cartesian grid that direction is diagonal, and the central KT stencil leaves an odd-even mode along the
 * anti-diagonal undamped; the polar grid puts it on a grid axis.
 *
 * ### Interface (InvariantCoordinateMap)
 * Fields and their derivatives are passed in axis order: `u[i]` is the field carried along axis \f$i\f$, `du[i][j]`
 * \f$= \partial_{x^j} u_i\f$, `d2u[i]` the hessian of \f$u_i\f$.
 * | Member | What it returns |
 * |--------|-----------------|
 * | `static constexpr int dim` | number of invariants = mesh dimension |
 * | `invariants(x)` | \f$\rho_a\f$ at grid point \f$x\f$ |
 * | `potential_gradient(x, u)` | \f$\partial_{x^i} U\f$, the gradient of \f$U\f$ in the grid coordinates |
 * | `values(x, u)` | InvariantState with \f$\rho_a, U_a\f$ (\f$U_{ab} = 0\f$) |
 * | `state(x, u, du)` | InvariantState with \f$\rho_a, U_a, U_{ab}\f$ |
 * | `third(x, u, du, d2u)` | \f$U_{abc}\f$ |
 * | `flux_weight(x, i)` | factor on \f$\mathcal F\f$ in the flux of \f$u_i\f$ along \f$e_i\f$ |
 * | `damping_flux(D, x, u, du)` | adds the damping flux of the non-gradient mode to `D[i]` |
 * | `initial_fields(x, f)` | the fields of an \f$O(N)\f$-symmetric start, \f$U_a = f(\textstyle\sum_b \rho_b)\f$ |
 * | `tolerance_scale(x, u, i)` | a magnitude of \f$u_i\f$ for relative per-dof tolerances (0: none) |
 *
 * A map may also implement `update(t, sample, fields)`, called once per accepted time step with the solution at
 * the cell centres (see def::GradientPotential); PolarInvariants refits its origin expansion there.
 *
 * ### Parameters
 * Read by the ConfigTree constructors, under `/discretization/invariant_coordinates/`:
 * | Key | Default | Map | Meaning |
 * |-----|---------|-----|---------|
 * | `damping` | 20 | all | damping rate \f$c\f$, \f$\nu_a = c\,h_a^2\f$ (0: off) |
 * | `polar_form` | `"w"` | polar | second field: `"w"` (\f$U_2 - U_1\f$) or `"W"` (\f$\partial_s U\f$) |
 * | `origin_boost` | 100 | polar | damping boost at \f$r = 0\f$ |
 * | `origin_cells` | 4 | polar | boost zone in first-\f$r\f$-cell widths |
 * | `origin_fit_r` | 0 | polar | radius of the origin curvature blend (0: off) |
 * | `origin_fit_rows` | 16 | polar | \f$r\f$ rows of cells in the origin fit |
 * | `origin_fit_t_min` | 0 | polar | blend only from RG time \f$t \ge\f$ this on |
 */
// clang-format on

namespace DiFfRG
{
  namespace FV
  {
    using namespace dealii;

    /**
     * @brief A point of a potential of `dim` invariants: \f$\rho_a\f$, \f$U_a\f$ and \f$U_{ab}\f$ (symmetric, full
     * storage).
     */
    template <int dim, typename NT> struct InvariantState {
      /// The invariants \f$\rho_a\f$.
      std::array<double, dim> rho{};
      /// \f$U_a = \partial_{\rho_a} U\f$.
      std::array<NT, dim> U1{};
      /// \f$U_{ab} = \partial_{\rho_a}\partial_{\rho_b} U\f$; zero when built by values().
      std::array<std::array<NT, dim>, dim> U2{};
    };

    /// \f$U_{abc}\f$, fully symmetric, full storage.
    template <int dim> using InvariantThirdDerivatives = std::array<std::array<std::array<double, dim>, dim>, dim>;

    /// The fields of a coordinate map in axis order.
    template <int dim, typename NT> using InvariantFields = std::array<NT, dim>;
    /// Their gradients: `du[i][j]` = \f$\partial_{x^j} u_i\f$.
    template <int dim, typename NT> using InvariantFieldGradients = std::array<Tensor<1, dim, NT>, dim>;
    /// Their hessians: `d2u[i][j][k]` = \f$\partial_{x^j}\partial_{x^k} u_i\f$.
    template <int dim> using InvariantFieldHessians = std::array<Tensor<2, dim, double>, dim>;

    /**
     * @brief A coordinate map for the gradient form of a potential of several invariants, see invariant_coordinates.hh.
     */
    template <typename Map>
    concept InvariantCoordinateMap =
        requires(const Map &map, const Point<Map::dim> &x, const InvariantFields<Map::dim, double> &u,
                 const InvariantFieldGradients<Map::dim, double> &du, const InvariantFieldHessians<Map::dim> &d2u,
                 InvariantFieldGradients<Map::dim, double> &D) {
          { map.invariants(x) } -> std::convertible_to<std::array<double, Map::dim>>;
          { map.potential_gradient(x, u) } -> std::convertible_to<std::array<double, Map::dim>>;
          { map.template values<double>(x, u) } -> std::convertible_to<InvariantState<Map::dim, double>>;
          { map.template state<double>(x, u, du) } -> std::convertible_to<InvariantState<Map::dim, double>>;
          { map.third(x, u, du, d2u) } -> std::convertible_to<InvariantThirdDerivatives<Map::dim>>;
          { map.flux_weight(x, 0u) } -> std::convertible_to<double>;
          map.template damping_flux<double>(D, x, u, du);
          { map.tolerance_scale(x, u, 0u) } -> std::convertible_to<double>;
        };

    /**
     * @brief The identity map: the grid is \f$(\rho_1, \dots, \rho_N)\f$, the fields are \f$u_a = U_a\f$.
     *
     * \f$U_{ab} = \tfrac12(\partial_b u_a + \partial_a u_b)\f$ and \f$U_{abc}\f$ the symmetrised hessians. The damping
     * is that of the curl \f$\omega_{ab} = \partial_b u_a - \partial_a u_b\f$,
     * \f[
     *   \partial_t u_a \mathrel{+}= \sum_b \partial_b(\nu_b\,\omega_{ab})\,,\qquad \nu_b = c\,h_b^2\,,
     * \f]
     * which vanishes on gradients and removes \f$\int \nu\,\omega^2\f$. With wave_speed_blocks separating the fields
     * (def::GradientPotential does), KT does not dissipate transverse grid modes of \f$u_a\f$ at all; without the
     * damping a checkerboard in \f$u_2\f$ along \f$\rho_1\f$ is pumped by the \f$U_{12}\f$ coupling.
     *
     * @tparam dim_ number of invariants.
     */
    template <int dim_> class CartesianInvariants
    {
    public:
      static constexpr int dim = dim_;

      /// From the cell widths of every axis and the damping rate \f$c\f$.
      CartesianInvariants(const std::array<AxisWidths, dim> &widths, const double damping)
          : widths(widths), damping(damping)
      {
      }

      /// From the mesh in `/discretization/grid/` and `/discretization/invariant_coordinates/damping`.
      explicit CartesianInvariants(const ConfigTree &json)
          : CartesianInvariants(axis_widths(Config::ConfigurationMesh<dim>(json)),
                                json.get_double("/discretization/invariant_coordinates/damping", 20.))
      {
      }

      /// \f$\rho_a = x_a\f$.
      std::array<double, dim> invariants(const Point<dim> &x) const
      {
        std::array<double, dim> rho;
        for (int a = 0; a < dim; ++a)
          rho[a] = x[a];
        return rho;
      }

      /// \f$\partial_a U = u_a\f$.
      std::array<double, dim> potential_gradient(const Point<dim> & /*x*/, const InvariantFields<dim, double> &u) const
      {
        return u;
      }

      /// \f$\rho_a\f$, \f$U_a = u_a\f$.
      template <typename NT>
      InvariantState<dim, NT> values(const Point<dim> &x, const InvariantFields<dim, NT> &u) const
      {
        InvariantState<dim, NT> state;
        state.rho = invariants(x);
        state.U1 = u;
        for (auto &row : state.U2)
          row.fill(NT(0.));
        return state;
      }

      /// \f$\rho_a\f$, \f$U_a\f$, \f$U_{ab} = \tfrac12(\partial_b u_a + \partial_a u_b)\f$.
      template <typename NT>
      InvariantState<dim, NT> state(const Point<dim> &x, const InvariantFields<dim, NT> &u,
                                    const InvariantFieldGradients<dim, NT> &du) const
      {
        auto state = values<NT>(x, u);
        for (int a = 0; a < dim; ++a)
          for (int b = 0; b < dim; ++b)
            state.U2[a][b] = 0.5 * (du[a][b] + du[b][a]);
        return state;
      }

      /// \f$U_{abc}\f$: the hessians of the \f$u_a\f$, symmetrised over the three indices.
      InvariantThirdDerivatives<dim> third(const Point<dim> & /*x*/, const InvariantFields<dim, double> & /*u*/,
                                           const InvariantFieldGradients<dim, double> & /*du*/,
                                           const InvariantFieldHessians<dim> &d2u) const
      {
        InvariantThirdDerivatives<dim> U3;
        for (int a = 0; a < dim; ++a)
          for (int b = 0; b < dim; ++b)
            for (int c = 0; c < dim; ++c)
              U3[a][b][c] = (d2u[a][b][c] + d2u[b][c][a] + d2u[c][a][b]) / 3.;
        return U3;
      }

      /// Every field carries \f$\mathcal F\f$ unweighted.
      double flux_weight(const Point<dim> & /*x*/, const unsigned int /*i*/) const { return 1.; }

      /// Adds the curl damping flux, `D[a][b] -= nu_b omega_ab`, `D[b][a] += nu_a omega_ab`.
      template <typename NT>
      void damping_flux(InvariantFieldGradients<dim, NT> &D, const Point<dim> &x,
                        const InvariantFields<dim, NT> & /*u*/, const InvariantFieldGradients<dim, NT> &du) const
      {
        if (!(damping > 0.)) return;
        std::array<double, dim> nu;
        for (int a = 0; a < dim; ++a)
          nu[a] = damping * widths[a].width(x[a]) * widths[a].width(x[a]);
        for (int a = 0; a < dim; ++a)
          for (int b = a + 1; b < dim; ++b) {
            const NT omega = du[a][b] - du[b][a];
            D[a][b] -= nu[b] * omega;
            D[b][a] += nu[a] * omega;
          }
      }

      /// \f$u_a = f(\sum_b \rho_b)\f$ for an \f$O(N)\f$-symmetric \f$U\f$ with \f$U' = f\f$.
      template <typename F> InvariantFields<dim, double> initial_fields(const Point<dim> &x, const F &f) const
      {
        double rho = 0.;
        for (int a = 0; a < dim; ++a)
          rho += x[a];
        InvariantFields<dim, double> u;
        u.fill(f(rho));
        return u;
      }

      /// No relative tolerance.
      double tolerance_scale(const Point<dim> & /*x*/, const InvariantFields<dim, double> & /*u*/,
                             const unsigned int /*i*/) const
      {
        return 0.;
      }

      /// The cell widths along axis @p a.
      const AxisWidths &axis(const unsigned int a) const { return widths[a]; }

    private:
      static std::array<AxisWidths, dim> axis_widths(const Config::ConfigurationMesh<dim> &mesh)
      {
        std::array<AxisWidths, dim> w;
        for (int a = 0; a < dim; ++a)
          w[a] = AxisWidths(mesh, a);
        return w;
      }

      std::array<AxisWidths, dim> widths;
      double damping;
    };
    static_assert(InvariantCoordinateMap<CartesianInvariants<2>>);
  } // namespace FV
} // namespace DiFfRG
