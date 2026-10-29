#pragma once

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/FV/coordinates/axis_widths.hh>
#include <DiFfRG/discretization/FV/coordinates/invariant_coordinates.hh>
#include <DiFfRG/discretization/FV/coordinates/polar_origin_fit.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>

// standard library
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    /// The second evolved field of PolarInvariants.
    enum class PolarForm {
      /// \f$w = U_2 - U_1\f$, regular at \f$r = 0\f$ (recommended).
      w,
      /// \f$W = \partial_s U = r w\f$.
      W
    };

    // clang-format off
    /**
     * @brief Polar coordinates for a potential of two invariants: \f$r = \rho_1 + \rho_2\f$, \f$s = \rho_2/r \in [0, 1]\f$.
     *
     * \f$\rho_1 = r(1 - s)\f$, \f$\rho_2 = rs\f$; the grid's x axis is \f$r \ge 0\f$, its y axis \f$s\f$ (within
     * \f$[0, 1]\f$). The lines \f$s\f$ = const are rays through the origin, so \f$v = \partial_r U = (1 - s)U_1 + sU_2\f$
     * is the derivative along the ray, and near the \f$O(N_1 + N_2)\f$-symmetric point the radial diffusion of the flow
     * is along a grid axis (on CartesianInvariants it is diagonal; see invariant_coordinates.hh).
     *
     * ### Fields and equations
     * | Form | fields | equations |
     * |------|--------|-----------|
     * | PolarForm::w | \f$v = \partial_r U\f$, \f$w = U_2 - U_1\f$ | \f$\partial_t v + \partial_r\mathcal F = 0\f$, \f$\partial_t w + \partial_s(\mathcal F/r) = 0\f$ |
     * | PolarForm::W | \f$v\f$, \f$W = \partial_s U = rw\f$ | \f$\partial_t v + \partial_r\mathcal F = 0\f$, \f$\partial_t W + \partial_s\mathcal F = 0\f$ |
     *
     * The \f$w\f$ equation is conservative in \f$s\f$ because \f$r\f$ is constant on \f$s\f$-faces; flux_weight() is
     * \f$1/r_e\f$ for \f$w\f$, \f$r_e = \max(r, r_{\rm reg})\f$, \f$r_{\rm reg}\f$ the centre of the first \f$r\f$ cell.
     * In the W form \f$U_2 - U_1 = W/r\f$ is \f$0/0\f$ at the origin, and \f$W/r_{\rm reg} \approx 0\f$ there makes the
     * origin \f$O(N_1 + N_2)\f$ symmetric even when it is not. The w form has no such problem; prefer it.
     *
     * ### Conversion
     * \f$U_1 = v - sw\f$, \f$U_2 = v + (1 - s)w\f$; with \f$Y = \tfrac12\big(w_r + (v_s - w)/r_e\big)\f$ (w form; W form:
     * \f$Y = (W_r + v_s)/(2r_e) - W/r_e^2\f$, \f$w_s/r_e \to W_s/r_e^2\f$)
     * \f[
     *   U_{11} = v_r - 2sY + s^2\frac{w_s}{r_e}\,,\quad
     *   U_{12} = v_r + (1 - 2s)Y - s(1 - s)\frac{w_s}{r_e}\,,\quad
     *   U_{22} = v_r + 2(1 - s)Y + (1 - s)^2\frac{w_s}{r_e}\,.
     * \f]
     * Subscripts \f$r, s\f$ are KT gradients. The flow of an \f$O(N_1)\times O(N_2)\f$ model needs only
     * \f$\rho_a U_{ab}\f$, finite as \f$r \to 0\f$. third() gives \f$U_{abc}\f$ by the same chain rule.
     *
     * ### Compatibility damping
     * A gradient satisfies \f$c = \partial_s v - \partial_r W = 0\f$ (w form: \f$c = v_s - w - r w_r\f$), and the
     * curvatures contain \f$c/r\f$. damping_flux() adds
     * \f[
     *   \partial_t v \mathrel{+}= \partial_s(\nu_s c)\,,\qquad \partial_t W \mathrel{-}= \partial_r(\nu_r c)\ \text{(W form only)}\,,\qquad
     *   \nu_a = \mathrm{rate}(r)\,h_a^2\,,\quad
     *   \mathrm{rate}(r) = c_d\Big(1 + b\,\max\big(0, 1 - \tfrac{r}{n\,h_{r,1}}\big)^2\Big)\,,
     * \f]
     * with \f$c_d\f$ = `damping`, \f$b\f$ = `origin_boost`, \f$n\f$ = `origin_cells`. It is zero on gradients and
     * dissipative for grid-scale \f$s\f$-oscillations of \f$c\f$. There is no damping flux through the domain boundary
     * (\f$s\f$ at the ends of its axis; W form also \f$r = 0\f$ and \f$r_{\max}\f$), where \f$c\f$ comes from ghost cells.
     * The boost zone must cover the band of \f$r\f$ in which the origin is noisy.
     *
     * ### Origin curvatures
     * With `origin_fit_r` \f$= r_c > 0\f$, the curvatures at \f$r < r_c\f$ are blended from those of the regular
     * expansion at the origin (PolarOriginFit, weight 1 at \f$r = 0\f$) to the finite-difference ones (weight 0 at
     * \f$r_c\f$). \f$U_a\f$ stay local. The fit is refreshed by update() once per accepted time step (from
     * def::GradientPotential) and is frozen in between, so that residual and jacobian of one step see the same fit;
     * it is only used from RG time \f$t \ge\f$ `origin_fit_t_min` on (in the UV \f$v\f$ may swamp the \f$r^2\f$ terms). A callback
     * set by set_post_fit() can adjust the coefficients (e.g. keep fitted gaps positive). state() takes a flag to
     * switch the blend off, e.g. for readouts at a point inside the first cell.
     *
     * ### Grid
     * The origin should be coarse rather than fine: \f$r\Delta s \ll \Delta r\f$ amplifies \f$s\f$-noise. Non-uniform
     * \f$s\f$ grids were seen to oscillate at the spacing change.
     */
    // clang-format on
    class PolarInvariants
    {
    public:
      static constexpr int dim = 2;

      /// Numerical parameters, see the table in invariant_coordinates.hh.
      struct Parameters {
        PolarForm form = PolarForm::w;
        double damping = 20.;
        double origin_boost = 100.;
        double origin_cells = 4.;
        double origin_fit_r = 0.;
        unsigned int origin_fit_rows = 16;
        double origin_fit_t_min = 0.;
      };

      /// Called after every origin fit with the coefficients and the fitted cells; may change the coefficients.
      using PostFit = std::function<void(PolarOriginFit::Coefficients &, const std::vector<PolarOriginFit::Cell> &)>;

      /// From the cell widths along \f$r\f$ and \f$s\f$.
      PolarInvariants(const AxisWidths &r_axis, const AxisWidths &s_axis, const Parameters &parameters)
          : r_axis(r_axis), s_axis(s_axis), prm(parameters), m_r_reg(r_axis.lower() + 0.5 * r_axis.first())
      {
        if (r_axis.lower() < 0.) throw std::invalid_argument("PolarInvariants: the r axis must start at r >= 0.");
        if (s_axis.lower() < 0. || s_axis.upper() > 1. + 1e-12)
          throw std::invalid_argument("PolarInvariants: the s axis must lie within [0, 1].");
      }

      /// From the mesh in `/discretization/grid/` and the parameters in `/discretization/invariant_coordinates/`.
      explicit PolarInvariants(const ConfigTree &json)
          : PolarInvariants(AxisWidths(Config::ConfigurationMesh<dim>(json), 0),
                            AxisWidths(Config::ConfigurationMesh<dim>(json), 1), parameters(json))
      {
      }

      /// The parameters from `/discretization/invariant_coordinates/`.
      static Parameters parameters(const ConfigTree &json)
      {
        const std::string root = "/discretization/invariant_coordinates/";
        Parameters p;
        const auto form = json.get_string(root + "polar_form", "w");
        if (form != "w" && form != "W")
          throw std::invalid_argument("PolarInvariants: polar_form must be \"w\" or \"W\", not \"" + form + "\".");
        p.form = form == "w" ? PolarForm::w : PolarForm::W;
        p.damping = json.get_double(root + "damping", p.damping);
        p.origin_boost = json.get_double(root + "origin_boost", p.origin_boost);
        p.origin_cells = json.get_double(root + "origin_cells", p.origin_cells);
        p.origin_fit_r = json.get_double(root + "origin_fit_r", p.origin_fit_r);
        p.origin_fit_rows = json.get_uint(root + "origin_fit_rows", p.origin_fit_rows);
        p.origin_fit_t_min = json.get_double(root + "origin_fit_t_min", p.origin_fit_t_min);
        return p;
      }

      /// \f$(\rho_1, \rho_2) = (r(1 - s), rs)\f$, with \f$r\f$ clamped at 0 (ghost cells).
      std::array<double, dim> invariants(const Point<dim> &x) const
      {
        const double r = std::max(x[0], 0.), s = x[1];
        return {{r * (1. - s), r * s}};
      }

      /// \f$\rho_a\f$, \f$U_1 = v - sw\f$, \f$U_2 = v + (1 - s)w\f$.
      template <typename NT>
      InvariantState<dim, NT> values(const Point<dim> &x, const InvariantFields<dim, NT> &u) const
      {
        const double s = x[1];
        const NT w = prm.form == PolarForm::w ? u[1] : NT(u[1] / r_e(x[0]));
        InvariantState<dim, NT> state;
        state.rho = invariants(x);
        state.U1 = {{u[0] - s * w, u[0] + (1. - s) * w}};
        for (auto &row : state.U2)
          row.fill(NT(0.));
        return state;
      }

      /**
       * @brief \f$\rho_a\f$, \f$U_a\f$ and \f$U_{ab}\f$ (see the class description).
       * @param origin_blend whether the origin curvatures of the fit may be blended in (if the fit is on and valid).
       */
      template <typename NT>
      InvariantState<dim, NT> state(const Point<dim> &x, const InvariantFields<dim, NT> &u,
                                    const InvariantFieldGradients<dim, NT> &du, const bool origin_blend = true) const
      {
        const double s = x[1], re = r_e(x[0]);
        auto state = values<NT>(x, u);
        const NT &vr = du[0][0], &vs = du[0][1], &ur = du[1][0], &us = du[1][1];
        // Y, and the divisor of the s-curvature term: w_s / r_e (w form), W_s / r_e^2 (W form)
        const NT Y =
            prm.form == PolarForm::w ? NT(0.5 * (ur + (vs - u[1]) / re)) : NT(0.5 * (ur + vs) / re - u[1] / (re * re));
        const double d = prm.form == PolarForm::w ? re : re * re;
        state.U2[0][0] = vr - 2. * s * Y + s * s * us / d;
        state.U2[0][1] = vr + (1. - 2. * s) * Y - s * (1. - s) * us / d;
        state.U2[1][1] = vr + 2. * (1. - s) * Y + (1. - s) * (1. - s) * us / d;

        if (origin_blend && origin_active && fit.valid) {
          const double lambda = PolarOriginFit::blend_weight(x[0], prm.origin_fit_r);
          if (lambda > 0.) {
            const auto k2 = PolarOriginFit::curvatures(fit);
            state.U2[0][0] = lambda * k2[0] + (1. - lambda) * state.U2[0][0];
            state.U2[0][1] = lambda * k2[1] + (1. - lambda) * state.U2[0][1];
            state.U2[1][1] = lambda * k2[2] + (1. - lambda) * state.U2[1][1];
          }
        }
        state.U2[1][0] = state.U2[0][1];
        return state;
      }

      /**
       * @brief \f$U_{abc}\f$ by the chain rule from the grid derivatives \f$D_{ij} = \partial_r^i\partial_s^j U\f$,
       * with the mixed ones symmetrised over the two fields; \f$1/r\f$ at \f$r_e\f$. No origin blend.
       */
      InvariantThirdDerivatives<dim> third(const Point<dim> &x, const InvariantFields<dim, double> &u,
                                           const InvariantFieldGradients<dim, double> &du,
                                           const InvariantFieldHessians<dim> &d2u) const
      {
        const double re = r_e(x[0]), s = x[1];
        // the W-form grid derivatives (w form: W = re w)
        double W = u[1], Wr = du[1][0], Ws = du[1][1], Wrr = d2u[1][0][0], Wrs = d2u[1][0][1], Wss = d2u[1][1][1];
        if (prm.form == PolarForm::w) {
          const double w = u[1], wr = du[1][0], ws = du[1][1];
          W = re * w;
          Wr = w + re * wr;
          Ws = re * ws;
          Wrr = 2. * wr + re * d2u[1][0][0];
          Wrs = ws + re * d2u[1][0][1];
          Wss = re * d2u[1][1][1];
        }
        const double vs = du[0][1], vrr = d2u[0][0][0], vrs = d2u[0][0][1], vss = d2u[0][1][1];
        const double D01 = W, D11 = 0.5 * (vs + Wr), D02 = Ws, D30 = vrr, D21 = 0.5 * (vrs + Wrr),
                     D12 = 0.5 * (vss + Wrs), D03 = Wss;
        const double r2 = re * re, r3 = r2 * re, s2 = s * s, s3 = s2 * s;
        const double U111 = (-3. * D21 * r2 * s + D30 * r3 + 3. * re * s * (2. * D11 + D12 * s) -
                             s * (6. * D01 + 6. * D02 * s + D03 * s2)) /
                            r3;
        const double U112 =
            (-6. * D01 * s + 2. * D01 - 6. * D02 * s2 + 4. * D02 * s - D03 * s3 + D03 * s2 + D21 * r2 * (1. - 3. * s) +
             D30 * r3 + re * (6. * D11 * s - 2. * D11 + 3. * D12 * s2 - 2. * D12 * s)) /
            r3;
        const double U122 = (-6. * D01 * s + 4. * D01 - 6. * D02 * s2 + 8. * D02 * s - 2. * D02 - D03 * s3 +
                             2. * D03 * s2 - D03 * s + D21 * r2 * (2. - 3. * s) + D30 * r3 +
                             re * (6. * D11 * s - 4. * D11 + 3. * D12 * s2 - 4. * D12 * s + D12)) /
                            r3;
        const double U222 = (-6. * D01 * s + 6. * D01 - 6. * D02 * s2 + 12. * D02 * s - 6. * D02 - D03 * s3 +
                             3. * D03 * s2 - 3. * D03 * s + D03 + 3. * D21 * r2 * (1. - s) + D30 * r3 +
                             3. * re * (2. * D11 * s - 2. * D11 + D12 * s2 - 2. * D12 * s + D12)) /
                            r3;
        InvariantThirdDerivatives<dim> U3;
        U3[0][0][0] = U111;
        U3[0][0][1] = U3[0][1][0] = U3[1][0][0] = U112;
        U3[0][1][1] = U3[1][0][1] = U3[1][1][0] = U122;
        U3[1][1][1] = U222;
        return U3;
      }

      /// 1 for \f$v\f$; for the second field 1 (W form) or \f$1/r_e\f$ (w form).
      double flux_weight(const Point<dim> &x, const unsigned int i) const
      {
        return i == 1 && prm.form == PolarForm::w ? 1. / r_e(x[0]) : 1.;
      }

      /// The compatibility residual \f$c\f$ (zero for a gradient).
      template <typename NT>
      NT compatibility(const Point<dim> &x, const InvariantFields<dim, NT> &u,
                       const InvariantFieldGradients<dim, NT> &du) const
      {
        if (prm.form == PolarForm::w) return du[0][1] - u[1] - std::max(x[0], 0.) * du[1][0];
        return du[0][1] - du[1][0];
      }

      /// The damping coefficients \f$(\nu_r, \nu_s)\f$ at @p x (0 at the boundaries, see the class description).
      std::array<double, 2> damping_coefficients(const Point<dim> &x) const
      {
        if (!(prm.damping > 0.)) return {{0., 0.}};
        const double r = x[0], s = x[1];
        const double hr = r_axis.width(r), hs = s_axis.width(s);
        const double f = std::max(0., 1. - (r - r_axis.lower()) / (prm.origin_cells * r_axis.first()));
        const double rate = prm.damping * (1. + prm.origin_boost * f * f);
        const bool r_edge = r < r_axis.lower() + 0.25 * r_axis.first() || r > r_axis.upper() - 0.25 * r_axis.last();
        const bool s_edge = s < s_axis.lower() + 1e-12 || s > s_axis.upper() - 1e-12;
        return {{r_edge || prm.form == PolarForm::w ? 0. : rate * hr * hr, s_edge ? 0. : rate * hs * hs}};
      }

      /// Adds the compatibility damping flux: `D[0][1] -= nu_s c`, and in the W form `D[1][0] += nu_r c`.
      template <typename NT>
      void damping_flux(InvariantFieldGradients<dim, NT> &D, const Point<dim> &x, const InvariantFields<dim, NT> &u,
                        const InvariantFieldGradients<dim, NT> &du) const
      {
        const auto nu = damping_coefficients(x);
        if (!(nu[0] > 0.) && !(nu[1] > 0.)) return;
        const NT c = compatibility<NT>(x, u, du);
        D[0][1] -= nu[1] * c;
        D[1][0] += nu[0] * c;
      }

      /// \f$v = f(r)\f$, \f$w = W = 0\f$, for an \f$O(N_1 + N_2)\f$-symmetric \f$U\f$ with \f$U' = f\f$.
      template <typename F> InvariantFields<dim, double> initial_fields(const Point<dim> &x, const F &f) const
      {
        return {{f(std::max(x[0], 0.)), 0.}};
      }

      /// For the second field \f$|v|\f$ (w form) or \f$r|v|\f$ (W form): its round-off from \f$s\f$-differences of
      /// \f$v\f$.
      double tolerance_scale(const Point<dim> &x, const InvariantFields<dim, double> &u, const unsigned int i) const
      {
        if (i != 1) return 0.;
        return (prm.form == PolarForm::w ? 1. : std::max(x[0], 0.)) * std::abs(u[0]);
      }

      /**
       * @brief Refit the origin expansion from the cell centres of @p sample, whose components @p fields hold
       * \f$(v, w)\f$ or \f$(v, W)\f$; the fit is used if \f$t \ge\f$ `origin_fit_t_min`. A no-op if `origin_fit_r` is
       * 0.
       *
       * Called by def::GradientPotential::needs_restart once per accepted time step; const because the fit is
       * numerical state of the map, not of the solution.
       */
      template <typename NT>
      void update(const double t, const SolutionSample<dim, NT> &sample,
                  const std::array<unsigned int, dim> &fields) const
      {
        if (!(prm.origin_fit_r > 0.)) return;
        std::vector<PolarOriginFit::Cell> cells;
        cells.reserve(sample.size());
        for (const auto &e : sample) {
          const double r = e.point[0], s = e.point[1];
          if (!(r > 0.)) continue;
          const double v = double(e.values[fields[0]]), u1 = double(e.values[fields[1]]);
          cells.push_back({r, s, v, prm.form == PolarForm::w ? r * u1 : u1});
        }
        update(t, cells);
      }

      /**
       * @brief Refit the origin expansion from @p cells ($r, s, v, W$ per cell centre), e.g. for a model that
       * drives its own time stepping and refreshes the fit itself. Same rules as the SolutionSample overload.
       */
      void update(const double t, const std::vector<PolarOriginFit::Cell> &cells) const
      {
        if (!(prm.origin_fit_r > 0.)) return;
        origin_active = t >= prm.origin_fit_t_min;
        auto f = PolarOriginFit::fit(cells, prm.origin_fit_rows);
        if (f.valid && post_fit) post_fit(f, cells);
        fit = f;
      }

      /// The origin fit in use.
      const PolarOriginFit::Coefficients &origin_fit() const { return fit; }
      /// Replace the origin fit (e.g. when restoring a state).
      void set_origin_fit(const PolarOriginFit::Coefficients &f, const bool active = true) const
      {
        fit = f;
        origin_active = active;
      }
      /// Set the callback run after every fit.
      void set_post_fit(PostFit callback) { post_fit = std::move(callback); }

      /// \f$r_{\rm reg}\f$, the centre of the first \f$r\f$ cell.
      double r_reg() const { return m_r_reg; }
      /// \f$r_e = \max(r, r_{\rm reg})\f$.
      double r_e(const double r) const { return std::max(r, m_r_reg); }
      /// The parameters.
      const Parameters &parameters() const { return prm; }
      /// The cell widths along \f$r\f$ (@p a = 0) or \f$s\f$ (@p a = 1).
      const AxisWidths &axis(const unsigned int a) const { return a == 0 ? r_axis : s_axis; }

    private:
      AxisWidths r_axis, s_axis;
      Parameters prm;
      double m_r_reg;
      PostFit post_fit;
      mutable PolarOriginFit::Coefficients fit;
      mutable bool origin_active = false;
    };
    static_assert(InvariantCoordinateMap<PolarInvariants>);
  } // namespace FV
} // namespace DiFfRG
