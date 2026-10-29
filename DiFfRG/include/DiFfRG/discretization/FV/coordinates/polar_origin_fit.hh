#pragma once

// standard library
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    /**
     * @brief The regular expansion of a smooth potential at the origin of the polar grid, fitted to the cells nearest
     * to it. Used by PolarInvariants for the curvatures close to \f$r = 0\f$.
     *
     * On the polar grid \f$\rho_1 = r(1 - s)\f$, \f$\rho_2 = rs\f$ every \f$s\f$ maps to the origin at \f$r = 0\f$. A
     * smooth potential has
     * \f[
     *   U = U_0 + r\,(a + d\,s) + r^2\,(c_0 + c_1 s + c_2 s^2) + O(r^3)\,,\qquad a = U_1(0)\,,\quad d = U_2(0) -
     * U_1(0)\,,
     * \f]
     * so the evolved fields are \f$v = \partial_r U = a + ds + 2r(c_0 + c_1 s + c_2 s^2)\f$ and
     * \f$W = \partial_s U = rd + r^2(c_1 + 2c_2 s)\f$, and the curvatures at the origin are
     * \f$U_{11} = 2c_0\f$, \f$U_{12} = 2c_0 + c_1\f$, \f$U_{22} = 2(c_0 + c_1 + c_2)\f$.
     *
     * Why: a first-row cell is a wedge of width \f$r\Delta s \ll \Delta r\f$, and the finite-difference curvatures
     * contain \f$s\f$-differences divided by \f$r\f$. Grid noise of the fields is amplified by \f$1/(r\Delta s)\f$
     * there and can exceed the curvature itself. The fit uses all cells of the first rows at once and has no such
     * amplification.
     *
     * fit() solves the least-squares problem for \f$(a, d, c_0, c_1, c_2)\f$ over the cells of the first `rows`
     * distinct \f$r\f$ values, two equations per cell (\f$v\f$ and \f$W/r\f$), with scaled columns.
     */
    class PolarOriginFit
    {
    public:
      /// The fitted coefficients.
      struct Coefficients {
        double a = std::numeric_limits<double>::quiet_NaN(), d = 0., c0 = 0., c1 = 0., c2 = 0.;
        /// Whether the fit succeeded; an invalid fit is not used.
        bool valid = false;
        /// Number of cells fitted.
        unsigned int cells = 0;
        /// rms misfit of \f$v\f$ and \f$W/r\f$ over the fitted cells.
        double rms = 0.;
        /// Largest \f$r\f$ of the fitted cells.
        double r_max = 0.;
      };

      /// One cell of the fit: its centre \f$(r, s)\f$ and \f$v\f$, \f$W\f$ there.
      struct Cell {
        double r, s, v, W;
      };

      /// \f$v\f$ of the expansion at \f$(r, s)\f$.
      static double v(const Coefficients &f, const double r, const double s)
      {
        return f.a + f.d * s + 2. * r * (f.c0 + f.c1 * s + f.c2 * s * s);
      }
      /// \f$W\f$ of the expansion at \f$(r, s)\f$.
      static double W(const Coefficients &f, const double r, const double s)
      {
        return r * f.d + r * r * (f.c1 + 2. * f.c2 * s);
      }
      /// \f$(U_1, U_2)\f$ at the origin.
      static std::array<double, 2> gradient(const Coefficients &f) { return {{f.a, f.a + f.d}}; }
      /// \f$(U_{11}, U_{12}, U_{22})\f$ at the origin.
      static std::array<double, 3> curvatures(const Coefficients &f)
      {
        return {{2. * f.c0, 2. * f.c0 + f.c1, 2. * (f.c0 + f.c1 + f.c2)}};
      }

      /**
       * @brief Fit the expansion to the cells with the @p rows smallest distinct \f$r > 0\f$.
       *
       * Equations per cell, with \f$x = r/R\f$, \f$R\f$ the largest fitted \f$r\f$ and \f$c_i = C_i/R\f$:
       * \f$a + ds + 2x(C_0 + C_1 s + C_2 s^2) = v\f$ and \f$d + x(C_1 + 2C_2 s) = W/r\f$.
       * Invalid if there are fewer than @p rows distinct radii or fewer than three cells.
       */
      static Coefficients fit(const std::vector<Cell> &cells, const unsigned int rows)
      {
        Coefficients f;
        if (rows == 0 || cells.empty()) return f;
        std::vector<double> rs;
        rs.reserve(cells.size());
        for (const auto &c : cells)
          if (c.r > 0.) rs.push_back(c.r);
        std::sort(rs.begin(), rs.end());
        rs.erase(std::unique(rs.begin(), rs.end(),
                             [](double x, double y) { return std::abs(x - y) <= 1e-12 * std::max(x, y); }),
                 rs.end());
        if (rs.size() < rows) return f;
        const double R = rs[rows - 1] * (1. + 1e-12);

        std::array<std::array<double, 5>, 5> A{};
        std::array<double, 5> b{};
        const auto add = [&](const std::array<double, 5> &row, const double rhs) {
          for (int i = 0; i < 5; ++i) {
            b[i] += row[i] * rhs;
            for (int j = 0; j < 5; ++j)
              A[i][j] += row[i] * row[j];
          }
        };
        unsigned int n = 0;
        for (const auto &c : cells) {
          if (!(c.r > 0.) || !(c.r <= R)) continue;
          const double x = c.r / R, s = c.s;
          add({{1., s, 2. * x, 2. * x * s, 2. * x * s * s}}, c.v);
          add({{0., 1., 0., x, 2. * x * s}}, c.W / c.r);
          ++n;
        }
        if (n < 3) return f;

        // Gaussian elimination with partial pivoting
        std::array<double, 5> sol{};
        for (int col = 0; col < 5; ++col) {
          int piv = col;
          for (int i = col + 1; i < 5; ++i)
            if (std::abs(A[i][col]) > std::abs(A[piv][col])) piv = i;
          if (!(std::abs(A[piv][col]) > 0.)) return f;
          std::swap(A[piv], A[col]);
          std::swap(b[piv], b[col]);
          for (int i = col + 1; i < 5; ++i) {
            const double m = A[i][col] / A[col][col];
            for (int j = col; j < 5; ++j)
              A[i][j] -= m * A[col][j];
            b[i] -= m * b[col];
          }
        }
        for (int i = 4; i >= 0; --i) {
          double s = b[i];
          for (int j = i + 1; j < 5; ++j)
            s -= A[i][j] * sol[j];
          sol[i] = s / A[i][i];
        }

        f.a = sol[0];
        f.d = sol[1];
        f.c0 = sol[2] / R;
        f.c1 = sol[3] / R;
        f.c2 = sol[4] / R;
        double e2 = 0.;
        for (const auto &c : cells) {
          if (!(c.r > 0.) || !(c.r <= R)) continue;
          e2 += std::pow(v(f, c.r, c.s) - c.v, 2) + std::pow((W(f, c.r, c.s) - c.W) / c.r, 2);
        }
        f.cells = n;
        f.rms = std::sqrt(e2 / (2. * n));
        f.r_max = R;
        f.valid = std::isfinite(f.a) && std::isfinite(f.d) && std::isfinite(f.c0) && std::isfinite(f.c1) &&
                  std::isfinite(f.c2);
        return f;
      }

      /**
       * @brief Weight of the fitted curvatures at radius @p r for a blend radius @p r_c: 1 at \f$r = 0\f$, 0 from
       * \f$r_c\f$ on, smoothstep \f$1 - x^2(3 - 2x)\f$, \f$x = r/r_c\f$, in between. 0 if \f$r_c \le 0\f$.
       */
      static double blend_weight(const double r, const double r_c)
      {
        if (!(r_c > 0.) || !(r < r_c)) return 0.;
        const double x = std::max(r, 0.) / r_c;
        return 1. - x * x * (3. - 2. * x);
      }
    };
  } // namespace FV
} // namespace DiFfRG
