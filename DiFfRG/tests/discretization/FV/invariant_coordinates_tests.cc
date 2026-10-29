#include <DiFfRG/discretization/FV/coordinates/axis_widths.hh>
#include <DiFfRG/discretization/FV/coordinates/invariant_coordinates.hh>
#include <DiFfRG/discretization/FV/coordinates/polar_invariants.hh>
#include <DiFfRG/discretization/FV/coordinates/polar_origin_fit.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>

#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <vector>

using namespace DiFfRG;
using dealii::Point;
using dealii::Tensor;

namespace
{
  /// A cubic potential of two invariants with all its derivatives.
  struct Cubic {
    double a1 = 0.3, a2 = -0.7, b11 = 1.1, b12 = -0.4, b22 = 0.8, c111 = 0.5, c112 = -0.9, c122 = 0.35, c222 = -0.6;

    std::array<double, 2> U1(double p, double q) const
    {
      return {{a1 + 2. * b11 * p + b12 * q + 3. * c111 * p * p + 2. * c112 * p * q + c122 * q * q,
               a2 + b12 * p + 2. * b22 * q + c112 * p * p + 2. * c122 * p * q + 3. * c222 * q * q}};
    }
    std::array<std::array<double, 2>, 2> U2(double p, double q) const
    {
      const double U11 = 2. * b11 + 6. * c111 * p + 2. * c112 * q;
      const double U12 = b12 + 2. * c112 * p + 2. * c122 * q;
      const double U22 = 2. * b22 + 2. * c122 * p + 6. * c222 * q;
      return {{{{U11, U12}}, {{U12, U22}}}};
    }
    FV::InvariantThirdDerivatives<2> U3() const
    {
      FV::InvariantThirdDerivatives<2> t;
      t[0][0][0] = 6. * c111;
      t[0][0][1] = t[0][1][0] = t[1][0][0] = 2. * c112;
      t[0][1][1] = t[1][0][1] = t[1][1][0] = 2. * c122;
      t[1][1][1] = 6. * c222;
      return t;
    }
  };

  using Field = std::function<std::array<double, 2>(double, double)>;

  /// Central finite-difference gradient and hessian of the two fields @p f at @p x.
  void differentiate(const Field &f, const Point<2> &x, FV::InvariantFields<2, double> &u,
                     FV::InvariantFieldGradients<2, double> &du, FV::InvariantFieldHessians<2> &d2u,
                     const double h = 1e-4)
  {
    const auto at = [&](double dx, double dy) { return f(x[0] + dx, x[1] + dy); };
    const auto c = at(0., 0.);
    for (int i = 0; i < 2; ++i) {
      u[i] = c[i];
      du[i][0] = (at(h, 0.)[i] - at(-h, 0.)[i]) / (2. * h);
      du[i][1] = (at(0., h)[i] - at(0., -h)[i]) / (2. * h);
      d2u[i][0][0] = (at(h, 0.)[i] - 2. * c[i] + at(-h, 0.)[i]) / (h * h);
      d2u[i][1][1] = (at(0., h)[i] - 2. * c[i] + at(0., -h)[i]) / (h * h);
      d2u[i][0][1] = d2u[i][1][0] = (at(h, h)[i] - at(h, -h)[i] - at(-h, h)[i] + at(-h, -h)[i]) / (4. * h * h);
    }
  }

  FV::AxisWidths uniform_axis(double lower, double upper, unsigned n)
  {
    return FV::AxisWidths(std::vector<double>(n, (upper - lower) / n), lower);
  }

  FV::PolarInvariants polar_map(FV::PolarForm form, double damping = 20.)
  {
    FV::PolarInvariants::Parameters p;
    p.form = form;
    p.damping = damping;
    return FV::PolarInvariants(uniform_axis(0., 1., 1000), uniform_axis(0., 1., 32), p);
  }
} // namespace

TEST_CASE("AxisWidths: cell, face, ghost and refined widths", "[FV][coordinates]")
{
  const std::vector<Config::GridAxis> x{Config::GridAxis(0., 0.1, 1.), Config::GridAxis(1., 0.5, 3.)};
  const std::vector<Config::GridAxis> y{Config::GridAxis(-1., 0.25, 1.)};
  const FV::AxisWidths w(Config::ConfigurationMesh<2>(0u, x, y), 0);
  CHECK(w.size() == 14);
  CHECK(w.lower() == Catch::Approx(0.));
  CHECK(w.upper() == Catch::Approx(3.));
  CHECK(w.width(0.55) == Catch::Approx(0.1));
  CHECK(w.width(1.0) == Catch::Approx(0.5)); // face between the two segments: the larger width
  CHECK(w.width(2.2) == Catch::Approx(0.5));
  CHECK(w.width(-0.05) == Catch::Approx(0.1)); // ghost
  CHECK(w.width(3.4) == Catch::Approx(0.5));

  const FV::AxisWidths wy(Config::ConfigurationMesh<2>(0u, x, y), 1);
  CHECK(wy.lower() == Catch::Approx(-1.));
  CHECK(wy.width(0.) == Catch::Approx(0.25));

  const FV::AxisWidths refined(Config::ConfigurationMesh<2>(1u, x, y), 0);
  CHECK(refined.size() == 28);
  CHECK(refined.width(0.55) == Catch::Approx(0.05));
  CHECK(refined.width(2.2) == Catch::Approx(0.25));
}

TEST_CASE("CartesianInvariants: chain rule and curl damping", "[FV][coordinates]")
{
  const Cubic U;
  const FV::CartesianInvariants<2> map({uniform_axis(0., 1., 10), uniform_axis(0., 2., 10)}, 20.);
  const Point<2> x(0.31, 0.47);
  const Field gradient = [&](double p, double q) { return U.U1(p, q); };
  FV::InvariantFields<2, double> u;
  FV::InvariantFieldGradients<2, double> du;
  FV::InvariantFieldHessians<2> d2u;
  differentiate(gradient, x, u, du, d2u);

  const auto st = map.state<double>(x, u, du);
  const auto U2 = U.U2(x[0], x[1]);
  CHECK(st.rho[0] == x[0]);
  CHECK(st.rho[1] == x[1]);
  for (int a = 0; a < 2; ++a) {
    CHECK(st.U1[a] == Catch::Approx(U.U1(x[0], x[1])[a]).epsilon(1e-12));
    for (int b = 0; b < 2; ++b)
      CHECK(st.U2[a][b] == Catch::Approx(U2[a][b]).epsilon(1e-7));
  }
  const auto U3 = map.third(x, u, du, d2u);
  const auto U3e = U.U3();
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 2; ++c)
        CHECK(U3[a][b][c] == Catch::Approx(U3e[a][b][c]).epsilon(1e-5));

  // the curl damping vanishes on a gradient ...
  FV::InvariantFieldGradients<2, double> D{};
  map.damping_flux<double>(D, x, u, du);
  CHECK(std::abs(D[0][1]) < 1e-8);
  CHECK(std::abs(D[1][0]) < 1e-8);
  CHECK(D[0][0] == 0.);
  CHECK(D[1][1] == 0.);
  // ... and acts on a curl, with nu_a = 20 h_a^2
  auto dc = du;
  dc[0][1] += 1.;
  D = {};
  map.damping_flux<double>(D, x, u, dc);
  CHECK(D[0][1] == Catch::Approx(-20. * 0.2 * 0.2));
  CHECK(D[1][0] == Catch::Approx(20. * 0.1 * 0.1));

  CHECK(map.flux_weight(x, 0) == 1.);
  const auto init = map.initial_fields(x, [](double rho) { return 2. * rho; });
  CHECK(init[0] == Catch::Approx(2. * (x[0] + x[1])));
  CHECK(init[1] == Catch::Approx(2. * (x[0] + x[1])));
}

TEST_CASE("PolarInvariants: chain rule in both forms", "[FV][coordinates]")
{
  const Cubic U;
  // v = d_r U = (1 - s) U_1 + s U_2, w = U_2 - U_1, W = r w at rho = (r (1 - s), r s)
  const Field fields_w = [&](double r, double s) {
    const auto g = U.U1(r * (1. - s), r * s);
    return std::array<double, 2>{{(1. - s) * g[0] + s * g[1], g[1] - g[0]}};
  };
  const Field fields_W = [&](double r, double s) {
    const auto g = U.U1(r * (1. - s), r * s);
    return std::array<double, 2>{{(1. - s) * g[0] + s * g[1], r * (g[1] - g[0])}};
  };

  for (const auto form : {FV::PolarForm::w, FV::PolarForm::W}) {
    const auto map = polar_map(form);
    for (const double s : {0., 0.1, 0.5, 0.9, 1.}) {
      CAPTURE(int(form), s);
      const Point<2> x(0.3, s);
      FV::InvariantFields<2, double> u;
      FV::InvariantFieldGradients<2, double> du;
      FV::InvariantFieldHessians<2> d2u;
      differentiate(form == FV::PolarForm::w ? fields_w : fields_W, x, u, du, d2u);

      const double p = x[0] * (1. - s), q = x[0] * s;
      const auto st = map.state<double>(x, u, du);
      CHECK(st.rho[0] == Catch::Approx(p));
      CHECK(st.rho[1] == Catch::Approx(q));
      const auto U1 = U.U1(p, q);
      const auto U2 = U.U2(p, q);
      for (int a = 0; a < 2; ++a) {
        CHECK(st.U1[a] == Catch::Approx(U1[a]).epsilon(1e-12));
        for (int b = 0; b < 2; ++b)
          CHECK(st.U2[a][b] == Catch::Approx(U2[a][b]).epsilon(1e-6));
      }
      const auto U3 = map.third(x, u, du, d2u);
      const auto U3e = U.U3();
      for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
          for (int c = 0; c < 2; ++c)
            CHECK(U3[a][b][c] == Catch::Approx(U3e[a][b][c]).epsilon(1e-4).margin(1e-5));

      CHECK(std::abs(map.compatibility<double>(x, u, du)) < 1e-7);
      CHECK(map.flux_weight(x, 0) == 1.);
      CHECK(map.flux_weight(x, 1) == Catch::Approx(form == FV::PolarForm::w ? 1. / 0.3 : 1.));
    }
  }
}

TEST_CASE("PolarInvariants: rho U_ab stays finite at the origin", "[FV][coordinates]")
{
  const Cubic U;
  const auto map = polar_map(FV::PolarForm::w);
  // inside the first r cell: 1 / r is taken at r_reg = h_r / 2
  const Point<2> x(0., 0.25);
  const auto g = U.U1(0., 0.);
  const FV::InvariantFields<2, double> u{{(1. - x[1]) * g[0] + x[1] * g[1], g[1] - g[0]}};
  const FV::InvariantFieldGradients<2, double> du{};
  const auto st = map.state<double>(x, u, du);
  CHECK(map.r_reg() == Catch::Approx(5e-4));
  CHECK(st.rho[0] == 0.);
  CHECK(st.rho[1] == 0.);
  CHECK(st.U1[0] == Catch::Approx(g[0]));
  CHECK(st.U1[1] == Catch::Approx(g[1]));
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      CHECK(std::isfinite(st.U2[a][b]));
}

TEST_CASE("PolarInvariants: compatibility damping", "[FV][coordinates]")
{
  for (const auto form : {FV::PolarForm::w, FV::PolarForm::W}) {
    CAPTURE(int(form));
    FV::PolarInvariants::Parameters p;
    p.form = form;
    p.damping = 10.;
    p.origin_boost = 3.;
    p.origin_cells = 4.;
    const FV::PolarInvariants map(uniform_axis(0., 1., 100), uniform_axis(0., 1., 32), p);
    const FV::InvariantFields<2, double> u{{1., 0.}};
    FV::InvariantFieldGradients<2, double> du{};
    du[0][1] = 0.5; // c = v_s = 0.5 for w = W = 0 and w_r = W_r = 0

    // interior, outside the boost zone
    {
      const Point<2> x(0.5, 0.5);
      FV::InvariantFieldGradients<2, double> D{};
      map.damping_flux<double>(D, x, u, du);
      CHECK(D[0][1] == Catch::Approx(-10. * 0.5 / (32. * 32.)));
      CHECK(D[1][0] == Catch::Approx(form == FV::PolarForm::W ? 10. * 0.5 * 0.01 * 0.01 : 0.));
    }
    // boost at r = 2 h_r: rate = 10 (1 + 3 (1 - 2/4)^2)
    {
      const Point<2> x(0.02, 0.5);
      const auto nu = map.damping_coefficients(x);
      CHECK(nu[1] == Catch::Approx(10. * (1. + 3. * 0.25) / (32. * 32.)));
    }
    // no damping flux through s = 0, 1, nor (W form) r = 0, r_max
    CHECK(map.damping_coefficients(Point<2>(0.5, 0.))[1] == 0.);
    CHECK(map.damping_coefficients(Point<2>(0.5, 1.))[1] == 0.);
    CHECK(map.damping_coefficients(Point<2>(0., 0.5))[0] == 0.);
    CHECK(map.damping_coefficients(Point<2>(1., 0.5))[0] == 0.);
  }
  // off
  const auto off = polar_map(FV::PolarForm::w, 0.);
  CHECK(off.damping_coefficients(Point<2>(0.5, 0.5))[1] == 0.);
}

TEST_CASE("PolarOriginFit recovers the regular expansion", "[FV][coordinates]")
{
  FV::PolarOriginFit::Coefficients exact;
  exact.a = 0.7;
  exact.d = -0.2;
  exact.c0 = 1.3;
  exact.c1 = -0.45;
  exact.c2 = 0.8;
  std::vector<FV::PolarOriginFit::Cell> cells;
  for (int i = 0; i < 20; ++i)
    for (int j = 0; j < 32; ++j) {
      const double r = (i + 0.5) * 1e-3, s = (j + 0.5) / 32.;
      cells.push_back({r, s, FV::PolarOriginFit::v(exact, r, s), FV::PolarOriginFit::W(exact, r, s)});
    }
  const auto f = FV::PolarOriginFit::fit(cells, 6);
  REQUIRE(f.valid);
  CHECK(f.cells == 6 * 32);
  CHECK(f.a == Catch::Approx(exact.a).epsilon(1e-10));
  CHECK(f.d == Catch::Approx(exact.d).epsilon(1e-10));
  CHECK(f.c0 == Catch::Approx(exact.c0).epsilon(1e-8));
  CHECK(f.c1 == Catch::Approx(exact.c1).epsilon(1e-8));
  CHECK(f.c2 == Catch::Approx(exact.c2).epsilon(1e-8));
  CHECK(f.rms < 1e-10);
  const auto k2 = FV::PolarOriginFit::curvatures(f);
  CHECK(k2[0] == Catch::Approx(2. * 1.3));
  CHECK(k2[1] == Catch::Approx(2. * 1.3 - 0.45));
  CHECK(k2[2] == Catch::Approx(2. * (1.3 - 0.45 + 0.8)));

  // too few rows: invalid
  CHECK_FALSE(FV::PolarOriginFit::fit(cells, 21).valid);

  CHECK(FV::PolarOriginFit::blend_weight(0., 1.) == 1.);
  CHECK(FV::PolarOriginFit::blend_weight(0.5, 1.) == Catch::Approx(0.5));
  CHECK(FV::PolarOriginFit::blend_weight(1., 1.) == 0.);
  CHECK(FV::PolarOriginFit::blend_weight(0., 0.) == 0.);
}

TEST_CASE("PolarInvariants: origin fit from a solution sample and curvature blend", "[FV][coordinates]")
{
  FV::PolarOriginFit::Coefficients exact;
  exact.a = 0.7;
  exact.d = -0.2;
  exact.c0 = 1.3;
  exact.c1 = -0.45;
  exact.c2 = 0.8;

  FV::PolarInvariants::Parameters p;
  p.form = FV::PolarForm::w;
  p.origin_fit_r = 4e-3;
  p.origin_fit_rows = 6;
  p.origin_fit_t_min = 1.;
  const FV::PolarInvariants map(uniform_axis(0., 0.02, 20), uniform_axis(0., 1., 32), p);

  using Entry = SolutionSampleEntry<2, double>;
  std::vector<Entry> entries;
  for (int i = 0; i < 20; ++i)
    for (int j = 0; j < 32; ++j) {
      const double r = (i + 0.5) * 1e-3, s = (j + 0.5) / 32.;
      // components: an unrelated one, then w, then v
      entries.push_back({Point<2>(r, s),
                         1e-3,
                         {42., FV::PolarOriginFit::W(exact, r, s) / r, FV::PolarOriginFit::v(exact, r, s)},
                         std::vector<Tensor<1, 2, double>>(3)});
    }
  const SolutionSample<2, double> sample(entries, 3);

  int callbacks = 0;
  FV::PolarInvariants map_cb = map;
  map_cb.set_post_fit([&](FV::PolarOriginFit::Coefficients &f, const std::vector<FV::PolarOriginFit::Cell> &cells) {
    ++callbacks;
    CHECK(cells.size() == 20 * 32);
    f.c0 += 1.;
  });

  // before t_min the fit is computed but not used
  map.update(0.5, sample, {{2u, 1u}});
  REQUIRE(map.origin_fit().valid);
  CHECK(map.origin_fit().c0 == Catch::Approx(exact.c0).epsilon(1e-8));
  const FV::InvariantFields<2, double> u{{0.7, -0.2}};
  const FV::InvariantFieldGradients<2, double> du{};
  const Point<2> origin(0., 0.3);
  const auto unblended = map.state<double>(origin, u, du);
  CHECK(unblended.U2[0][0] != Catch::Approx(2. * exact.c0));

  // from t_min on, at r = 0: the fitted curvatures, U_a stay local
  map.update(1., sample, {{2u, 1u}});
  const auto st = map.state<double>(origin, u, du);
  const auto k2 = FV::PolarOriginFit::curvatures(exact);
  CHECK(st.U2[0][0] == Catch::Approx(k2[0]).epsilon(1e-8));
  CHECK(st.U2[0][1] == Catch::Approx(k2[1]).epsilon(1e-8));
  CHECK(st.U2[1][0] == Catch::Approx(k2[1]).epsilon(1e-8));
  CHECK(st.U2[1][1] == Catch::Approx(k2[2]).epsilon(1e-8));
  CHECK(st.U1[0] == Catch::Approx(0.7 - 0.3 * -0.2));
  // switched off on request, and beyond r_c
  CHECK(map.state<double>(origin, u, du, false).U2[0][0] == Catch::Approx(unblended.U2[0][0]));
  CHECK(map.state<double>(Point<2>(5e-3, 0.3), u, du).U2[0][0] ==
        Catch::Approx(map.state<double>(Point<2>(5e-3, 0.3), u, du, false).U2[0][0]));

  map_cb.update(1., sample, {{2u, 1u}});
  CHECK(callbacks == 1);

  // the cell-list overload gives the same fit
  std::vector<FV::PolarOriginFit::Cell> cells;
  for (const auto &e : sample)
    cells.push_back({e.point[0], e.point[1], e.values[2], e.point[0] * e.values[1]});
  const FV::PolarInvariants map_cells(uniform_axis(0., 0.02, 20), uniform_axis(0., 1., 32), p);
  map_cells.update(1., cells);
  CHECK(map_cells.origin_fit().c0 == map.origin_fit().c0);
  CHECK(map_cells.origin_fit().d == map.origin_fit().d);
  CHECK(map_cb.origin_fit().c0 == Catch::Approx(exact.c0 + 1.).epsilon(1e-8));
}

TEST_CASE("PolarInvariants: parameters and validation", "[FV][coordinates]")
{
  auto json = ConfigTree(
      json::value({{"discretization",
                    {{"grid", {{"x_grid", "0:0.01:1"}, {"y_grid", "0:0.03125:1"}, {"refine", 0}}},
                     {"invariant_coordinates", {{"polar_form", "W"}, {"damping", 7.}, {"origin_fit_r", 1e-3}}}}}}));
  const FV::PolarInvariants map(json);
  CHECK(map.parameters().form == FV::PolarForm::W);
  CHECK(map.parameters().damping == 7.);
  CHECK(map.parameters().origin_fit_r == 1e-3);
  CHECK(map.parameters().origin_boost == 100.);
  CHECK(map.r_reg() == Catch::Approx(0.005));
  CHECK(map.axis(1).width(0.5) == Catch::Approx(0.03125));

  CHECK_THROWS(FV::PolarInvariants(uniform_axis(-0.1, 1., 10), uniform_axis(0., 1., 8), {}));
  CHECK_THROWS(FV::PolarInvariants(uniform_axis(0., 1., 10), uniform_axis(0., 1.5, 8), {}));
  json.set_string("/discretization/invariant_coordinates/polar_form", "x");
  CHECK_THROWS(FV::PolarInvariants(json));
}
