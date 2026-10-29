#pragma once

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/fixed_string.hh>
#include <DiFfRG/common/tuples.hh>
#include <DiFfRG/discretization/FV/coordinates/invariant_coordinates.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/model/batch.hh>
#include <DiFfRG/model/model.hh>

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <tbb/tbb.h>

// standard library
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

// clang-format off
/**
 * @file gradient_potential.hh
 * @brief def::GradientPotential: the Kurganov-Tadmor (KT) model of a potential of several invariants, in gradient form
 * on a coordinate map (discretization/FV/coordinates/invariant_coordinates.hh).
 *
 * A model of \f$\partial_t U = -\mathcal F(\rho, U_a, U_{ab})\f$ on \f$N\f$ invariants derives from
 * def::GradientPotential instead of def::AbstractModel, names the \f$N\f$ FE functions that hold the gradient fields
 * of the map, and writes \f$\mathcal F\f$ once, at a point given in invariants:
 * @code
 * using Map = FV::PolarInvariants;
 * class MyModel : public def::GradientPotential<MyModel, Components, Map, "v", "w">, public def::fRG,
 *                 public def::FVDefaultBoundaries<MyModel>, public def::AD<MyModel>
 * {
 * public:
 *   MyModel(const ConfigTree &json) : GradientPotential(json), def::fRG(json) {}
 *
 *   template <typename Vector> void initial_condition(const Point<2> &x, Vector &values) const
 *   {
 *     gradient_initial_condition(x, values, [](double rho) { return m2 + lambda * rho; }); // U' of U_Lambda(rho)
 *   }
 *
 *   // dt U = -F at the invariants state.rho, with U_a = state.U1[a], U_ab = state.U2[a][b]
 *   template <typename NT, typename Solution>
 *   NT potential_flux(const FV::InvariantState<2, NT> &state, const Point<2> &x, const Solution &sol) const;
 * };
 * @endcode
 *
 * ### What it provides
 * | Hook | |
 * |------|-|
 * | flux | \f$F_i = w_i(x)\,\mathcal F(U_a, 0)\,e_i\f$ for gradient field \f$i\f$, \f$w_i\f$ = `flux_weight` of the map; 0 for all other FE functions |
 * | diffusion_flux | \f$w_i\,[\mathcal F(U_a, U_{ab}) - \mathcal F(U_a, 0)]\,e_i\f$ (only if `potential_curvature_active()`), plus the damping flux of the map |
 * | evaluate_batch | as flux and diffusion_flux, through the model's `potential_flux_batch` if it has one, else per point |
 * | wave_speed_blocks | one block per gradient field, all other FE functions in one further block |
 * | abs_tolerances | \f$\max(\mathrm{atol}, \epsilon\cdot\f$`tolerance_scale`\f$)\f$ per gradient field, \f$\epsilon\f$ = `field_rel_tol` (0: uniform atol) |
 * | needs_restart | only for maps with `update` (PolarInvariants): refreshes the map's numerical state, never restarts |
 * | raw_potential_gradient | \f$\partial_{x^i} U\f$ in the grid coordinates (`potential_gradient` of the map), for the reconstructed potential |
 * | EoM | the same: the EoM point is the minimum of \f$U\f$. A model with explicit symmetry breaking shadows it |
 *
 * The split puts the curvature-free part of \f$\mathcal F\f$ into the hyperbolic flux, with its wave speed, and the
 * rest into the diffusion flux; with `potential_curvature_active()` false (e.g. far in the UV, where the curvature
 * part is negligible but its gradients noisy) the curvature part is dropped. Separate wave-speed blocks dissipate
 * each field only across faces normal to its own axis: along axis \f$i\f$ the flux jacobian is nonzero only in row
 * \f$i\f$, and a shared block dissipated the other fields with \f$\partial\mathcal F/\partial u_i\f$, a spurious source
 * along the axes.
 *
 * ### Optional model hooks
 * | Hook | Default | |
 * |------|---------|-|
 * | `bool potential_curvature_active() const` | true | whether diffusion_flux carries the curvature part |
 * | `template <typename NT, typename Batch> void potential_flux_batch(std::vector<NT> &F, const std::vector<FV::InvariantState<dim, NT>> &states, const Batch &batch) const` | per point | \f$\mathcal F\f$ at all states at once, e.g. one `map_points()` call; `states[i]` belongs to `batch.x(i)` |
 * | `template <typename NT, typename Solution> NT potential_curvature_flux(const FV::InvariantState<dim, NT> &state, const Point<dim> &x, const Solution &sol) const` | difference of two potential_flux | the curvature part \f$\mathcal F(U_a, U_{ab}) - \mathcal F(U_a, 0)\f$ |
 * | `template <typename NT, typename Batch> void potential_curvature_flux_batch(std::vector<NT> &D, const std::vector<FV::InvariantState<dim, NT>> &states, const Batch &batch) const` | difference of two potential_flux_batch | the curvature part at all states at once |
 *
 * The curvature hooks let a model take the difference inside its integrand (less cancellation) or treat points
 * where only the full flux is singular separately.
 *
 * A model with further FE functions, its own wave_speed_blocks or its own needs_restart shadows the hook and calls
 * the one of GradientPotential (`GradientPotential::flux(F, x, sol)`, `update_coordinates(t, sample)`, ...) from it.
 * The batched path of evaluate_batch writes only the gradient fields' flux and diffusion flux (and calls source()
 * per point); a model with fluxes of further FE functions and a `potential_flux_batch` shadows evaluate_batch too.
 *
 * ### Parameters
 * The ConfigTree constructor builds the map from the json (see invariant_coordinates.hh) and reads
 * `/discretization/invariant_coordinates/field_rel_tol` (default 0).
 */
// clang-format on

namespace DiFfRG
{
  namespace FV
  {
    /// Whether the map @p Map keeps numerical state that is refreshed from the solution once per time step.
    template <typename Map, int dim, typename NT>
    concept UpdatableInvariantCoordinates =
        requires(const Map &map, const SolutionSample<dim, NT> &sample,
                 const std::array<unsigned int, Map::dim> &fields) { map.update(0., sample, fields); };
  } // namespace FV

  namespace def
  {
    /**
     * @brief Base class of a KT model of a potential of several invariants in gradient form; see gradient_potential.hh.
     *
     * @tparam Model the derived model class (CRTP).
     * @tparam Components_ a DiFfRG::ComponentDescriptor.
     * @tparam Map a coordinate map, FV::InvariantCoordinateMap.
     * @tparam fields the FE functions holding the gradient fields of the map, in axis order.
     */
    template <typename Model, typename Components_, typename Map, FixedString... fields>
    class GradientPotential : public AbstractModel<Model, Components_>
    {
      static_assert(FV::InvariantCoordinateMap<Map>, "GradientPotential: Map must be an FV::InvariantCoordinateMap.");
      static_assert(sizeof...(fields) == Map::dim,
                    "GradientPotential: name exactly one FE function per invariant (axis) of the map.");

      using Base = AbstractModel<Model, Components_>;
      const Model &asImp() const { return static_cast<const Model &>(*this); }

    public:
      using Components = Components_;
      using Coordinates = Map;
      /// Number of invariants, the dimension of the mesh.
      static constexpr int n_invariants = Map::dim;
      static constexpr size_t n_fe_functions = Components::count_fe_functions(0);

      /// The FE function indices of the gradient fields, in axis order.
      static constexpr std::array<unsigned int, n_invariants> field_indices()
      {
        return {{static_cast<unsigned int>(typename Components::FEFunction_Descriptor{}(fields))...}};
      }

      /// From a configured map and the relative tolerance of the gradient fields (0: uniform tolerance).
      explicit GradientPotential(const Map &coordinates, const double field_rel_tol = 0.)
          : m_coordinates(coordinates), m_field_rel_tol(field_rel_tol)
      {
      }

      /// The map from the json, see invariant_coordinates.hh.
      explicit GradientPotential(const ConfigTree &json)
          : m_coordinates(json),
            m_field_rel_tol(json.get_double("/discretization/invariant_coordinates/field_rel_tol", 0.))
      {
      }

      /// The coordinate map.
      const Map &coordinates() const { return m_coordinates; }
      /// The coordinate map, writable (e.g. to set a PolarInvariants post-fit callback).
      Map &coordinates() { return m_coordinates; }

      /// Default: the curvature part of the flow is always in diffusion_flux.
      bool potential_curvature_active() const { return true; }

      /// The fields of the map at @p x from the FE functions @p u (anything indexable by FE function).
      template <typename NT, typename Values>
      static FV::InvariantFields<n_invariants, NT> gradient_fields(const Values &u)
      {
        FV::InvariantFields<n_invariants, NT> f;
        constexpr auto idx = field_indices();
        for (int i = 0; i < n_invariants; ++i)
          f[i] = NT(u[idx[i]]);
        return f;
      }

      /// The gradients of the fields of the map from the FE function gradients @p du.
      template <typename NT, typename Gradients>
      static FV::InvariantFieldGradients<n_invariants, NT> gradient_field_gradients(const Gradients &du)
      {
        FV::InvariantFieldGradients<n_invariants, NT> g;
        constexpr auto idx = field_indices();
        for (int i = 0; i < n_invariants; ++i)
          for (int d = 0; d < n_invariants; ++d)
            g[i][d] = NT(du[idx[i]][d]);
        return g;
      }

      /**
       * @brief \f$\rho_a, U_a, U_{ab}\f$ at @p x from the FE functions and their gradients, e.g. in extract() or the
       * readouts. The origin blend of PolarInvariants is off unless @p origin_blend.
       */
      template <typename NT, typename Values, typename Gradients>
      FV::InvariantState<n_invariants, NT> potential_state(const Point<n_invariants> &x, const Values &u,
                                                           const Gradients &du, const bool origin_blend = false) const
      {
        const auto f = gradient_fields<NT>(u);
        const auto g = gradient_field_gradients<NT>(du);
        if constexpr (requires(const Map &m) { m.template state<NT>(x, f, g, true); })
          return m_coordinates.template state<NT>(x, f, g, origin_blend);
        else
          return m_coordinates.template state<NT>(x, f, g);
      }

      /// \f$U_{abc}\f$ at @p x from the FE functions, their gradients and hessians (e.g. in extract()).
      template <typename Values, typename Gradients, typename Hessians>
      FV::InvariantThirdDerivatives<n_invariants> potential_third(const Point<n_invariants> &x, const Values &u,
                                                                  const Gradients &du, const Hessians &d2u) const
      {
        FV::InvariantFieldHessians<n_invariants> h;
        constexpr auto idx = field_indices();
        for (int i = 0; i < n_invariants; ++i)
          for (int a = 0; a < n_invariants; ++a)
            for (int b = 0; b < n_invariants; ++b)
              h[i][a][b] = double(d2u[idx[i]][a][b]);
        return m_coordinates.third(x, gradient_fields<double>(u), gradient_field_gradients<double>(du), h);
      }

      /// Writes the fields of an \f$O(N)\f$-symmetric start with \f$U' = f(\sum_a\rho_a)\f$ into @p values.
      template <typename Vector, typename F>
      void gradient_initial_condition(const Point<n_invariants> &x, Vector &values, const F &f) const
      {
        const auto u = m_coordinates.initial_fields(x, f);
        constexpr auto idx = field_indices();
        for (int i = 0; i < n_invariants; ++i)
          values[idx[i]] = u[i];
      }

      /// The flux, see gradient_potential.hh.
      template <int dim, typename NT, typename Solutions, size_t n_fe>
      void flux(std::array<Tensor<1, dim, NT>, n_fe> &F, const Point<dim> &x, const Solutions &sol) const
      {
        static_assert(dim == n_invariants, "GradientPotential: the mesh dimension must be the number of invariants.");
        for (auto &F_c : F)
          F_c = Tensor<1, dim, NT>();
        const auto f = gradient_fields<NT>(get<"fe_functions">(sol));
        const NT J = asImp().potential_flux(m_coordinates.template values<NT>(x, f), x, sol);
        constexpr auto idx = field_indices();
        for (int i = 0; i < n_invariants; ++i)
          F[idx[i]][i] = m_coordinates.flux_weight(x, i) * J;
      }

      /// The diffusion flux, see gradient_potential.hh.
      template <int dim, typename NT, typename Solutions, size_t n_fe>
      void diffusion_flux(std::array<Tensor<1, dim, NT>, n_fe> &F, const Point<dim> &x, const Solutions &sol) const
      {
        static_assert(dim == n_invariants, "GradientPotential: the mesh dimension must be the number of invariants.");
        for (auto &F_c : F)
          F_c = Tensor<1, dim, NT>();
        const auto f = gradient_fields<NT>(get<"fe_functions">(sol));
        const auto g = gradient_field_gradients<NT>(get<"fe_derivatives">(sol));
        FV::InvariantFieldGradients<n_invariants, NT> D{};
        if (asImp().potential_curvature_active()) {
          const auto state = m_coordinates.template state<NT>(x, f, g);
          NT J;
          if constexpr (requires(const Model &m) { m.potential_curvature_flux(state, x, sol); })
            J = asImp().potential_curvature_flux(state, x, sol);
          else
            J = asImp().potential_flux(state, x, sol) -
                asImp().potential_flux(m_coordinates.template values<NT>(x, f), x, sol);
          for (int i = 0; i < n_invariants; ++i)
            D[i][i] = m_coordinates.flux_weight(x, i) * J;
        }
        m_coordinates.template damping_flux<NT>(D, x, f, g);
        constexpr auto idx = field_indices();
        for (int i = 0; i < n_invariants; ++i)
          F[idx[i]] = D[i];
      }

      /**
       * @brief flux and diffusion_flux at all points of a batch. With a model `potential_flux_batch`, \f$\mathcal F\f$
       * is evaluated once for all points (twice for the diffusion flux); without, the per-point hooks are called.
       */
      template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
      {
        using NT = typename Batch::number_type;
        constexpr bool batched =
            requires(const Model &m, std::vector<NT> &F, const std::vector<FV::InvariantState<n_invariants, NT>> &s,
                     const Batch &b) { m.potential_flux_batch(F, s, b); };
        if constexpr (!batched)
          Base::evaluate_batch(out, batch);
        else {
          constexpr Term available = DiFfRG::internal::batch_terms<Batch>();
          constexpr auto idx = field_indices();
          const size_t n = batch.size();
          const auto fields_at = [&](const size_t i) {
            FV::InvariantFields<n_invariants, NT> f;
            for (int c = 0; c < n_invariants; ++c)
              f[c] = batch.values(idx[c])[i];
            return f;
          };

          if constexpr (DiFfRG::internal::contains(available, Term::source))
            if (out.requested(Term::source))
              DiFfRG::internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
                std::array<NT, Out::n_components> S{};
                asImp().source(S, x, sol);
                out.store_source(i, S);
              });

          if (out.requested(Term::flux)) {
            std::vector<FV::InvariantState<n_invariants, NT>> states(n);
            tbb::parallel_for(size_t(0), n, [&](const size_t i) {
              states[i] = m_coordinates.template values<NT>(batch.x(i), fields_at(i));
            });
            std::vector<NT> J(n, NT(0.));
            asImp().potential_flux_batch(J, states, batch);
            for (int c = 0; c < n_invariants; ++c) {
              auto col = out.flux(idx[c], c);
              tbb::parallel_for(size_t(0), n,
                                [&](const size_t i) { col[i] = m_coordinates.flux_weight(batch.x(i), c) * J[i]; });
            }
          }

          if constexpr (DiFfRG::internal::contains(available, Term::diffusion_flux))
            if (out.requested(Term::diffusion_flux)) {
              const auto gradients_at = [&](const size_t i) {
                FV::InvariantFieldGradients<n_invariants, NT> g;
                for (int c = 0; c < n_invariants; ++c)
                  for (int d = 0; d < n_invariants; ++d)
                    g[c][d] = batch.derivatives(idx[c], d)[i];
                return g;
              };
              std::vector<NT> J(n, NT(0.));
              if (asImp().potential_curvature_active()) {
                std::vector<FV::InvariantState<n_invariants, NT>> states(n);
                tbb::parallel_for(size_t(0), n, [&](const size_t i) {
                  states[i] = m_coordinates.template state<NT>(batch.x(i), fields_at(i), gradients_at(i));
                });
                if constexpr (requires(const Model &m) { m.potential_curvature_flux_batch(J, states, batch); })
                  asImp().potential_curvature_flux_batch(J, states, batch);
                else {
                  auto states0 = states;
                  for (auto &st : states0)
                    for (auto &row : st.U2)
                      row.fill(NT(0.));
                  std::vector<NT> J0(n, NT(0.));
                  asImp().potential_flux_batch(J, states, batch);
                  asImp().potential_flux_batch(J0, states0, batch);
                  for (size_t i = 0; i < n; ++i)
                    J[i] = J[i] - J0[i];
                }
              }
              tbb::parallel_for(size_t(0), n, [&](const size_t i) {
                const auto x = batch.x(i);
                FV::InvariantFieldGradients<n_invariants, NT> D{};
                for (int c = 0; c < n_invariants; ++c)
                  D[c][c] = m_coordinates.flux_weight(x, c) * J[i];
                m_coordinates.template damping_flux<NT>(D, x, fields_at(i), gradients_at(i));
                for (int c = 0; c < n_invariants; ++c)
                  for (int d = 0; d < n_invariants; ++d)
                    out.diffusion_flux(idx[c], d)[i] = D[c][d];
              });
            }
        }
      }

      /// \f$\partial_{x^i} U\f$ in the grid coordinates, from which DiFfRG reconstructs the scalar potential.
      template <int dim, typename Vector>
      std::array<double, dim> raw_potential_gradient(const Point<dim> &x, const Vector &u) const
      {
        static_assert(dim == n_invariants, "GradientPotential: the mesh dimension must be the number of invariants.");
        return m_coordinates.potential_gradient(x, gradient_fields<double>(u));
      }

      /// The EoM point is the minimum of \f$U\f$: \f$\partial_{x^i} U = 0\f$.
      template <int dim, typename Vector> std::array<double, dim> EoM(const Point<dim> &x, const Vector &u) const
      {
        return raw_potential_gradient(x, u);
      }

      /// One wave-speed block per gradient field; all other FE functions share one further block.
      template <size_t n> void wave_speed_blocks(std::array<int, n> &blocks) const
      {
        blocks.fill(n_invariants);
        constexpr auto idx = field_indices();
        for (int i = 0; i < n_invariants; ++i)
          blocks[idx[i]] = i;
      }

      /// Per-dof absolute tolerances of the gradient fields, see gradient_potential.hh.
      template <int dim, typename Solution, size_t n>
      void abs_tolerances(std::array<double, n> &atol, const Point<dim> &x, const Solution &sol, const double abs_tol,
                          const double /*rel_tol*/) const
      {
        if (!(m_field_rel_tol > 0.)) return;
        const auto f = gradient_fields<double>(get<"fe_functions">(sol));
        constexpr auto idx = field_indices();
        for (int i = 0; i < n_invariants; ++i)
          atol[idx[i]] = std::max(abs_tol, m_field_rel_tol * m_coordinates.tolerance_scale(x, f, i));
      }

      /**
       * @brief Refresh the numerical state of the map (PolarInvariants: the origin fit) from the accepted solution
       * @p sample at time @p t. Call this from the model's own needs_restart if it has one.
       */
      template <int dim, typename NT>
      void update_coordinates(const double t, const SolutionSample<dim, NT> &sample) const
      {
        if constexpr (FV::UpdatableInvariantCoordinates<Map, dim, NT>) m_coordinates.update(t, sample, field_indices());
      }

      /// Only for maps with state (FV::UpdatableInvariantCoordinates): update_coordinates(), never a restart.
      template <int dim, typename NT>
      bool needs_restart(const double t, const Point<dim> & /*EoM*/, const SolutionSample<dim, NT> &sample) const
        requires FV::UpdatableInvariantCoordinates<Map, dim, NT>
      {
        update_coordinates(t, sample);
        return false;
      }

    protected:
      Map m_coordinates;
      double m_field_rel_tol;
    };
  } // namespace def
} // namespace DiFfRG
