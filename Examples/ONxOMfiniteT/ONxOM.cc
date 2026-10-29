#include "model.hh"

#include <string>

/**
 * The O(N1) x O(N2) model of model.hh. /ONxOM/map selects the grid of the potential:
 *   polar     -- U(rho1, rho2) on r = rho1 + rho2, s = rho2 / r (FV::PolarInvariants)
 *   cartesian -- U(rho1, rho2) on the invariants themselves (FV::CartesianInvariants<2>)
 *   1D        -- the O(N1 + N2) model in rho = rho1 + rho2, the reference for /physical/delta = 0
 *
 * With parameter.toml (O(2) x O(2), delta = 0) the polar run follows the 1D O(4) run: sigma to < 1 % in the broken
 * phase (up to the EoM point resolved on the 30-cell r grid near the transition at t ~ 1.9), m_pi^2 to 7e-6 relative
 * after the restoration, to t = 5 (about 3 min on 16 cores). -sd /physical/delta=0.02 selects the first multiplet
 * (the minimum moves to rho2 = 0, m_pi2^2 ~ 0.015 GeV^2) and also runs to t = 5.
 *
 * Limitations seen with these parameters:
 *   - the Cartesian map (with y_grid = x_grid) breaks the O(4) symmetry near the transition and stops at t ~ 1.9; this
 *     is the odd-even mode across the radial direction that the polar map removes (invariant_coordinates.hh).
 *   - where the potential is flat in the infrared (a broken phase down to small k, e.g. N1 = N2 = 1), the minimum of U
 *     is not unique and the default EoM point may sit on the plateau; the readouts there are not meaningful, the flow
 *     is. A model then chooses its readout point itself (extractor_point).
 *   - with a strong breaking (delta = 0.1) only N1 modes stay light, the broken phase persists, and the run stops at
 *     t ~ 2.16 where the non-convex interior approaches the regulator pole (also with the origin fit on). Flowing into
 *     the flat region needs a model-side treatment of that pole (e.g. a soft floor on the gap in the integrand).
 */
template <typename Model> int run(const ConfigTree &config)
{
  constexpr uint dim = Model::dim;
  using Discretization = FV::Discretization<typename Model::Components, RectangularMesh<dim>>;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  Model model(config);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(config)};
  OutputSession<Assembler> data_out(config);
  const auto log = data_out.report_port();
  Discretization discretization(mesh, config, log);
  Assembler assembler(discretization, model, config);
  NoAdaptivity mesh_adaptor(assembler);
  TimeStepper time_stepper(config, assembler, data_out, mesh_adaptor);

  FV::FlowingVariables<Discretization> initial_condition(discretization);
  initial_condition.interpolate(model);

  Timer timer;
  try {
    time_stepper.run(initial_condition, 0., config.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Simulation finished with exception {}", e.what());
    return -1;
  }
  log.info("Simulation finished after " + time_format(timer.wall_time()));
  return 0;
}

int main(int argc, char *argv[])
{
  const auto config_helper = DiFfRG::Init(argc, argv, "parameter.toml").get_configuration_helper();
  const auto config = config_helper.get_config();

  const std::string map = config.get_string("/ONxOM/map", "polar");
  if (map == "polar") return run<ONxOM::Model<FV::PolarInvariants>>(config);
  if (map == "cartesian") return run<ONxOM::Model<FV::CartesianInvariants<2>>>(config);
  if (map == "1D") return run<ONxOM::Model1D>(config);
  std::cerr << "/ONxOM/map must be polar, cartesian or 1D, not " << map << std::endl;
  return -1;
}
