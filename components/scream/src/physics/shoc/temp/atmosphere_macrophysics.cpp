#include "../../share/scream_assert.hpp"
#include "physics/shoc/scream_shoc_interface.hpp"
#include "physics/shoc/atmosphere_macrophysics.hpp"

namespace scream
{

// =========================================================================================
SHOCMacrophysics::SHOCMacrophysics (const Comm& comm,const ParameterList& /* params */)
  : m_shoc_comm (comm)
{
/* Anything that can be initialized without grid information can be initialized here.
 * Like universal constants, shoc options.
*/
}

// =========================================================================================
void SHOCMacrophysics::set_grids(const std::shared_ptr<const GridsManager> grids_manager)
{

  using namespace units;
  // The units of mixing ratio Q are technically non-dimensional.
  // Nevertheless, for output reasons, we like to see 'kg/kg'.
  auto Q = kg/kg;
  auto Qdp = Q * Pa;
  Q.set_string("kg/kg");
  Qdp.set_string("kg/kg Pa");

  constexpr int NVL = 72;  /* TODO THIS NEEDS TO BE CHANGED TO A CONFIGURABLE */
  constexpr int QSZ =  9;  /* TODO THIS NEEDS TO BE CHANGED TO A CONFIGURABLE */

  auto grid = grids_manager->get_grid("Physics");
  const int num_dofs = grid->num_dofs();
  const int nc = num_dofs;

  auto VL = FieldTag::VerticalLevel;
  auto CO = FieldTag::Column;
  auto VR = FieldTag::Variable;
  auto TL = FieldTag::TimeLevel;

  FieldLayout scalar3d_layout { {CO,VL}, {nc,NVL} }; // Note that C++ and Fortran read array dimensions in reverse
  FieldLayout vector3d_layout { {CO,VR,VL}, {nc,QSZ,NVL} };
  FieldLayout tracers_state_layout { {CO,TL,VR,VL}, {nc,2,4,NVL} };
  FieldLayout scalar_state_3d_mid_layout { {CO,TL,VL} , {nc,2,NVL}};
  FieldLayout q_forcing_layout  { {CO,VR,VL}, {nc,4,NVL} };

  // set requirements
  m_required_fields.emplace("dp"         , scalar_state_3d_mid_layout,      Pa, "Physics");
  m_required_fields.emplace("qdp"        , tracers_state_layout,           Qdp, grid->name());
  // set computed
  m_computed_fields.emplace("q"           , vector3d_layout,          Q, "Physics");
  m_computed_fields.emplace("FQ"          , q_forcing_layout,         Q, grid->name());

}
// =========================================================================================
void SHOCMacrophysics::initialize (const util::TimeStamp& t0)
{
  m_current_ts = t0;
  auto q_ptr = m_shoc_fields_out.at("q").get_view().data();

  shoc_c_init ();

}

// =========================================================================================
void SHOCMacrophysics::run (const Real dt)
{
  auto q_ptr = m_shoc_fields_out.at("q").get_view().data();
  auto FQ_ptr = m_shoc_fields_out.at("FQ").get_view().data();
  auto dp_ptr = m_shoc_fields_in.at("dp").get_view().data();
  auto qdp_ptr = m_shoc_fields_in.at("qdp").get_view().data();

  shoc_c_main (dt,q_ptr);//,FQ_ptr,qdp_ptr);

  m_current_ts += dt;
  m_shoc_fields_out.at("q").get_header().get_tracking().update_time_stamp(m_current_ts);
  m_shoc_fields_out.at("FQ").get_header().get_tracking().update_time_stamp(m_current_ts);
}
// =========================================================================================
void SHOCMacrophysics::finalize()
{
  shoc_c_finalize ();
}
// =========================================================================================

void SHOCMacrophysics::register_fields (FieldRepository<Real, device_type>& field_repo) const {
  for (auto& fid : m_required_fields) {
    field_repo.register_field(fid);
  }
  for (auto& fid : m_computed_fields) {
    field_repo.register_field(fid);
  }
}

void SHOCMacrophysics::set_required_field_impl (const Field<const Real, device_type>& f) {
  // Store a copy of the field. We need this in order to do some tracking checks
  // at the beginning of the run call. Other than that, there would be really
  // no need to store a scream field here; we could simply set the view ptr
  // in the Homme's view, and be done with it.
  m_shoc_fields_in.emplace(f.get_header().get_identifier().name(),f);

  // Add myself as customer to the field
  f.get_header_ptr()->get_tracking().add_customer(weak_from_this());
}

void SHOCMacrophysics::set_computed_field_impl (const Field<      Real, device_type>& f) {
  // Store a copy of the field. We need this in order to do some tracking updates
  // at the end of the run call. Other than that, there would be really
  // no need to store a scream field here; we could simply set the view ptr
  // in the Homme's view, and be done with it.
  m_shoc_fields_out.emplace(f.get_header().get_identifier().name(),f);

  // Add myself as provider for the field
  f.get_header_ptr()->get_tracking().add_provider(weak_from_this());
}

} // namespace scream
