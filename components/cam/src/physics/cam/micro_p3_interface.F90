module micro_p3_interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!
  !! Interface between E3SM and P3 microphysics
  !!
  !! Author: Peter Caldwell
  !!
  !! Last updated: 2018-09-12
  !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  use shr_kind_mod,   only: r8=>shr_kind_r8
  use ppgrid,         only: pcols,pver

!comment: I think Kai added handle_errmsg. It would be better to 
!use standard E3SM libraries if possible.
  use error_messages, only: handle_errmsg

  use physics_types,  only: physics_state, &
                            physics_ptend
       
  implicit none
  private

  !Define indices for state%q constituents at module level so
  !defining them in micro_p3_register makes them permanently 
  !available.

  integer, public ::    &
       ixcldliq = -1,   &
       ixice = -1,      &
       ixnumliq = -1,   &
       ixnumice = -1,   &
       ixrain   = -1,   &
       ixnumrain= -1,   &
       ixrim = -1,      &
       ixrimvol  = -1 

  !Define th (potential temperature) and (water vapor mixing 
  !ratio) qv at module level so "_old" values can easily be 
  !assigned at the beginning of each step from the value at the end 
  !of the step before.

  real(r8) :: th(pcols,pver)
  real(r8) :: qv(pcols,pver)

  contains
  !================================================================================================

  subroutine micro_p3_register

    ! Register Microphysics Constituents 
    ! (i.e. members of state%q) and save indices.
    !================
    call cnst_add(cnst_names(1), mwdry, cpair, 0._r8, ixcldliq, &
         longname='Grid box averaged cloud liquid amount', &
         is_convtran1=.true.)
    call cnst_add(cnst_names(2), mwdry, cpair, 0._r8, ixice, &
         longname='Grid box averaged cloud ice amount', &
         is_convtran1=.true.)
    call cnst_add(cnst_names(3), mwh2o, cpair, 0._r8, ixnumliq, &
         longname='Grid box averaged cloud liquid number', &
         is_convtran1=.true.)
    call cnst_add(cnst_names(4), mwh2o, cpair, 0._r8, ixnumice, &
         longname='Grid box averaged cloud ice number', &
         is_convtran1=.true.)
    call cnst_add(cnst_names(5), mwh2o, cpair, 0._r8, ixrain, &
         longname='Grid box averaged rain amount', &
         is_convtran1=.true.)
    call cnst_add(cnst_names(6), mwh2o, cpair, 0._r8, ixrim, &
         longname='Grid box averaged riming amount', &
         is_convtran1=.true.)
    call cnst_add(cnst_names(7), mwh2o, cpair, 0._r8, ixnumrain, &
         longname='Grid box averaged rain number', &
         is_convtran1=.true.)
    call cnst_add(cnst_names(8), mwh2o, cpair, 0._r8, ixrimvol, &
         longname='Grid box averaged riming volume', &
         is_convtran1=.true.)

    ! Add Variables to Pbuf
    !================
    !! module radiation_data & module cloud_rad_props
    call pbuf_add_field('DEI',        'physpkg',dtype_r8,(/pcols,pver/), dei_idx)
    call pbuf_add_field('MU',         'physpkg',dtype_r8,(/pcols,pver/), mu_idx)
    call pbuf_add_field('LAMBDAC',    'physpkg',dtype_r8,(/pcols,pver/), lambdac_idx)

  end subroutine micro_p3_register

  !================================================================================================

  subroutine micro_p3_interface_init()
    use phys_control,   only: phys_getopts
    use micro_p3,       only: micro_p3_init
    use cam_history,    only: addfld, add_default

    character(128) :: p3_lookup_dir

    ! PULL DIRECTORY OF LOOKUP TABLE FROM ATM NAMELIST
    ! =============
    call phys_getopts(p3_lookup_dir_out = p3_lookup_dir)

    ! CALL P3 INIT:
    !==============
    !might want to add all E3SM parameter vals to p3_init call...

    !TODO: add errstring and constants to init function
    call p3_init(lookup_file_dir,errstring)
    call handle_errmsg(errstring, subname="micro_p3_init")

    ! INITIALIZE OUTPUT
    !==============
    !TODO: put addfld and add_default calls here.

  end subroutine micro_p3_interface_init

  !================================================================================================

  subroutine micro_p3_interface_tend(state, ptend, dtime, pbuf)

    use cam_history,    only: outfld
    use time_manager,   only: is_first_step

    !INPUT/OUTPUT VARIABLES
    type(physics_state),         intent(in)    :: state
    type(physics_ptend),         intent(out)   :: ptend
    real(r8),                    intent(in)    :: dtime
    type(physics_buffer_desc),   pointer       :: pbuf(:)
    logical :: lq(pcnst)   !list of what constituents to update

    !INTERNAL VARIABLES
    real(r8) :: th_old(pcols,pver)     !potential temperature from last step   K
    real(r8) :: dzq(pcols,pver)        !geometric layer thickness              m
    real(r8) :: cldliq                 !cloud liquid water mixing ratio        kg/kg
    real(r8) :: numliq                 !cloud liquid water drop concentraiton  #/kg
    real(r8) :: rain                   !rain water mixing ratio                kg/kg
    real(r8) :: numrain                !rain water number concentration        #/kg
    real(r8) :: qv                     !water vapor mixing ratio               kg/kg
    real(r8) :: ice                    !total ice water mixing ratio           kg/kg
    real(r8) :: qirim                  !rime ice mixing ratio                  kg/kg
    real(r8) :: numice                 !total ice crystal number concentration #/kg
    real(r8) :: rimvol                 !rime volume mixing ratio               m3/kg
    integer :: it                      !timestep counter                       -

    logical :: log_predictNc           !prognostic droplet concentration or not?

    ! Associate Pbuf Variables
    !==============

    !liq effective radius (m)
    call pbuf_get_field(pbuf, rel_idx,         rel,         col_type=col_type)
    !ice effective radius (m)
    call pbuf_get_field(pbuf, rei_idx,         rei,         col_type=col_type)

    ! INITIALIZE PTEND
    !==============
    !ptend is an output variable. Since not substepping in micro, don't need 
    !a local copy.

    lq           = .false. !initialize all constituents to false.
    lq(1)        = .true.
    lq(ixcldliq) = .true.
    lq(ixice)    = .true.
    lq(ixnumliq) = .true.
    lq(ixnumice) = .true.
    lq(ixrain)   = .true.
    lq(ixrim)    = .true.
    lq(ixnumrain)= .true.
    lq(ixrimvol)  = .true.

    call physics_ptend_init(ptend, psetcols, "micro_p3", ls=.true., lq=lq)

    ! HANDLE AEROSOL ACTIVATION
    !==============
    !saving this for later... use prescribed Nd for now.
    log_predictNc = .false.

    ! GET "OLD" VALUES
    !==============
    ! TODO: answer question: is_first_step =1 for each submission, or just for type=initial first step?
    !          does p3 want _old values from last p3 call, or from some other point?
    if ( is_first_step() ) then
       th_old=state%t*state%exner !this is wrong, just using to get started.
       qv_old=state%q(:,:,1)
    else
       th_old = th !use th from end of last p3 step
       qv_old = qv !use qv from end of last p3 step
    end if

    ! CONVERT T TO POTENTIAL TEMPERATURE
    !==============
    ! Someday we may want to make P3 take in T rather than theta... it uses both.

    th=state%t*state%exner

    ! COMPUTE GEOMETRIC THICKNESS OF GRID
    !==============
    dzq = state%zi(:,1:pver) - state%zi(:,2:pver+1)

    ! ASSIGN TOP AND BOTTOM INDICES FOR GRID
    !==============
    !kts is closest level to top of model. Instead of 1 (top-of-model), 
    !we use the previously-defined trop_cloud_top_lev to reduce the number of 
    !levels we need to calculate and to avoid upper-atmos regions where this
    !micro-physics is inappropriate. kte is the near-surface level = pver.

    kts=top_lev
    kte=pver 

    ! DEAL WITH SSAT
    !==============
    !ssat (supersaturated mixing ratio measured in kg/kg) can be prognosed
    !or diagnosed in p3 depending on p3's hardcoded log_predictSsat parameter.
    !ssat is an intent inout variable, but when log_predictSsat is false 
    !(as we will set it initially), ssat is overwritten rather than used by p3.
    !Thus it shouldn't matter what ssat is, so we give it -999.

    ssat(:,:) = -999._r8

    ! HANDLE TIMESTEP COUNTER
    !==============
    !p3 wants to know the timestep number (which it calls "it") because it
    !handles things differently on the first step, where it doesn't have values
    !yet. E3SM has a handy function for deciding if this is the first step, so 
    !we hack "it" with "is_first_step()" for now. Eventually, we should replace
    !"it" with a logical.

    if (is_first_step()) then
       it=1
    else
       it=999 !integer
    end if

    ! MAKE LOCAL COPIES OF VARS MODIFIED BY P3
    !==============
    !local copies are needed because state is passed into this routine as intent=in
    !while P3 seeks to modify state variables in-place. Also, we need a copy of 
    !old values in order to back out ptend values later. Traditionally, a local copy 
    !is created by copying the whole state. It is much cheaper to just copy the 
    !variables we need. 

    cldliq = state_loc%q(:,:,ixcldliq)
    numliq = state_loc%q(:,:,ixnumliq)
    rain = state_loc%q(:,:,ixrain)
    numrain = state_loc%q(:,:,ixnumrain)
    qv = state_loc%q(:,:,1)
    ice = state_loc%q(:,:,ixice)
    qirim = state_loc%q(:,:,ixqirim)
    numice = state_loc%q(:,:,ixnumice)
    rimvol = state_loc%q(:,:,ixrimvol)

    ! CALL P3
    !==============
    call p3_main( &
         cldliq, & ! INOUT  cloud, mass mixing ratio         kg kg-1
         numliq, & ! INOUT  cloud, number mixing ratio       #  kg-1
         rain,   & ! INOUT  rain, mass mixing ratio          kg kg-1
         numrain,& ! INOUT  rain, number mixing ratio        #  kg-1
         th_old,                & ! INOUT  beginning of time step theta     K
         th,                    & ! INOUT  potential temperature            K
         qv_old,                & ! INOUT  beginning of time step qv        kg kg-1
         qv,        & ! INOUT  water vapor mixing ratio         kg kg-1
         dtime,                 & ! IN     model time step                  s
         ice,    & ! INOUT  ice, total mass mixing ratio     kg kg-1
         qirim,  & ! INOUT  ice, rime mass mixing ratio      kg kg-1
         numice, & ! INOUT  ice, total number mixing ratio   #  kg-1
         rimvol, & ! INOUT  ice, rime volume mixing ratio    m3 kg-1
         ssat,                  & ! INOUT  supersaturation (i.e., qv-qvs)   kg kg-1
         state%pmid(:,:),       & ! IN     pressure at cell midpoints       Pa
         dzq,                   & ! IN     vertical grid spacing            m
         it,                    & ! IN     time step counter NOTE: starts at 1 for first time step
         prt_liq,               & ! OUT    surface liquid precip rate       m s-1
         prt_sol,               & ! OUT    surface frozen precip rate       m s-1
         its,                   & ! IN     horizontal index lower bound     -
         ite,                   & ! IN     horizontal index upper bound     -
         kts,                   & ! IN     vertical index lower bound       -
         kte,                   & ! IN     vertical index upper bound       -
         diag_ze,               & ! OUT    equivalent reflectivity          dBZ  UNUSED?
         rel,                   & ! OUT    effective radius, cloud          m
         rei,                   & ! OUT    effective radius, ice            m
         diag_vmi,              & ! OUT    mass-weighted fall speed of ice  m s-1
         diag_di,               & ! OUT    mean diameter of ice             m
         diag_rhoi,             & ! OUT    bulk density of ice              kg m-1
         log_predictNc,         & ! IN     .true.=prognostic Nc, .false.=specified Nc
         )

    !MASSAGE OUTPUT TO FIT E3SM EXPECTATIONS
    !============= 
    precl=prt_liq+prt_sol

    !TODO: figure out what else other E3SM parameterizations need from micro and make sure 
    !they are assigned here. The comments below are a step in that direction.

    !cloud_rad_props needs ice effective diameter, which Kai calculates as below:
    !   dei = rei*diag_rhopo(i,k,iice)/rhows*2._r8
    !where rhopo is bulk ice density from table lookup (taken from f1pr16, but not 
    !done in my ver yet) and rhows=917.0 is a constant parameter.

    !cloud_rad_props also uses snow radiative properties which aren't available from 
    !P3 (perhaps because ice phase in p3 includes *all* ice already?).

    !BACK OUT TENDENCIES FROM STATE CHANGES
    !=============
    !convert theta to T
    temp=th/state%exner

    !note s=cp*T has units J/kg
    ptend%s                = cpair*(temp - state%t)/dtime 
    ptend%q(:,:,1)         = (qv - state%q(:,:,1) )/dtime
    ptend%q(:,:,ixcldliq)  = (cldliq - state%q(:,:,ixcldliq) )/dtime
    ptend%q(:,:,ixnumliq)  = (numliq - state%q(:,:,ixnumliq) )/dtime
    ptend%q(:,:,ixrain)    = (rain - state%q(:,:,ixrain) )/dtime
    ptend%q(:,:,ixnumrain) = (numrain - state%q(:,:,ixnumrain) )/dtime
    ptend%q(:,:,ixice)     = (ice - state%q(:,:,ixice) )/dtime
    ptend%q(:,:,ixnumice)  = (numice - state%q(:,:,ixnumice) )/dtime
    ptend%q(:,:,ixrim)     = (rim - state%q(:,:,ixrim) )/dtime
    ptend%q(:,:,ixrimvol)  = (rimvol - state%q(:,:,ixrimvol) )/dtime


    !WRITE OUTPUT
    !=============
    !call outfld('P3_QCAUT',   qcaut_grid,  pcols, lchnk)

    !TODO: add other outfld calls. Probably not worth doing until we get the code compiling...

  end subroutine micro_p3_interface_tend

  !================================================================================================

end module micro_p3_interface
