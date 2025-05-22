import torch

from .base_optimization import BaseOptimization, DefaultOptimizationConfig
from pyutils.config import Config


class DefaultConfig(DefaultOptimizationConfig):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                design_region_param_cfgs=dict(),
                sim_cfg=dict(
                    solver="ceviche",
                    binary_projection=dict(
                        fw_threshold=180,
                        bw_threshold=180,
                        mode="regular",
                    ),
                    border_width=[0, 0, 1.5, 1.5],
                    PML=[0.8, 0.8],
                    cell_size=None,
                    resolution=50,
                    wl_cen=0.85,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/metalens",
                ),
                obj_cfgs=dict(
                    # fwd_trans=dict(
                    #     weight=1,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="farfield_1",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=("Hz1",),
                    #     type="flux_near2far",
                    #     direction="x+",
                    # ),
                    # rad_trans_yp=dict(
                    #     weight=-0.2,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_slice_yp",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=("Hz1",),
                    #     type="flux_near2far",
                    #     direction="x+",
                    # ),
                    # rad_trans_farfield_yp=dict(
                    #     weight=-0.2,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_trans_farfield_yp",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=("Hz1",),
                    #     type="flux_near2far",
                    #     direction="x+",
                    # ),
                    # fwd_refl_trans=dict(
                    #     weight=-0.1,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="refl_slice_1",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Hz1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="flux_minus_src",
                    #     direction="x",
                    # ),
                    # rad_trans_yp=dict(
                    #     weight=-0.2,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_slice_yp",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Hz1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     # type="flux_near2far",
                    #     type="flux",
                    #     direction="y",
                    # ),
                    # rad_trans_farfield_yp=dict(
                    #     weight=-0.2,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_trans_farfield_yp",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Hz1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     # type="flux_near2far",
                    #     type="flux_near2far",
                    #     direction="y",
                    # ),
                    # rad_trans_ym=dict(
                    #     weight=-0.2,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_slice_ym",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Hz1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     # type="flux_near2far",
                    #     type="flux",
                    #     direction="y",
                    # ),
                    # rad_trans_farfield_ym=dict(
                    #     weight=-0.2,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_trans_farfield_ym",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         "Hz1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     # type="flux_near2far",
                    #     type="flux_near2far",
                    #     direction="y",
                    # ),
                    # rad_trans_xp=dict(
                    #     weight=0,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_slice_ym",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         1,
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     # type="flux_near2far",
                    #     type="flux",
                    #     direction="y",
                    # ),
                    # rad_trans_farfield_ym=dict(
                    #     weight=-0.2,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="rad_trans_farfield_ym",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     temp=[300],
                    #     wl=[0.85],
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     out_modes=(
                    #         1,
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     # type="flux_near2far",
                    #     type="flux_near2far",
                    #     direction="y",
                    # ),

                    # tot_ff_reg_plt=dict(
                    #     weight=0,
                    #     #### objective is evaluated at this port
                    #     in_slice_name="in_slice_1",
                    #     out_slice_name="total_farfield_region",
                    #     #### objective is evaluated at all points by sweeping the wavelength and modes
                    #     in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                    #     wl=[0.85],
                    #     temp=[300],
                    #     out_modes=(
                    #         "Hz1",
                    #     ),  # can evaluate on multiple output modes and get average transmission
                    #     type="flux_near2far",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                    #     direction="x+",
                    # ),

                    near_field_response_record=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="nearfield_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        temp=[300],
                        wl=[0.85],
                        in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=("Hz1",),
                        type="response_record",
                        direction="x+",
                    ),
                    source_field_response_record=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="in_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        temp=[300],
                        wl=[0.85],
                        in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=("Hz1",),
                        type="response_record",
                        direction="x+",
                    ),
                ),
            )
        )


class MetaLensOptimization(BaseOptimization):
    def __init__(
        self,
        device,
        hr_device,
        design_region_param_cfgs=dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device=torch.device("cuda:0"),
        initialization_file=None,
    ):
            # design_region_param_cfgs[region_name] = dict(
            #     method="levelset",
            #     rho_resolution=[0, 1/0.25],
            #     interpolation="bilinear",
            #     # transform=[dict(type="mirror_symmetry", dims=[1])],
            #     transform=[],
            #     # init_method="grating_1d_random",
            #     # init_method="grating_1d_minmax",
            #     # init_method="grating_0.2",
            #     init_method="grating_1d_random",
            #     # init_method="random",
            #     binary_projection=dict(
            #         fw_threshold=100,
            #         bw_threshold=100,
            #         mode="regular",
            #     ),
            #     initialization_file=initialization_file,
            # )
        design_region_param_cfgs_copy = design_region_param_cfgs.copy()
        design_region_param_cfgs = dict()
        for region_name in device.design_region_cfgs.keys():
            design_region_param_cfgs[region_name] = dict(
                method=design_region_param_cfgs_copy.get("method", "levelset"),
                rho_resolution=design_region_param_cfgs_copy.get("rho_resolution", [0, 1/0.15]),
                interpolation=design_region_param_cfgs_copy.get("interpolation", "bilinear"),
                transform=design_region_param_cfgs_copy.get("transform", []),
                init_method=design_region_param_cfgs_copy.get("init_method", "grating_1d_random"),
                binary_projection=design_region_param_cfgs_copy.get(
                    "binary_projection",
                    dict(
                        fw_threshold=100,
                        bw_threshold=100,
                        mode="regular",
                    ),
                ),
                denorm_mode="linear_eps",
                initialization_file=initialization_file,
            )
            for key in design_region_param_cfgs_copy.keys():
                if key not in design_region_param_cfgs.keys():
                    design_region_param_cfgs[region_name][key] = design_region_param_cfgs_copy[key]

        cfgs = DefaultConfig()  ## this is default configurations
        ## here we accept new configurations and update the default configurations
        cfgs.update(
            dict(
                design_region_param_cfgs=design_region_param_cfgs,
                sim_cfg=sim_cfg,
                obj_cfgs=obj_cfgs,
            )
        )

        super().__init__(
            device=device,
            hr_device=hr_device,
            design_region_param_cfgs=cfgs.design_region_param_cfgs,
            sim_cfg=cfgs.sim_cfg,
            obj_cfgs=cfgs.obj_cfgs,
            operation_device=operation_device,
        )
