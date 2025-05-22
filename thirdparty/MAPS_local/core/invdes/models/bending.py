import torch

from .base_optimization import BaseOptimization, DefaultOptimizationConfig


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
                    border_width=[0, 0, 4, 4],
                    PML=[0.8, 0.8],
                    cell_size=None,
                    resolution=50,
                    wl_cen=1.55,
                    wl_width=0,
                    n_wl=1,
                    plot_root="./figs/bending",
                ),
                obj_cfgs=dict(
                    fwd_trans=dict(
                        weight=1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="out_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                        wl=[1.55],
                        temp=[300],
                        out_modes=(
                            "Hz1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="eigenmode",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                        # type="flux",  # the reason that the energy is not conserved is that the forward efficiency is caluculated in terms of the eigenmode coeff not the flux
                        direction="y+",
                    ),
                    refl_trans=dict(
                        weight=-0.1,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="refl_slice_1",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                        wl=[1.55],
                        temp=[300],
                        out_modes=(
                            "Hz1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux_minus_src",
                        direction="x",
                    ),
                    rad_trans_xp=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_xp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Hz1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    rad_trans_xm=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_xm",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Hz1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="x",
                    ),
                    rad_trans_yp=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_yp",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Hz1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                    rad_trans_ym=dict(
                        weight=0,
                        #### objective is evaluated at this port
                        in_slice_name="in_slice_1",
                        out_slice_name="rad_slice_ym",
                        #### objective is evaluated at all points by sweeping the wavelength and modes
                        wl=[1.55],
                        temp=[300],
                        in_mode="Hz1",  # only one source mode is supported, cannot input multiple modes at the same time
                        out_modes=(
                            "Hz1",
                        ),  # can evaluate on multiple output modes and get average transmission
                        type="flux",
                        direction="y",
                    ),
                ),
            )
        )


class BendingOptimization(BaseOptimization):
    def __init__(
        self,
        device,
        hr_device,
        design_region_param_cfgs=dict(),
        sim_cfg: dict = dict(),
        obj_cfgs=dict(),
        operation_device=torch.device("cuda:0"),
    ):
        design_region_param_cfgs = dict()
        for region_name in device.design_region_cfgs.keys():
            design_region_param_cfgs[region_name] = dict(
                method="levelset",
                interpolation="gaussian",
                # interpolation="bilinear",
                rho_resolution=[25, 25],
                transform=[
                    dict(type="transpose_symmetry", rot_k=3),
                    dict(type="blur", mfs=0.1, resolutions=[hr_device.resolution, hr_device.resolution], dim="xy"),
                    # dict(type="binarize"),
                    # dict(type="litho", res=310),
                    dict(type="binarize"),
                ],
                init_method="random",
                # init_method="ring",
                binary_projection=dict(
                    fw_threshold=100,
                    bw_threshold=100,
                    mode="regular",
                ),
            )

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
