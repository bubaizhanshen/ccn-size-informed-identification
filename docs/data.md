# Data preparation

## Hourly aerosol archive

Download the SMPS and CCN CSV files from the
[Figshare archive](https://figshare.com/articles/dataset/27913806). Keep the
original filenames and columns. Arrange the extracted files as follows:

```text
workspace/
  inputs/
    figshare_27913806/
      selected/
        smps/       *_smps_hour*.csv
        ccn/        *_ccn_colb_hour*.csv
        ccn_cola/   *_ccn_cola.csv
```

The record identifiers are `ANX`, `COR`, `ENA`, `GUC`, `MAO`, `MOS`,
`SBS_CP`, `SBS_SPL` and `SGP`. The two SBS records use column-A CCN files;
the others use the hourly column-B files. The analysis uses CCN at 0.4%
supersaturation, with number concentrations in cm⁻³ and timestamps in UTC.

Required SMPS fields include `Start_date`, the `D_*` size channels,
`N_CN_SMPS_STP`, `Qc_CPC_SMPS`, `Qc_ACSM_SMPS`, `P_SMPS`, `T_SMPS`
and `RH_SMPS`.
The `D_*` values are number densities, dN/dlog₁₀(Dp), not concentrations
per bin. The reader obtains diameters in nm from the channel names and
integrates with log-diameter bin widths.

Column-B CCN files require `Start_date`, `SS_setpoint_B`,
`N_CCN_mean_STP_B`, `Qc_CPC_SMPS` and `Qc_CCN04_N80_B`. Column-A files
require `Start_date`, `SS_harmonized_A`, `N_CCN_mean_STP_A` and
`Qc_CCN04_N80_A`. The CCN–N80 flag is retained for diagnostic comparisons;
it is not used to screen the primary cohort.

Preprocessing maps spectra conservatively to 24 log-spaced bins over
15–300 nm. It checks number conservation and agreement with the supplied
SMPS total before allowing analysis. The 82 nm count uses partial overlap
at the cutoff rather than assigning a whole bin to one side.

## Optional SGP meteorological inputs

The meteorological controls use daily `sgpmetE13.b1` NetCDF files from the
[ARM Data Center](https://adc.arm.gov/), covering 2017-04-01 through
2023-10-16:

```text
workspace/inputs/arm_sgpmetE13_b1/sgpmetE13.b1.*.nc
```

These checks need the complete stated period. They read temperature,
relative humidity, pressure, wind speed, wind direction and precipitation,
apply the supplied quality flags and aggregate to hourly observations.
ARM authentication is handled outside this repository.

## Local files

Set `CCN_WORKSPACE` before running a script if the inputs are stored
elsewhere. Results go to its `results/` subdirectory and temporary files
to `cache/`. Many stages stop if completed outputs already exist; use a
new workspace for a fresh run.

Use a fresh results directory when rerunning the reorganized code. Old
output records are not interchangeable with the current file-record format.
