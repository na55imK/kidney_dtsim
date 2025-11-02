# post_transplant_risk_helpers.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from multiprocessing import Pool

def calc_hazard(cum_haz):
    lag_cum_haz = np.roll(cum_haz, 1)

    # Set the first lagged value to NaN, as there is no lag for the first element
    lag_cum_haz[0] = np.nan

    # Calculate hazard_gf
    hazard_rate = cum_haz - lag_cum_haz

    # Handle the first element where lag_cumulative_hazard_gf is NaN
    hazard_rate[np.isnan(lag_cum_haz)] = cum_haz[np.isnan(lag_cum_haz)]

    return hazard_rate
  
def calc_cumulative_hazard_dwfg(donor_age, recipient_age, parameters, spline_terms):

    intercept = 1

    # Extract columns by name
    rcs1 = spline_terms["rcs1"].to_numpy()
    rcs2 = spline_terms["rcs2"].to_numpy()
    rcs3 = spline_terms["rcs3"].to_numpy()
    tvc1_rcs1 = spline_terms["tvc1_rcs1"].to_numpy()
    tvc1_rcs2 = spline_terms["tvc1_rcs2"].to_numpy()

    cumulative_hazard_dwfg = np.exp(
        parameters["_parametercons"] * intercept +
        parameters["_parameterdonor_age"] * donor_age +
        parameters["_parameterrec_age"] * recipient_age +
        parameters["_parameterrcs1"] * rcs1 +
        parameters["_parameterrcs2"] * rcs2 +
        parameters["_parameterrcs3"] * rcs3 +
        parameters["_parameterrcs_rec_age1"] * recipient_age * tvc1_rcs1 +
        parameters["_parameterrcs_rec_age2"] * recipient_age * tvc1_rcs2
    )

    return cumulative_hazard_dwfg
    
def calc_cumulative_hazard_gf(donor_age, recipient_age, parameters, spline_terms):
    
    intercept = 1
    da_2 = donor_age ** 2
    ra_2 = recipient_age ** 2

    # Extract columns by name
    rcs1 = spline_terms["rcs1"].to_numpy()
    rcs2 = spline_terms["rcs2"].to_numpy()
    rcs3 = spline_terms["rcs3"].to_numpy()
    tvc1_rcs1 = spline_terms["tvc1_rcs1"].to_numpy()
    tvc1_rcs2 = spline_terms["tvc1_rcs2"].to_numpy()
    tvc2_rcs1 = spline_terms["tvc2_rcs1"].to_numpy()
    tvc2_rcs2 = spline_terms["tvc2_rcs2"].to_numpy()

    cumulative_hazard_gf = np.exp(
        parameters["_parametercons"] * intercept +
        parameters["_parameterdonor_age"] * donor_age +
        parameters["_parameterda_2"] * da_2 +
        parameters["_parameterrec_age"] * recipient_age +
        parameters["_parameterra_2"] * ra_2 +
        parameters["_parameterrcs1"] * rcs1 +
        parameters["_parameterrcs2"] * rcs2 +
        parameters["_parameterrcs3"] * rcs3 +
        parameters["_parameterrcs_rec_age1"] * recipient_age * tvc1_rcs1 +
        parameters["_parameterrcs_rec_age2"] * recipient_age * tvc1_rcs2 +
        parameters["_parameterrcs_ra_21"] * ra_2 * tvc2_rcs1 +
        parameters["_parameterrcs_ra_22"] * ra_2 * tvc2_rcs2
    )

    return cumulative_hazard_gf
    
def simulate_time_from_uniform(U, cum_haz, max_time):
    """
    Simulate survival time T by solving H(T) =  - log(U)
    Solve H(T) = target for T using numerical optimization
    Since t is a continuous variable (float), indexing composite_cum_haz requires handling:
    - One option would be the use of interpolation to compute composite_cum_haz at non-integer values of t.
      np.interp(t, np.arange(len(composite_cum_haz)), composite_cum_haz)
    - The here used option is to round t to the nearest integer, avoiding the need for interpolation.
    """
    target = -np.log(U)

    survival_time_optimization = minimize_scalar(
        lambda t: np.abs(cum_haz[int(round(t))] - target),
        bounds=(0, max_time-1),
        method="bounded"
        )

    return survival_time_optimization.x

def process_patient(args):
    donor_age, recipient_age, spline_gf, spline_dwfg, parameters_gf, parameters_dwfg, t_max, U, pred_dial_free_surv_time, pred_death_time, pred_gf_time = args
    
    cum_haz_gf = calc_cumulative_hazard_gf(donor_age, recipient_age, parameters=parameters_gf, spline_terms=spline_gf)
    hazard_gf = calc_hazard(cum_haz_gf)

    cum_haz_dwfg = calc_cumulative_hazard_dwfg(donor_age, recipient_age, parameters=parameters_dwfg, spline_terms=spline_dwfg)
    hazard_dwfg = calc_hazard(cum_haz_dwfg)

    composite_cum_haz = cum_haz_gf + cum_haz_dwfg
    composite_survival = np.exp(-composite_cum_haz)
    gf_composite_survival = hazard_gf * composite_survival
    dwfg_composite_survival = hazard_dwfg * composite_survival

    cum_inc_gf = np.cumsum(gf_composite_survival)
    cum_inc_dwfg = np.cumsum(dwfg_composite_survival)

    sim_surv_time, sim_death_time, sim_gf_time = None, None, None
    if pred_dial_free_surv_time:
        sim_surv_time = simulate_time_from_uniform(U, composite_cum_haz, t_max)
    if pred_death_time:
        sim_death_time = simulate_time_from_uniform(U, cum_haz_dwfg, t_max)
    if pred_gf_time:
        sim_gf_time = simulate_time_from_uniform(U, cum_haz_gf, t_max)
        
    return(1 - composite_survival[int(round(365.25*5))],
           1 - composite_survival[int(round(365.25*10))],
           cum_inc_gf[int(round(365.25*5))],
           cum_inc_gf[int(round(365.25*10))],
           cum_inc_dwfg[int(round(365.25*5))],
           cum_inc_dwfg[int(round(365.25*10))],
           composite_survival,
           sim_surv_time,
           sim_death_time,
           sim_gf_time
           )

def calc_risks(donor_ages,
               recipient_ages,
               spline_gf,
               spline_dwfg,
               parameters_gf,
               parameters_dwfg,
               t_max,
               pred_dial_free_surv_time = True,
               pred_death_time = False,
               pred_gf_time = False,
               random_numbergen=np.random.default_rng(0),
               processes = None):

    t_max = int(t_max) # convert t_max to an int

    #the index is passed as a string, when called via r,
    #thus the dataframe is reindexed, to circumvent this
    spline_gf = spline_gf.reset_index(drop=True)
    spline_dwfg = spline_dwfg.reset_index(drop=True)
    
    spline_gf = spline_gf[0:t_max]
    spline_dwfg = spline_dwfg[0:t_max]

    list_cum_inc_comp_5y = [None] * len(donor_ages)
    list_cum_inc_comp_10y = [None] * len(donor_ages)
    list_cum_inc_gf_5y = [None] * len(donor_ages)
    list_cum_inc_gf_10y = [None] * len(donor_ages)
    list_cum_inc_dwfg_5y = [None] * len(donor_ages)
    list_cum_inc_dwfg_10y = [None] * len(donor_ages)

    list_sim_survival_times = [None] * len(donor_ages)
    list_sim_death_times = [None] * len(donor_ages)
    list_sim_gf_times = [None] * len(donor_ages)
    
    survival_curves = [None] * len(donor_ages)

    U = random_numbergen.uniform(0, 1, len(donor_ages))
    args_list = [(donor_ages[i], recipient_ages[i], spline_gf,
                  spline_dwfg, parameters_gf, parameters_dwfg,
                  t_max, U[i], pred_dial_free_surv_time, pred_death_time, pred_gf_time)
                  for i in range(len(donor_ages))]

    print("Processing Patients", flush=True)
    pool = Pool(processes)
    results = pool.map(process_patient, args_list)
    #results = [process_patient(args) for args in args_list]
    
    print("Zipping Results", flush=True)

    (list_cum_inc_comp_5y,
     list_cum_inc_comp_10y,
     list_cum_inc_gf_5y,
     list_cum_inc_gf_10y,
     list_cum_inc_dwfg_5y,
     list_cum_inc_dwfg_10y,
     survival_curves,
     list_sim_survival_times,
     list_sim_death_times,
     list_sim_gf_times) = zip(*results)

    mean_survival_probs = np.mean(survival_curves, axis=0)

    print("Return Results", flush=True)
    df_post_tx_risk = pd.DataFrame({
        "pred_cum_inc_comp_5y": list_cum_inc_comp_5y,
        "pred_cum_inc_comp_10y": list_cum_inc_comp_10y,
        "pred_cum_inc_gf_5y": list_cum_inc_gf_5y,
        "pred_cum_inc_gf_10y": list_cum_inc_gf_10y,
        "pred_cum_inc_dwfg_5y": list_cum_inc_dwfg_5y,
        "pred_cum_inc_dwfg_10y": list_cum_inc_dwfg_10y,
        "pred_dialysis_free_surv_time": list_sim_survival_times,
        "pred_death_time": list_sim_death_times,
        "pred_gf_time": list_sim_gf_times
    })
    return {"df_post_tx_risk": df_post_tx_risk, "mean_survival_probs": mean_survival_probs}
