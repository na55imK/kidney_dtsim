# model.py
import warnings
import numpy as np

from scipy.stats import gaussian_kde
from math import floor

# Data manipulation and analysis.
import pandas as pd

# ABM
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import itertools

# Time module to measure code execution time
import time

# for multiprocessing
from multiprocessing import Pool


# helper functions for the model -------------------------------------------------------

def batch_run(constant_params, variable_params, processes=None):
    """
    Run multiple simulations with constant and variable parameters, merging patient and donor data
    and returning combined results.

    This function performs batch simulations by combining constant parameters with a set of variable
    parameters for each simulation run. Each simulation is executed in parallel, and the results 
    are processed to merge patient and donor data at the last simulation step. The function returns
    concatenated DataFrames of all patient-donor data and model data across simulation runs, with
    each run and its parameters uniquely identified.

    Parameters
    ----------
    constant_params : dict
        A dictionary of constant parameters used for each simulation run.

    variable_params : list of dict
        A list of dictionaries, where each dictionary contains a unique set of parameters that vary
        for each simulation run. Each dictionary is combined with `constant_params` to create a full
        set of parameters for a run.
    processes: optional
         Number of worker processes to use. If processes is None then the number returned by
         os.process_cpu_count() is used.

    Returns
    -------
    tuple of pd.DataFrame
        - combined_patient_donor_data : pd.DataFrame
            A DataFrame containing merged patient and donor data from all simulation runs, with 
            `model_run_id` and the variable parameters added as columns for each run.
        - combined_model_data : pd.DataFrame
            A DataFrame containing model-specific data from all simulation runs, with `model_run_id`
            and the variable parameters added as columns for each run.

    Notes
    -----
    - The function calculates the total number of simulation steps based on `years_to_simulate` and 
      `step_size`.
    - Simulations are executed in parallel using a multiprocessing pool.
    - For each run, data from the final step is retrieved, and patients are merged with their linked
      donors.
    - `model_run_id` and each variable parameter are included as identifiers in the output DataFrames.
    - The merged patient-donor and model DataFrames are concatenated across runs to create a single 
      DataFrame for each.

    Example
    -------
    >>> constant_params = {"years_to_simulate": 10, "step_size": 0.5}
    >>> variable_params = [
    ...     {"param1": 1, "param2": 2},
    ...     {"param1": 3, "param2": 4},
    ... ]
    >>> combined_patient_donor_data, combined_model_data = batch_run(constant_params, variable_params)
    """
    steps = 1+ int(constant_params.get("years_to_simulate") / constant_params.get("step_size"))

    # Combine constant and varying parameters for each simulation
    parameter_sets = [{**constant_params, **var_params} for var_params in variable_params]

    # Run the simulations in parallel
    pool = Pool(processes)
    results = pool.map(run_simulation, parameter_sets)

    all_merged_patients_donor_data = []
    all_model_data = []

    for run_id, (agent_data, model_data) in enumerate(results, start=1):
        # Get the variable parameters for this run
        var_params = variable_params[run_id - 1]  # Adjusting for zero-based indexing

        # Get data from the last step of the model and merge donor to linked patients
        end_agent_data = agent_data.xs(steps-1, level="Step")
        merged_patients_donor = merge_patients_and_donor(end_agent_data)

        # Add the run_id to each DataFrame
        merged_patients_donor.insert(0, "model_run_id", run_id)
        model_data.insert(0, "model_run_id", run_id)

        # Add each variable parameter as a column
        for key, value in var_params.items():
            merged_patients_donor.insert(1, "model_" + key, value)
            model_data.insert(1, "model_" + key, value)

        # Append to lists
        all_merged_patients_donor_data.append(merged_patients_donor)
        all_model_data.append(model_data)

    # Concatenate all agent_data and model_data into big DataFrames
    combined_patient_donor_data = pd.concat(all_merged_patients_donor_data, ignore_index=False)
    combined_model_data = pd.concat(all_model_data, ignore_index=False)

    return combined_patient_donor_data, combined_model_data


def run_simulation(params, quiet = True):
    """
    Runs a simulation of the organ allocation model over a specified number of steps, 
    using parameters from the provided dictionary.

    Parameters
    ----------
    params : dict
        A dictionary of parameters required for the simulation. The dictionary should contain all variables necessary for the Allocation model.
        Additionally it should contain years_to_simulate.
    quiet : bool
        If true, the model doesn't print information at each step.

    Returns
    -------
    tuple
        A tuple containing:
            - agent_data : DataFrame
                Data on agent-level variables collected during the simulation.
            - model_data : DataFrame
                Data on model-level variables collected during the simulation.

    Notes
    -----
    The function initializes an `AllocationModel` object using the parameters provided in `params`, 
    calculates the number of steps required based on `years_to_simulate` and `step_size`, and 
    runs the model for the specified number of steps. At each step, it prints progress information 
    (step number, day count) and advances the model. Finally, it returns two DataFrames with collected data.

    Example
    -------
    params = {
        'df_init_patients': patient_df,
        'n_pts_add_per_step': 5,
        'n_organ_donors_step': 3,
        'pts_age_list': [30, 40, 50],
        ...
        'years_to_simulate': 10,
        'step_size': 0.1,
        'seed': 42
    }
    agent_data, model_data = run_simulation(params)
    """

    model = AllocationModel(
        df_init_patients=params["df_init_patients"],
        n_pts_add_per_step=params["n_pts_add_per_step"],
        n_organ_donors_step=params["n_organ_donors_step"],
        pts_age_list=params["pts_age_list"],
        donor_age_list=params["donor_age_list"],
        vpra_df=params["vpra_df"],
        removal_model=params["removal_model"],
        df_nt_status_prob=params["df_nt_status_prob"],
        dict_log_odds_ETKAS=params["dict_log_odds_ETKAS"],
        dict_log_odds_ESP=params["dict_log_odds_ESP"],
        df_dial_to_reg_time=params["df_dial_to_reg_time"],
        hla_a_freq_dict=params["hla_a_freq_dict"],
        hla_b_freq_dict=params["hla_b_freq_dict"],
        hla_dr_freq_dict=params["hla_dr_freq_dict"],
        hla_hapl_freq_df=params["hla_hapl_freq_df"],
        donor_reg_freq_dict=params["donor_reg_freq_dict"],
        recipient_reg_freq_dict=params["recipient_reg_freq_dict"],
        living_donor_prob_dict=params["living_donor_prob_dict"],
        donor_blood_grp_freq_dict=params["donor_blood_grp_freq_dict"],
        recipient_blood_grp_freq_dict=params["recipient_blood_grp_freq_dict"],
        max_rank_pts_organ_offer=params["max_rank_pts_organ_offer"],
        et_groups_bool=params["et_groups_bool"],
        esp_national_allo=params["esp_national_allo"],
        extended_allo=params["extended_allo"],
        esp_age_border=params["esp_age_border"],
        prob_stay_in_etkas=params["prob_stay_in_etkas"],
        step_size=params["step_size"],
        datacollector_interval=params["datacollector_interval"],
        seed=params["seed"],
        quiet = quiet
    )

    steps = 1 + int(params["years_to_simulate"] / params["step_size"])

    for i in range(steps):
        print(f"Simulation (seed {params['seed']}): Step {i}, Day {i * model.step_size * 365}")
        model.step()

    agent_data = model.datacollector.get_agent_vars_dataframe()
    model_data = model.datacollector.get_model_vars_dataframe()

    return agent_data, model_data

def merge_patients_and_donor(agent_data_step):
    """
    Merge patient and organ donor data from a DataFrame based on linked donor information.

    This function takes a DataFrame containing both patient and organ donor information,
    filters it by agent type, and then merges the data on the donor linkage.
    The resulting DataFrame includes recipient and donor-specific columns with suffixes 
    to differentiate between recipient and donor attributes.

    Parameters:
    ----------
    agent_data_step : pd.DataFrame
        A DataFrame containing agent data with columns including `agent_type`, 
        `AgentID`, `linked_donor`, and attributes specific to patients and donors such as 
        `age`, `blood_group`, `status`, `hla`, and `declined_offers`.

    Returns:
    -------
    pd.DataFrame
        A merged DataFrame with patient data and linked donor data combined. Columns 
        from the original DataFrame are suffixed with `_recipient` or `_donor` to 
        differentiate between recipient (patient) and donor attributes. 
        The `declined_offers` column is retained only for donors.
        
    Notes:
    ------
    - The function drops the `declined_offers` column from patient data.
    - Only selected columns (`age`, `blood_group`, `status`, `hla`, `declined_offers`) are 
      retained for donor data.
    - The merging is performed using an outer join on the `linked_donor` column.
    """
    
    # Filter DataFrame for patients and donors
    df_patients = agent_data_step[agent_data_step["agent_type"] == "patient"]
    df_donor = agent_data_step[agent_data_step["agent_type"] == "organ_donor"]
    
    df_patients = df_patients.drop("declined_offers", axis = 1)
    df_donor = df_donor.loc[:, ["age", "blood_group", "status", "region", "hla", "declined_offers"]]
    
    df_donor.reset_index(inplace=True)
    
    df_donor.rename(columns={"AgentID": "linked_donor"}, inplace=True)
    
    df_patients.reset_index(inplace=True)
    merged_df = pd.merge(df_patients, df_donor,
                         how="outer", on="linked_donor", suffixes=["_recipient", "_donor"])
    
    return merged_df

# function for checking processing time
def start_timer():
    return time.time()

# Function to calculate and print the elapsed time since start
def end_timer(start_time, label=None, quiet = False):
    if not quiet:
        elapsed_time = round(time.time() - start_time, 5)
        if label:
            print(f"    {label} time: {elapsed_time}s")
        else:
            print(f"    Time taken: {elapsed_time}s")


def get_n_pts_waiting(model):
    return len([agent for agent in model.schedule.agents if (isinstance(agent, Patient) and (agent.status in ["transplantable", "not_transplantable"]))])

def get_n_pts_waiting_transplantable(model):
    return len([agent for agent in model.schedule.agents if (isinstance(agent, Patient) and (agent.status == "transplantable"))])

def get_n_pts_transplanted(model):
    return len([agent for agent in model.schedule.agents if isinstance(agent, Patient) and (agent.status == "transplanted")])

def get_n_pts_removed(model):
    return len([agent for agent in model.schedule.agents if isinstance(agent, Patient) and agent.status == "removed"])

def get_n_pts_living_donor_transplanted(model):
    return len([agent for agent in model.schedule.agents if isinstance(agent, Patient) and agent.status == "living_donor_transplanted"])

def normalize_dict_values(input_dict):
    """helper function to normalize dictionary values"""
    total_sum = np.sum(list(input_dict.values()))
    normalized_dict = {key: value / total_sum for key, value in input_dict.items()}
    return normalized_dict

def distribute_patients_evenly(original_list, step_size):
     """
     Distributes the number of patients evenly according to the specified step size.
 
     Args:
     - original_list (list): The original list containing the number of patients.
     - step_size (float): The step size for distributing the patients evenly.
 
     Returns:
     - list: A new list with the number of patients distributed evenly based on the step size.
     """
 
     patients_per_step = []
     remainder = 0
     steps_per_period = int(round(1/step_size))
 
     # Loop through the original list
     for i in range(len(original_list)):
         remainder = original_list[i] % steps_per_period
 
         for step in range(steps_per_period):
             patients_per_step.append(floor(original_list[i] * step_size))
 
             if step < remainder:
                patients_per_step[-1] += 1
 
     return patients_per_step


# Agent classes ---------------------------------------------------

class Patient(Agent):
    """
    A class representing a patient within the simulation, inheriting from Agent. This class encapsulates the 
    properties and behaviors of a patient awaiting transplantation.

    Attributes:
        unique_id (int): A unique identifier for the patient.
        step_added_to_model (int): Step at which the agent was added to the model.
        waitinglist_time (float): The time spent on the waiting list in years,
                                  initialized randomly between 0 and 1 for new patients.
        time_dial_to_reg (int): Time in days. Resampled from Gaussian Kernel Density Estimation (KDE)
                                by age groups of 10 years. If a dialyis time is already there, then
                                time_dial_to_reg is calculated by dialysis_time subtracted by waitinglist_time. 
        dialysis_time (float): The time spent on dialysis in years. Calculated by waitinglist_time + time_to_dial_to_reg.
        status (str): The current status of the patient. Possible values are "transplantable", "not transplantable", "transplanted",
                      "living_donor_transplanted", or "removed" from the list.
        step_transplanted (int): The simulation step at which the patient was transplanted. None if not transplanted.
        step_removed_from_list (int): The simulation step at which the patient was removed from the list. None if not removed.
        age (int): The age of the patient, assigned at initialization or resampled from a Gaussian Kernel Density Estimation (KDE).
        blood_group (str): The blood group of the patient, either assigned at initialization or resampled based on frequency.
        et_program (str): The Eurotransplant (ET) program assigned to the patient, based on age. "esp" for patients over 65, 
                          "etkas" for those under 65. Only assigned if the model's `et_groups_bool` attribute is True. Patients stay with a 36% probability in the EKTAS program.
        hla_alleles (dict): A dictionary representing the patient's HLA type.
        vpra (float): The vPRA value. Resampled from the peak measured vPRA values of all candidates.
        unacceptable_antigens (list): Unacceptable antigens, filtered first for HLA type antigens present in the unacceptable antigens list and then
                        resampled from the peak measured vPRA of all candidates.
        hla_n_homozygous_loci (int): The number of homozygous loci in the patient's HLA alleles.
        mmp (float): The mismatch probability points for the patient.
        hla_mm(int): HLA mismatches to the donor. If the patient is not transplanted the value is none.
        region (str): The DSO region of the patient, either assigned at initialization or resampled from frequency.
        subregion (str): random choosen subregion, from the subregions in the DSO region.
        cum_haz (array): An array of cumulative hazard for removal from the waiting list, predicted based on a removal model and the patient's age at registration.
        surv_time (int): Predicted time until removal from waiting list.

    Methods:
        __init__(self, unique_id, model, age=None, waitinglist_time=0, blood_group=None, hla_alleles=None, vpra=None, region=None):
            Initializes a new patient agent with the specified attributes.

        assign_subregion(self):
            Random assignment to one of the subregions in the region.

        calc_mmp(self):
            Calculates the mismatch probability (MMP) for the patient using their HLA alleles, blood group, and the model's allele frequency dictionaries.

        predict_transplantable_status(self, interpolate=True):
            Predicts whether the patient is not transplantable based on their waiting time and age at waiting list registration.

        set_patient_random_nt(self):
            Randomly assigns the patient's status to either "not_transplantable" or "transplantable" based on the predicted probability.
        
        calc_cumulative_hazard(self):
            Calculates the cumulative hazard for the patient being removed from the waiting list based on their age at registration.    

        calc_surv_prob_in_next_step(self):
            Calculates the patient's survival probability in the next simulation step based on their cumulative hazard.

        remove_patient(self):
            Determines if the patient should be removed from the waiting list in the current step based on their survival probability.

        simulate_time_from_uniform(self, U, cum_haz):
            Simulates survival time by solvin H(T) = -log(U)

        predict_survival_time(self):
            Prediction of survival time by the approach from Bender et al. Calls the method
            simulate_time_from_uniform
            
        step(self):
            Performs one simulation step for the patient, updating their waiting time, age, and status, and potentially removing them from the list if necessary.
    """

    def __init__(self, unique_id, model, age= None, dialysis_time = None, waitinglist_time = 0, blood_group = None, status = None, hla_alleles = None, et_program = None, vpra = None, unacceptable_antigens = None, region = None):
        super().__init__(unique_id, model)
        self.model: AllocationModel = model
        self.type = "patient"
        self.step_added_to_model = self.model.schedule.time
        self.step_transplanted = None
        self.step_removed_from_list = None

        # set age
        self.age = age
        if pd.isnull(self.age) or self.age is None:
            self.age = self.model.sample_ages(1)[0]

        # set waiting time
        self.waitinglist_time = waitinglist_time
        if self.waitinglist_time > self.age: 
            self.waitinglist_time = self.age

        # set dialysis time
        self.dialysis_time = dialysis_time
        if pd.isnull(self.dialysis_time) or self.dialysis_time is None:
            self.dialysis_time = self.model.sample_dialysis_times(self.age, self.waitinglist_time)

        self.time_dial_to_reg = (self.dialysis_time - self.waitinglist_time)*365
        
        # set living donor transplantation
        self.status = status

        
        if self.model.schedule.steps == 0:
            # During model initialization, evaluate living donor donation status.
            # If the candidate is transplanted via a living donor, update the status.
            # Otherwise, retain the initial status or assign a new one if not set.
            if self.model.sample_living_donor_status(1,
                                                     self.age,
                                                     self.waitinglist_time) == "living_donor_transplanted":
                self.status = "living_donor_transplanted"

            if pd.isnull(self.status) or self.status is None:
                self.set_patient_random_nt()
        
        else:
            # For subsequent steps, check and assign status if not already set.
            # The default status returned from sample_living_donor_status is "transplanted".
            # Patients with this status are further assessed to determine if they are transplantable.
            if pd.isnull(self.status) or self.status is None:
                self.status = self.model.sample_living_donor_status(1,
                                                         self.age,
                                                         self.waitinglist_time)
            if self.status == "transplantable":
                self.set_patient_random_nt()

        # set blood group
        self.blood_group = blood_group
        if pd.isnull(self.blood_group) or self.blood_group is None:
            self.blood_group = self.model.sample_blood_groups(1)[0]
        
        self.removal_model = self.model.removal_model

        # assign et_program if et_group = True
        self.et_program = et_program
        if self.model.et_groups_bool:
            if pd.isnull(self.et_program) or self.et_program is None:
                self.et_program = self.model.assign_et_prgm(1, self.age)
        else:
            self.et_program = None

        # assign hla_alleles
        self.hla_alleles = hla_alleles
        if self.hla_alleles is None:
            self.hla_alleles = self.model.gen_hla_haplotypes()

        # assign vpra and unacceptable antigens
        self.vpra = vpra
        self.unacceptable_antigens = unacceptable_antigens

        if not isinstance(self.unacceptable_antigens, list) and pd.notnull(self.unacceptable_antigens):
            raise TypeError(f"unacceptable_antigens value: {self.unacceptable_antigens}, is neither a list nor NAN")
        
        if pd.isnull(self.vpra) or self.vpra is None:
                self.vpra, self.unacceptable_antigens = self.model.assign_vpra_unaccept_antigen_bulk(self.hla_alleles)
        
        # calculate mmp
        self.mmp = self.calc_mmp()

        # has to be after mmp calculation
        # because in calc_mmp are missing second antigens (in cases of homozygoty) filled with the first antigen
        self.hla_n_homozygous_loci = self.model.calc_homozygous_loci(self.hla_alleles)

        self.hla_mm = None

        # set region
        self.region = region
        if self.region is None: 
            self.region = self.model.random.choice(list(self.model.recipient_reg_freq_dict.keys()),
                                              p=list(self.model.recipient_reg_freq_dict.values()))
        self.subregion = self.assign_subregion()

        self.cum_haz = self.calc_cumulative_hazard()

        self.surv_time = self.predict_survival_time()


    def assign_subregion(self):
        """
        Random assignment to one of the subregions in the region.
        """
        if self.region == "baden_württemberg":
            return(self.model.random.choice(["Stuttgart", "Freiburg"]))
        if self.region == "bayern":
            return(self.model.random.choice(["Muenchen", "Erlangen"]))
        if self.region == "mitte":
            return(self.model.random.choice(["Mainz", "Homburg", "Marburg"]))
        if self.region == "nord":
            return(self.model.random.choice(["Hannover", "Hamburg"]))
        if self.region == "nord_ost":
            return(self.model.random.choice(["Berlin", "Rostock"]))
        if self.region == "nordrhein_westfalen":
            return(self.model.random.choice(["Duesseldorf", "Koeln-Bonn", "Muenster"]))
        if self.region == "ost":
            return("Leipzig")
        

    def calc_mmp(self):
        """
        Calculation of the mismatch probability. The formula is derived from 
        the ET transplant manual chapter 4.

        Args:
            hla_alleles: dictionary with HLA allele pairs for A, B, and DR
            blood_grp: string with blood group ('A', 'B', 'O')
            vpra: vPRA in percent
            hla_a_freq_dict: dictionary with allele and frequency for HLA-A
            hla_b_freq_dict: dictionary with allele and frequency for HLA-B
            hla_dr_freq_dict: dictionary with allele and frequency for HLA-DR
            blood_grp_freq_dict: dictionary with the frequency of each blood group
        """
        # Extracting HLA allele frequencies
        a_1_freq = self.model.hla_a_freq_dict.get(self.hla_alleles["a_1"])
        
        if pd.isnull(self.hla_alleles["a_2"]):
            self.hla_alleles["a_2"] = self.hla_alleles["a_1"]
        
        a_2_freq = self.model.hla_a_freq_dict.get(self.hla_alleles["a_2"])

        b_1_freq = self.model.hla_b_freq_dict.get(self.hla_alleles["b_1"])
        
        if pd.isnull(self.hla_alleles["b_2"]):
            self.hla_alleles["b_2"] = self.hla_alleles["b_1"]
        b_2_freq = self.model.hla_b_freq_dict.get(self.hla_alleles["b_2"])

        dr_1_freq = self.model.hla_dr_freq_dict.get(self.hla_alleles["dr_1"])

        if pd.isnull(self.hla_alleles["dr_2"]):
            self.hla_alleles["dr_2"] = self.hla_alleles["dr_1"]
        dr_2_freq = self.model.hla_dr_freq_dict.get(self.hla_alleles["dr_2"])

        abo_freq = self.model.recipient_blood_grp_freq_dict.get(self.blood_group)
        # Error handling for missing frequencies
        for key, value in self.hla_alleles.items():
            if self.model.hla_a_freq_dict.get(value) is None and any(subkey in key for subkey in ["a_1", "a_2"]):
                raise ValueError(f"{value} frequency not found in HLA-A")
            if self.model.hla_b_freq_dict.get(value) is None and any(subkey in key for subkey in ["b_1", "b_2"]):
                raise ValueError(f"{value} frequency not found in HLA-B")
            if self.model.hla_dr_freq_dict.get(value) is None and any(subkey in key for subkey in ["dr_1", "dr_2"]):
                raise ValueError(f"{value} frequency not found in HLA-DR")

        if abo_freq is None:
            raise ValueError(f"{self.blood_group} blood_grp frequency not found")

        # Calculations
        a_sum2 = sum(value**2 for value in self.model.hla_a_freq_dict.values())
        b_sum2 = sum(value**2 for value in self.model.hla_b_freq_dict.values())
        dr_sum2 = sum(value**2 for value in self.model.hla_dr_freq_dict.values())

        mmp0 = ((a_1_freq + a_2_freq) ** 2) * ((b_1_freq + b_2_freq) ** 2) * ((dr_1_freq + dr_2_freq) ** 2)

        a = ((2 * (a_1_freq + a_2_freq) * (1 - a_1_freq - a_2_freq)) - a_1_freq ** 2 - a_2_freq ** 2 + a_sum2) / ((a_1_freq + a_2_freq) ** 2)
        b = ((2 * (b_1_freq + b_2_freq) * (1 - b_1_freq - b_2_freq)) - b_1_freq ** 2 - b_2_freq ** 2 + b_sum2) / ((b_1_freq + b_2_freq) ** 2)
        dr = ((2 * (dr_1_freq + dr_2_freq) * (1 - dr_1_freq - dr_2_freq)) - dr_1_freq ** 2 - dr_2_freq ** 2 + dr_sum2) / ((dr_1_freq + dr_2_freq) ** 2)

        mmp1 = mmp0 * (a + b + dr)

        mmp = 100 * (1 - (abo_freq * (1 - (self.vpra / 100)) * (mmp0 + mmp1))) ** 1000

        if mmp is None:
            raise ValueError(f"MMP calculation error")

        return mmp

    def predict_transplantable_status(self, interpolate=False):
        age_at_waitinglist_time_reg = self.age - self.waitinglist_time
        
        # get nearest age
        min_arg = np.abs(self.model.df_nt_status_prob["age_at_waiting_list_registration"].unique() - age_at_waitinglist_time_reg).argmin()
        nearest_age = self.model.df_nt_status_prob["age_at_waiting_list_registration"].unique()[min_arg]

        df_prob = self.model.df_nt_status_prob[self.model.df_nt_status_prob["age_at_waiting_list_registration"] == nearest_age]

        df_prob = df_prob.sort_index()

        if (interpolate):
            prob = np.interp(self.waitinglist_time*365, df_prob.index, df_prob["pstate.not_transplantable"])
        else:
            nearest_index = np.abs(df_prob.index - self.waitinglist_time*365).argmin()
            # If needed, get the corresponding row from the DataFrame
            prob = df_prob.iloc[nearest_index]["pstate.not_transplantable"]
        
        #print(f"Prob NT: {prob}, age at wating list registration {age_at_waitinglist_time_reg}, days waiting time {self.waitinglist_time*365}")
        return(prob)
    
    def set_patient_random_nt(self):
        if self.predict_transplantable_status(interpolate=False) >= self.model.random.random():
            self.status = "not_transplantable"
        else:
            self.status = "transplantable"
    
    def calc_cumulative_hazard(self):
        #t_max = np.max(self.model.removal_model.durations)
        t_max = 10000
        times = np.arange(1, t_max)

        df = pd.DataFrame({"age_at_reg_waiting_list": [self.age - self.waitinglist_time],
                           "no_zero_time_dial_to_registration": [1 if self.time_dial_to_reg <= 0 else self.time_dial_to_reg],
                           "sqrt_time_dial_to_registration": np.sqrt([1 if self.time_dial_to_reg <= 0 else self.time_dial_to_reg])})
        
        def assign_age_group(age):
            if age <= 29:
                return "0-29"
            elif age <= 39:
                return "30-39"
            elif age <= 44:
                return "40-44"
            elif age <= 49:
                return "45-49"
            elif age <= 54:
                return "50-54"
            elif age <= 59:
                return "55-59"
            elif age <= 64:
                return "60-64"
            elif age <= 69:
                return "65-69"
            elif age <= 74:
                return "70-74"
            elif age <= 79:
                return "75-79"
            else:
                return "80+"

        df["age_group"] = df["age_at_reg_waiting_list"].apply(assign_age_group)

        cum_haz = self.model.removal_model.predict_cumulative_hazard(df, times=times)
        # if more than one cum_haz is predicted at once, one has to keep in mind to match the index correctly,
        # since the order is changed when using a stratified CPH

        return(cum_haz)

    def simulate_time_from_uniform(self, U, cum_haz):
        """
        Simulate survival time T by solving H(T) =  - log(U)
        Solve H(T) = target for T, by going through the cum_haz list
        and returning the last index with a value below target.
        """
        target = -np.log(U)
        
        survival_time = np.searchsorted(cum_haz, target, side="right") - 1
        survival_time = np.clip(survival_time, 0, len(cum_haz) - 1)
        return survival_time
    
    def predict_survival_time(self):
        """
        Predicts a survival time until removal from the waiting list using the cumulative hazard function.

        If the patient is part of the initial population (i.e., present at simulation start),
        survival time is simulated conditional on having already survived up to their waiting time (left-truncation).
        For all other cases, a standard draw from the full survival distribution is used.

        Returns:
            int: Predicted survival time in days.
        """
        U = self.model.random.uniform(0, 1, 1)

        if self.model.schedule.time == 0:
            # simulate survival times conditional on survival up to t_trunc
            # for initial waiting list candidates
            t_trunc = int(self.waitinglist_time * 365)

            if t_trunc in self.cum_haz.index:
                H_trunc = self.cum_haz.loc[t_trunc].values[0]
            else:
                H_trunc = np.interp(t_trunc, self.cum_haz.index, self.cum_haz.values.flatten())

            U = U * np.exp(-H_trunc)

        cum_haz_matrix = self.cum_haz.to_numpy()
        survival_time = self.simulate_time_from_uniform(U[0], cum_haz_matrix[:,0])

        return(survival_time)
    
    def calc_surv_prob_in_next_step(self):
        waiting_list_time_days = 1 if self.waitinglist_time == 0 else self.waitinglist_time * 365
        t0 = waiting_list_time_days
        t1 = waiting_list_time_days + self.model.step_size * 365

        cum_haz_at_t0 = self.cum_haz.loc[t0].values if t0 in self.cum_haz.index else np.interp(t0, self.cum_haz.index, self.cum_haz.values.flatten())
        cum_haz_at_t1 = self.cum_haz.loc[t1].values if t1 in self.cum_haz.index else np.interp(t1, self.cum_haz.index, self.cum_haz.values.flatten())

        delta_hazard = cum_haz_at_t1 - cum_haz_at_t0
        survival_probability = np.exp(-delta_hazard)
        
        return(survival_probability)
    
    def remove_patient(self, removal_pred_type = "bender"):
        if removal_pred_type == "bender":
            # Complete survival times from the cumulative hazard distribution are simulated
            if  self.surv_time <= self.waitinglist_time * 365:
                self.status = "removed"
                self.step_removed_from_list = self.model.schedule.time
        if removal_pred_type == "delta_haz":
            # at each step the delta hazard is calculated and
            # compared to a uniform number to decide if removal occurs
            surv_prob = self.calc_surv_prob_in_next_step()
            rand = self.model.random.random()

            if  surv_prob <= rand:
                self.status = "removed"
                self.step_removed_from_list = self.model.schedule.time

    def step(self):
        if self.status in ["transplanted", "living_donor_transplanted", "removed"]:
            pass
        else:
            self.waitinglist_time += 1 * self.model.step_size
            self.dialysis_time += 1 * self.model.step_size
            self.age += 1 * self.model.step_size
            if self.model.et_groups_bool:
                if (self.age >= self.model.esp_age_border) and ((self.age - self.model.step_size) < self.model.esp_age_border):
                    self.et_program = self.model.assign_et_prgm(1, self.age)
            self.set_patient_random_nt()
            self.remove_patient()

class OrganDonor(Agent):
    """
   Class representing an organ donor within the simulation.

    Attributes:
        unique_id (int): Unique identifier.
        step_added_to_model (int): Step at which the agent was added to the model.
        status (str): Current status either "transplantable", "transplanted", or "removed"
        transplanted (bool): Indicates if the donor has undergone transplantation.
        organ_removed (bool): Indicates if the organ couldn't be allocated and was removed.
        age (int): Age of the donor, sampled from a Gaussian Kernel Density Estimation.
        blood_group (str): Blood group of the donor, sampled based on frequency distribution.
        hla_alleles (dict): Dict of HLA alleles representing the donor's HLA type.
        hla_n_homozygous_loci (int): number of homozygous loci. 3 = fully homozygous, 2 = two homozygous loci etc.
        region (str): DSO region. If None, it's resampled from frequency. Defaults to None.
        subregion (int): random choosen subregion, from the subregions in the DSO region.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.model: AllocationModel = model
        self.type = "organ_donor"
        self.step_added_to_model = self.model.schedule.time
        self.status = "transplantable"
        self.declined_offers = None

        self.age = self.sample_donor_age()

        self.blood_group = self.model.random.choice(list(self.model.donor_blood_grp_freq_dict.keys()),
                                              p=list(self.model.donor_blood_grp_freq_dict.values()))
        self.hla_alleles = self.model.gen_hla_haplotypes()
        self.unacceptable_antigens = None

        self.hla_n_homozygous_loci = self.model.calc_homozygous_loci(self.hla_alleles)

        self.region = self.model.random.choice(list(self.model.donor_reg_freq_dict.keys()),
                                              p=list(self.model.donor_reg_freq_dict.values()))
        
        self.subregion = self.assign_subregion()
        
    def sample_donor_age(self):
        """
        Resample donor ages, any negative values are resampled again.
        """
        age = self.model.kde_age_donors.resample(size = 1, seed=self.model.random)[0][0] #Assign age using KDE

        while age < 0:
                age = self.model.kde_age_donors.resample(size = 1, seed=self.model.random)[0][0] #Assign age using KDE

        return age

    def assign_subregion(self):
        """
        Random assignment to one of the subregions in the region.
        """
        if self.region == "baden_württemberg":
            return(self.model.random.choice(["Stuttgart", "Freiburg"]))
        if self.region == "bayern":
            return(self.model.random.choice(["Muenchen", "Erlangen"]))
        if self.region == "mitte":
            return(self.model.random.choice(["Mainz", "Homburg", "Marburg"]))
        if self.region == "nord":
            return(self.model.random.choice(["Hannover", "Hamburg"]))
        if self.region == "nord_ost":
            return(self.model.random.choice(["Berlin", "Rostock"]))
        if self.region == "nordrhein_westfalen":
            return(self.model.random.choice(["Duesseldorf", "Koeln-Bonn", "Muenster"]))
        if self.region == "ost":
            return("Leipzig")

    def step(self):
        pass


# the model ----------------------------------------------

class AllocationModel(Model):
    """
    Initializes the AllocationModel, a simulation model for allocating organ donors to patients.
    
    Args:
        df_init_patients (DataFrame): Initial patient data.
        n_pts_add_per_step (int/list): Number of patients added per step, can be a constant or a list for varying numbers.
        n_organ_donors_step (int/list): Number of organ donors added per step, can be a constant or a list for varying numbers.
        removal_model (model): Model used for determining removal from the waiting list.
        df_nt_status_prob (DataFrame): Dataframe containing probabilities for non-transplantable status.
        df_dial_to_reg_time (DataFrame): Dataframe containing age_at_reg_waiting_list, time_dial_to_registration, age_at_reg_bins.
                From this Dataframe the age dependent KDE for time from dialysis to registration are derived.
        pts_age_list (list): Ages of patients for KDE. Defaults to an empty list.
        donor_age_list (list): Ages of donors for KDE. Defaults to an empty list.
        vpra_df (DataFrame): First measured vPRA values and corresponding unacceptable antigens for resampling.
        donor_blood_grp_freq_dict (dict, optional): Frequency of blood groups. Defaults to specified values for A, AB, B, O groups.
        recipient_blood_grp_freq_dict (dict, optional): Frequency of blood groups. Defaults to specified values for A, AB, B, O groups.
        hla_a_freq_dict (dict): Frequency of HLA-A alleles. Defaults to an empty dict.
        hla_b_freq_dict (dict): Frequency of HLA-B alleles. Defaults to an empty dict.
        hla_dr_freq_dict (dict): Frequency of HLA-DR alleles. Defaults to an empty dict.
        hla_hapl_freq_df (DataFrame): DataFrame containing a hla_a_broad, hla_b_broad, hla_drb1_split, and normalized frequencies columns.
                From this DataFrame new haplotypes are generated.
        donor_reg_freq_dict (dict): Frequency of donor regions. Defaults to an empty dict.
        recipient_reg_freq_dict (dict): Frequency of recipient regions. Defaults to an empty dict.
        living_donor_prob_dict (dict): Probability dict for living donors. Defaults to an empty dict.
        et_groups_bool (bool, optional): If True, divides agents into ESP and ETKAS groups based on age. Defaults to True.
        esp_national_allo (bool, optional): Boolean indication if organs in the ESP program should be offered nationally
                if not accepted on regional level. This is not as in the ET allocation rules. Defaults to False.
        extended_allo (bool, optional): If True, organs in the ESP program not accepted on regional level are offered
                in the extended allocation program to all ETKAS patients. Defaults to True
        esp_age_border (int, optional): Define the age at which patients can go into the ESP program. Defaults to 65.
        prob_stay_in_etkas(double, optional): Probability for staying in ETKAS program after turning 65.
        max_rank_pts_organ_offer (int, optional): Maximum rank of patients considered for an organ.
        step_size (int, optional): The step size for the simulation. Defaults to 1.
        seed (int, optional): Seed for random number generation. Defaults to None.
        datacollector_interval (int, optional): Interval for collecting data. If None, collects data every step.
        quiet(bool, optional): If quiet is true, no results are printed for each step. Default is False.
        start_year(int, optional): Starting year of the simulation. This is used to match the simulation time to the living donor probability of the time period.
    
    This class simulates the allocation of organs to patients in a healthcare system. It includes functionality for adding new
    patients and organ donors over time, determining the removal of patients from the waiting list, and allocating organs based on 
    various criteria, including blood group compatibility and regional preferences. The simulation uses KDE for resampling patient
    and donor ages, normalizes frequency dictionaries for blood groups and HLA alleles, and includes a mechanism for data collection
    to analyze the simulation's progress and outcomes.
    """

    def __init__(self, df_init_patients,
                 n_pts_add_per_step,
                 n_organ_donors_step,
                 removal_model,
                 df_nt_status_prob,
                 dict_log_odds_ETKAS,
                 dict_log_odds_ESP,
                 df_dial_to_reg_time,
                 pts_age_list,
                 donor_age_list,
                 vpra_df,
                 hla_a_freq_dict,
                 hla_b_freq_dict,
                 hla_dr_freq_dict,
                 hla_hapl_freq_df,
                 donor_reg_freq_dict,
                 recipient_reg_freq_dict,
                 living_donor_prob_dict,
                 donor_blood_grp_freq_dict = {"A": 0.440, "AB": 0.0486, "B": 0.113,"O": 0.398},
                 recipient_blood_grp_freq_dict = {"A": 0.426, "AB": 0.0565, "B": 0.127,"O": 0.390},
                 et_groups_bool = False,
                 esp_national_allo = False,
                 extended_allo = True,
                 esp_age_border = 65,
                 prob_stay_in_etkas = 0.36,
                 max_rank_pts_organ_offer = 100,
                 step_size = 1.0,
                 seed = None,
                 datacollector_interval = None,
                 quiet = False,
                 start_year = 2006):

        super().__init__()

        self.seed = seed 
        self.quiet = quiet
        self.random = np.random.default_rng(seed) #create a random number generator

        self.df_init_patients = df_init_patients
        
        # n_pts_add_per_step can be a scalar for a different value of patients per cycle
        # or a single value, which gets repeated for each step
        if np.isscalar(n_pts_add_per_step):
            self.n_pts_add_per_step = itertools.cycle([n_pts_add_per_step])
        else:    
            self.n_pts_add_per_step = itertools.cycle(n_pts_add_per_step)

        # n_organ_donors_step
        if np.isscalar(n_organ_donors_step):
            self.n_organ_donors_step = itertools.cycle([n_organ_donors_step])
        else:
            self.n_organ_donors_step = itertools.cycle(n_organ_donors_step)

        self.schedule = RandomActivation(self)  # Create a random activation schedule

        self.removal_model = removal_model #model for waiting list removal
        self.df_nt_status_prob = df_nt_status_prob #probability for non_transplantable

        self.dict_log_odds_ETKAS = dict_log_odds_ETKAS
        self.dict_log_odds_ESP = dict_log_odds_ESP

        self.kde_age_pts = gaussian_kde(pts_age_list)
        self.kde_age_donors = gaussian_kde(donor_age_list)

        self.dict_kde_time_dial_to_reg = self.make_kde_dict_dial_to_reg(df_dial_to_reg_time)

        self.vpra_df = vpra_df

        self.vpra_proportions = self.calculate_vpra_proportions(self.vpra_df)

        #the frequencies have to be normalized
        # otherwise np.random.choice throws an error ('probabilities do not sum to 1')
        self.donor_blood_grp_freq_dict = normalize_dict_values(donor_blood_grp_freq_dict)
        self.recipient_blood_grp_freq_dict = normalize_dict_values(recipient_blood_grp_freq_dict)

        self.hla_a_freq_dict = normalize_dict_values(hla_a_freq_dict)
        self.hla_b_freq_dict = normalize_dict_values(hla_b_freq_dict)
        self.hla_dr_freq_dict = normalize_dict_values(hla_dr_freq_dict)
        self.hla_hapl_freq_df = hla_hapl_freq_df

        self.donor_reg_freq_dict = normalize_dict_values(donor_reg_freq_dict)
        self.recipient_reg_freq_dict = normalize_dict_values(recipient_reg_freq_dict)

        self.living_donor_prob_dict = living_donor_prob_dict

        self.max_rank_pts_organ_offer = max_rank_pts_organ_offer

        self.et_groups_bool = et_groups_bool
        self.esp_national_allo = esp_national_allo
        self.extended_allo = extended_allo
        self.esp_age_border = esp_age_border
        self.prob_stay_in_etkas = prob_stay_in_etkas

        self.step_size = step_size
        self.start_year = start_year
        
        # Initialize patients in the constructor
        self.add_init_patients_with_data(df_init_patients)

        self.datacollector_interval = datacollector_interval

        # Create a data collector to collect waiting times at each step
        self.datacollector = DataCollector(
            agent_reporters={
                "agent_type": "type",
                "age": "age",
                "blood_group": "blood_group",
                "waitinglist_time": "waitinglist_time",
                "dialysis_time": "dialysis_time",
                "status": "status",
                "step_added_to_model": "step_added_to_model",
                "step_transplanted": "step_transplanted",
                "linked_donor": "linked_donor",
                "step_removed_from_list": "step_removed_from_list",
                "et_program": "et_program",
                "hla": lambda a: ', '.join(v for v in a.hla_alleles.values() if pd.notnull(v)) if a.hla_alleles else np.nan,
                "vpra": "vpra",
                "unacceptable_antigens": lambda a: ', '.join(v for v in a.unacceptable_antigens if pd.notnull(v)) if isinstance(a.unacceptable_antigens, list) else np.nan,
                "mmp": "mmp",
                "hla_mm": "hla_mm",
                "region": "region",
                "declined_offers": "declined_offers"
            },
            model_reporters={
                "pts_on_waiting_list": get_n_pts_waiting,
                "pts_transplantable": get_n_pts_waiting_transplantable,
                "pts_transplanted": get_n_pts_transplanted,
                "pts_removed": get_n_pts_removed,
                "living_donor_transplanted": get_n_pts_living_donor_transplanted
            }
        )
    
    def add_init_patients_with_data(self, patient_data):
        """
        Add initial patients with given data.

        Args:
            patient_data (DataFrame): DataFrame containing patient data.
        """

        for index, row in patient_data.iterrows():
            age = row["age"]
            waitinglist_time = row["waitinglist_time"]
            dialysis_time = row["dialysis_time"]
            blood_group = row["blood_group"]
            hla_alleles = {
                "a_split_1": row["hla_a_split_1"],
                "a_split_2": row["hla_a_split_2"],
                "a_1": row["hla_a_broad_1"],
                "a_2": row["hla_a_broad_2"],
                "b_split_1": row["hla_b_split_1"],
                "b_split_2": row["hla_b_split_2"],
                "b_1": row["hla_b_broad_1"],
                "b_2": row["hla_b_broad_2"],
                "c_split_1": row["hla_c_split_1"],
                "c_split_2": row["hla_c_split_2"],
                "c_1": row["hla_c_broad_1"],
                "c_2": row["hla_c_broad_2"],
                "dr_1": row["hla_drb1_split_1"],
                "dr_2": row["hla_drb1_split_2"]
            }
            status = row["status"]
            vpra = row["latest_vPRA"]
            unacceptable_antigens = row["latest_unacceptable_antigens"]
            if pd.notnull(unacceptable_antigens):
                unacceptable_antigens = unacceptable_antigens.split(" ")

            # Ensure the waiting time is not greater than the age
            waitinglist_time = min(waitinglist_time, age)
            dialysis_time = max(dialysis_time, waitinglist_time)
            dialysis_time = min(dialysis_time, age)
            patient = Patient(index,
                              self,
                              age = age,
                              waitinglist_time = waitinglist_time,
                              dialysis_time = dialysis_time,
                              blood_group = blood_group,
                              status= status,
                              hla_alleles = hla_alleles,
                              vpra = vpra,
                              unacceptable_antigens = unacceptable_antigens)
            self.schedule.add(patient)

    def get_agent_count_by_type(self, agent_type):
        return sum(1 for agent in self.get_agents_of_type(agent_type))

    def add_patient(self, num_to_add):
        # It is important to keep in mind, that changes in the code for bulk creation have also
        # to be made in the agent class, where only a single value is generated/resampled per variable.

        st_model = start_timer()

        agent_count = self.get_agent_count_by_type(Patient)  # Get the current count of agents in the model

        # waiting time is set to a value between 0 and step_size
        rand_waitinglist_time_list = self.sample_waiting_times(num_to_add)

        # Resample ages in bulk
        add_pts_age_list = self.sample_ages(num_to_add)
        
        # Resample dialysis times in bulk
        dialysis_times = self.sample_dialysis_times(add_pts_age_list, rand_waitinglist_time_list)

        # Resample blood groups in bulk
        add_pts_blood_group_list = self.sample_blood_groups(num_to_add)
        
        # Sample living donor transplantation status
        add_pts_status_list = self.sample_living_donor_status(num_to_add, add_pts_age_list, rand_waitinglist_time_list)

        # Sample HLA alleles in bulk
        hla_alleles_list = self.gen_hla_haplotypes(num_to_add)

        end_timer(st_model, "     Sampling attributes in Bulk: ", self.quiet)

        st_vpra = start_timer()
        vpra_list, unacceptable_antigens_list = self.assign_vpra_unaccept_antigen_bulk(hla_alleles_list=hla_alleles_list)
        end_timer(st_vpra, "     Sampling vpra attributes in Bulk", self.quiet)

        # Sample ET program in bulk
        et_program_list = self.assign_et_prgm(num_to_add, add_pts_age_list)


        st_model = start_timer()
        patients_to_add = [
            Patient(i + agent_count,
            self,
            age=add_pts_age_list[i],
            waitinglist_time=rand_waitinglist_time_list[i],
            dialysis_time=dialysis_times[i],
            blood_group=add_pts_blood_group_list[i],
            status = add_pts_status_list[i],
            hla_alleles = hla_alleles_list[i],
            et_program = et_program_list[i],
            vpra = vpra_list[i],
            unacceptable_antigens= unacceptable_antigens_list[i]
            )
            for i in range(num_to_add)
            ]
        
        for pt in patients_to_add:
            self.schedule.add(pt)  # Add the agent to the schedule

        end_timer(st_model, "     Add patients loop: ", self.quiet)

    def sample_waiting_times(self, num_to_add):
        """
        Sample waiting times for a given number of patients.
        """
        return self.random.random(size=num_to_add) * self.step_size
    
    def sample_ages(self, num_to_add):
        """
        Resample patient ages, any negative values are resampled again.
        """
        ages = self.kde_age_pts.resample(size=num_to_add, seed=self.random)[0]

        while len(ages[ages < 0]) > 0:
                ages[ages < 0] = self.kde_age_pts.resample(size=len(ages[ages < 0]), seed=self.random)[0]

        return ages

    def sample_dialysis_times(self, ages, waiting_times):
        """
        Sample dialysis times based on age bins and waiting times.

        If a negative value is resampled for time from dialysis to registration, the time from dialysis to registration is set to zero
        """
        # Check if ages and waiting_times are single values (scalars) or arrays
        is_scalar = np.isscalar(ages)

        if is_scalar:
            ages = np.array([ages])
            waiting_times = np.array([waiting_times])
        
        age_bins = np.minimum(np.floor(ages // 10) * 10, 80)  # Age bins floored and capped at 80
        time_dial_to_reg_list = np.zeros(len(ages))

        # Resample dialysis registration times based on age bins
        for age_bin in np.unique(age_bins):
            mask = age_bins == age_bin
            time_dial_to_reg_list[mask] = self.dict_kde_time_dial_to_reg[age_bin].resample(size=np.sum(mask), seed=self.random)[0]

        # Ensure no negative values for time_dial_to_reg
        time_dial_to_reg_list[time_dial_to_reg_list < 0] = 0

        dialysis_times =  np.where(
            time_dial_to_reg_list / 365 + waiting_times > ages,
            waiting_times,
            time_dial_to_reg_list / 365 + waiting_times
        )
        
        if is_scalar:
          return dialysis_times[0]
    
        return dialysis_times

    def sample_blood_groups(self, num_to_add):
        """
        Sample blood groups for the patients in bulk.
        """
        return self.random.choice(
            list(self.recipient_blood_grp_freq_dict.keys()),
            size=num_to_add,
            p=list(self.recipient_blood_grp_freq_dict.values())
        )

    def sample_living_donor_status(self, num_to_add, ages, waiting_times):
        """
        Bulk assign living donor transplantation status based on probabilities and conditions.
        """
        def get_living_donor_prob(age, year, prob_lookup):
            age_bin = ((age // 5) + 1) * 5
            for (a_bin, start, end), prob in prob_lookup.items():
                if a_bin == age_bin and start < year <= end:
                    return prob
            return 0  # fallback if no match

        is_scalar = np.isscalar(ages)

        if is_scalar:
            ages = np.array([ages])
            waiting_times = np.array([waiting_times])
        
        add_pts_status_list = np.full(num_to_add, "transplantable", dtype="<U25")  # Default to 'transplantable'

        if self.schedule.steps == 0:
            # Living donor transplantations are only estimated for patients less than one year on the waiting list
            # at model initialization
            short_waiting_mask = waiting_times * 365 < 365 * 1

            current_year = 2005
            if np.any(short_waiting_mask):
                living_donor_probs = np.array([
                    get_living_donor_prob(age, current_year, self.living_donor_prob_dict)
                    for age in ages
                    ])
                random_probs = self.random.random(size=len(add_pts_status_list))
                living_donor_mask = random_probs <= living_donor_probs
                add_pts_status_list[short_waiting_mask] = np.where(
                    living_donor_mask, "living_donor_transplanted", "transplantable"
                    )

        else:
            current_year = self.start_year + self.schedule.steps * self.step_size

            living_donor_probs = np.array([
                get_living_donor_prob(age, current_year, self.living_donor_prob_dict)
                for age in ages
                ])

            # Assign 'living_donor_transplanted' or 'transplantable' based on random probabilities
            random_probs = self.random.random(size=num_to_add)
            living_donor_mask = random_probs <= living_donor_probs
            add_pts_status_list[living_donor_mask] = "living_donor_transplanted"

        if is_scalar:
          return add_pts_status_list[0]
    
        return add_pts_status_list
    
    def assign_et_prgm(self, num_to_add, ages):
        is_scalar = np.isscalar(ages)

        if is_scalar:
            ages = np.array([ages])
        
        et_program_list = np.full(num_to_add, np.nan, dtype=object)
        random_probs = self.random.random(size=num_to_add)

        et_program_mask = (ages >= self.esp_age_border) & (random_probs >= self.prob_stay_in_etkas)
        et_program_list[et_program_mask] = "esp" # assign esp, when mask condition is met
        et_program_list[~et_program_mask] = "etkas" # assign etkas, when mask condition is not met
        
        if is_scalar:
          return et_program_list[0]

        return et_program_list

    def add_organ_donor(self, num_to_add):
        agent_count = self.get_agent_count_by_type(OrganDonor)

        for i in range(num_to_add):
            o = OrganDonor(i + agent_count, self)
            self.schedule.add(o)

    def make_kde_dict_dial_to_reg(self, df_time_dial_to_reg):
        dict_kde_dial_to_reg = {}
        for age_bin in df_time_dial_to_reg.age_at_reg_bins:
            kde = gaussian_kde(df_time_dial_to_reg.loc[df_time_dial_to_reg["age_at_reg_bins"] == age_bin, "time_dial_to_registration"])
            dict_kde_dial_to_reg.setdefault(age_bin, kde)
        return  dict_kde_dial_to_reg

    def gen_hla_haplotypes(self, num_to_add=1):
        """
        Generates multiple random HLA haplotypes based on a given probability distribution.

        Args:
            random_gen: Random number generator.
            num_to_add: The number of HLA haplotypes to generate (default is 1).

        Returns:
            list of dict: List of dictionaries containing HLA haplotypes.
        """

        # Generate all indices for haplotype1 and haplotype2 at once using random choice
        haplotype1_indices = self.random.choice(
            self.hla_hapl_freq_df.index, size=num_to_add, p=self.hla_hapl_freq_df["normalized_freq"]
        )
        haplotype2_indices = self.random.choice(
            self.hla_hapl_freq_df.index, size=num_to_add, p=self.hla_hapl_freq_df["normalized_freq"]
        )

        # Fetch all haplotypes at once using iloc, this avoids looping to select one by one
        haplotype1_df = self.hla_hapl_freq_df.iloc[haplotype1_indices]
        haplotype2_df = self.hla_hapl_freq_df.iloc[haplotype2_indices]

        hla_alleles_list = [None] * num_to_add

        # Loop over the number of haplotypes to generate and construct the dictionaries
        for i in range(num_to_add):
            # Construct the HLA alleles dictionary for the i-th haplotype
            hla_alleles_list[i] = {
                "a_split_1": haplotype1_df["hla_a_split"].values[i],
                "a_split_2": haplotype2_df["hla_a_split"].values[i],
                "a_1": haplotype1_df["hla_a_broad"].values[i],
                "a_2": haplotype2_df["hla_a_broad"].values[i],
                "b_split_1": haplotype1_df["hla_b_split"].values[i],
                "b_split_2": haplotype2_df["hla_b_split"].values[i],
                "b_1": haplotype1_df["hla_b_broad"].values[i],
                "b_2": haplotype2_df["hla_b_broad"].values[i],
                "c_split_1": haplotype1_df["hla_c_split"].values[i],
                "c_split_2": haplotype2_df["hla_c_split"].values[i],
                "c_1": haplotype1_df["hla_c_broad"].values[i],
                "c_2": haplotype2_df["hla_c_broad"].values[i],
                "dr_1": haplotype1_df["hla_drb1_split"].values[i],
                "dr_2": haplotype2_df["hla_drb1_split"].values[i]
        }

        if num_to_add == 1:
          return hla_alleles_list[0]
    
        return hla_alleles_list
    
    def calculate_vpra_proportions(self, vpra_df):
        total = vpra_df["frequency"].sum()

        vpra_proportions = {
            "0": vpra_df.loc[vpra_df["vpra"] == 0, "frequency"].sum() / total,
            "0to49": vpra_df.loc[(vpra_df['vpra'] > 0) & (vpra_df['vpra'] < 50), "frequency"].sum() / total,
            "50to84": vpra_df.loc[(vpra_df['vpra'] >= 50) & (vpra_df['vpra'] < 85), "frequency"].sum() / total,
            "85+": vpra_df.loc[vpra_df["vpra"] >= 85, "frequency"].sum() / total
        }

        return vpra_proportions

    def assign_vpra_unaccept_antigen_bulk(self, hla_alleles_list):
        """
        Assign vPRA values and corresponding unacceptable antigens (UA) in bulk 
        for multiple patients, using a hybrid stratified and weighted sampling approach.

        This function:
        - Stratifies the sampling space into vPRA groups ('0', '0to49', '50to84', '85+'),
          preserving target proportions defined in `self.vpra_proportions`.
        - Within each stratum, samples UA profiles according to empirical frequencies,
          adjusted to exclude any profiles that conflict with the patient's own HLA alleles.
        - Falls back to alternative groups if no valid profiles exist in the initially chosen group.

        Parameters
        ----------
        hla_alleles_list : list of dict or dict
            Either a list of dictionaries (each representing a patient's HLA alleles), 
            or a single dictionary for a single patient.

        Returns
        -------
        tuple
            A tuple of two elements:
            - vpra_values: numpy array or scalar
                The assigned vPRA values for each patient. If a single patient was provided,
                returns a scalar value.
            - unacceptable_antigens_values: list or list of lists
                The assigned unacceptable antigens corresponding to each patient. 
                If a single patient was provided, returns a single list.

        Raises
        ------
        ValueError
            If after exhausting all vPRA strata, no compatible UA profile could be 
            found for a given patient's HLA.
        """
        if isinstance(hla_alleles_list, dict):
            single_patient_mode = True
            hla_alleles_list = [hla_alleles_list]
        else:
            single_patient_mode = False

        hla_alleles_sets = [set(hla_alleles.values()) for hla_alleles in hla_alleles_list]

        strata = {
            "0": self.vpra_df[self.vpra_df["vpra"] == 0].copy(),
            "0to49": self.vpra_df[(self.vpra_df["vpra"] > 0) & (self.vpra_df["vpra"] < 50)].copy(),
            "50to84": self.vpra_df[(self.vpra_df["vpra"] >= 50) & (self.vpra_df["vpra"] < 85)].copy(),
            "85+": self.vpra_df[self.vpra_df["vpra"] >= 85].copy()
        }

        def get_random_group(exclude=None):
            exclude = exclude or []

            # Filter out excluded groups
            filtered_items = [(g, p) for g, p in self.vpra_proportions.items() if g not in exclude]

            if not filtered_items:
                raise ValueError("No groups left to sample after applying exclusions.")

            groups, probs = zip(*filtered_items)
            probs = np.array(probs)
            probs /= probs.sum()  # Normalize

            return self.random.choice(groups, p=probs)

        vpra_values = [None] * len(hla_alleles_list)
        unacceptable_antigens_values = [None] * len(hla_alleles_list)

        for i, hla_set in enumerate(hla_alleles_sets):
            attempted_groups = []
            while True:
                group = get_random_group(exclude=attempted_groups)
                df_stratum = strata[group]

                weights = df_stratum["frequency"].to_numpy()

                mask_conflict = df_stratum["unacceptable_antigens"].apply(
                    lambda ua_list: isinstance(ua_list, (list, set, tuple)) and any(ua in hla_set for ua in ua_list)
                )
                adjusted_weights = np.where(mask_conflict, 0, weights)
                total_weight = adjusted_weights.sum()

                if total_weight > 0:
                    break  # Found a valid group
                attempted_groups.append(group)

                if len(attempted_groups) == len(self.vpra_proportions):
                    raise ValueError(f"No compatible strata found for patient {i} after trying all groups.")

            adjusted_probs = adjusted_weights / total_weight
            idx = self.random.choice(len(df_stratum), p=adjusted_probs)
            row = df_stratum.iloc[idx]

            vpra_values[i] = row["vpra"]
            unacceptable_antigens_values[i] = row["unacceptable_antigens"]

        vpra_values = np.array(vpra_values) if not single_patient_mode else vpra_values[0]
        unacceptable_antigens_values = unacceptable_antigens_values if not single_patient_mode else unacceptable_antigens_values[0]

        return vpra_values, unacceptable_antigens_values


    def calc_homozygous_loci(self, hla_dict):
        """
        Calculate the number of homozygous loci in an HLA dictionary.
    
        Args:
            hla_dict (dict): Dictionary containing HLA haplotypes.
    
        Returns:
            int: Number of homozygous loci.
        """
        hla_a = [hla_dict.get("a_1")[1:], hla_dict.get("a_2")[1:]]
        hla_b = [hla_dict.get("b_1")[1:], hla_dict.get("b_2")[1:]]
        hla_dr = [hla_dict.get("dr_1")[2:], hla_dict.get("dr_2")[2:]]
        n_homozygous_loci = 0

        if len(set(hla_a)) == 1:
             n_homozygous_loci += 1
        if len(set(hla_b)) == 1:
             n_homozygous_loci += 1
        if len(set(hla_dr)) == 1:
            n_homozygous_loci += 1

        return n_homozygous_loci

    def mmHLA_loci(self, d_hla, p_hla):
        """
        Calculation of HLA mismatches for one loci
        Mismatches are defined as donor antigens that are different from the patients antigens.
        In case of 1 HLA-antigen the donor or patient is assumed to be homozygous.
        In case of 1 HLA-antigen only 1 mismatch is calculated.
        """
        d_hla_set, p_hla_set = set(d_hla), set(p_hla)

        if d_hla_set == p_hla_set:
            return 0

        if len(d_hla_set) == 1:
            if d_hla_set.issubset(p_hla_set):
                return 0
            else:
                return 1

        if len(d_hla) == 2:
            if d_hla[0] in p_hla and d_hla[1] not in p_hla:
                return 1
            elif d_hla[0] not in p_hla and d_hla[1] in p_hla:
                return 1
            else:
                return 2
    
    def mmHLA(self, d_hla_dict, p_hla_dict, return_allele = "all"):
        """
        Calculation of HLA mismatches
        Args:
            d_hla_dict: dictionary with the hla haplotypes
            p_hla_dict: dictionary with the hla haplotypes
        """
        d_hla_a = [d_hla_dict.get("a_1")[1:], d_hla_dict.get("a_2")[1:]]
        d_hla_b = [d_hla_dict.get("b_1")[1:], d_hla_dict.get("b_2")[1:]]
        d_hla_dr = [d_hla_dict.get("dr_1")[2:], d_hla_dict.get("dr_2")[2:]]
    
        p_hla_a = [p_hla_dict.get("a_1")[1:], p_hla_dict.get("a_2")[1:]]
        p_hla_b = [p_hla_dict.get("b_1")[1:], p_hla_dict.get("b_2")[1:]]
        p_hla_dr = [p_hla_dict.get("dr_1")[2:], p_hla_dict.get("dr_2")[2:]]
    
    
        mm_hla_a = self.mmHLA_loci(d_hla_a, p_hla_a)
        mm_hla_b = self.mmHLA_loci(d_hla_b, p_hla_b)
        mm_hla_dr = self.mmHLA_loci(d_hla_dr, p_hla_dr)

        if return_allele == "all":
            return mm_hla_a + mm_hla_b + mm_hla_dr
        elif return_allele == "a":
            return mm_hla_a
        elif return_allele == "b":
            return mm_hla_b
        elif return_allele == "dr":
            return mm_hla_dr
        
    def calc_points(self, agent, donor):
        """
        Calculate points according to ET manual chapter 4.
        """
        pts = 0
        
        pts += agent.dialysis_time * 33.3 #per year 33.3 points are added
        pts += agent.mmp

        hla_mismatches = self.mmHLA(donor.hla_alleles, agent.hla_alleles)

        if agent.age <= 18:
            pts += 100 + (400 * (1 - hla_mismatches / 6)) * 2
        else:
            pts += 400 * (1 - hla_mismatches / 6)

        if agent.region == donor.region: #for a regional match 200 points are given
            pts += 200
        
        pts += 100 #for a national match 100 points (in this model all patients)

        return pts
    
    def allocate(self):
        """
        Allocate organs to patients.
        """

        organ_donors = [agent for agent in self.schedule.agents if isinstance(agent, OrganDonor) and (agent.status == "transplantable")]

        def filter_agents_by_unacceptable_antigens(agent_list, donor_hla):
            """
            Filters a list of agents by checking whether their unacceptable antigens
            are compatible with the given donor HLA values.

            An agent is included in the returned list if:
              - They have no unacceptable antigens (represented as a scalar np.nan), or
              - None of their listed unacceptable antigens are found in the donor's HLA values.

            Parameters:
            ----------
            agent_list : list
                A list of agent objects. Each agent is expected to have an attribute
                `unacceptable_antigens` which is either a list of antigen strings or a single np.nan.

            donor_hla : iterable
            A collection of donor HLA antigen values (e.g., list or set of strings) to check against.

            Returns:
            -------
            list
                A filtered list of agents who are compatible with the donor's HLA values.
            """
            filtered_list = [
                agent for agent in agent_list
                if ( #handle cases without unacceptable antigens (these are single np.nan values)
                    (not isinstance(agent.unacceptable_antigens, list) and
                     pd.isna(agent.unacceptable_antigens))
                     or
                     ( #check if any unacceptable antigens are in donor_hla values
                         isinstance(agent.unacceptable_antigens, list) and
                         not any(antigen in donor_hla for antigen in agent.unacceptable_antigens)
                         )
                    )
                ]
            return filtered_list

        def organ_acceptance(donor, patient, et_group):
            """
            Checks if an organ is accepted by the patient.
            This is done by calculating the acceptance probability based on the coefficients of a piecewise logistic regression model.
            """
            if et_group == "etkas":
                dict_values = {
                "intercept": 1,
                "vpra_percent": patient.vpra,
                "vpra_percentover_25": max(patient.vpra-25,0),
                "vpra_percentover_50": max(patient.vpra-50,0),
                "vpra_percentover_85": max(patient.vpra-85,0),
                "r_match_age": patient.age,
                "r_match_ageunder_25": max(25-patient.age, 0),
                "r_match_ageover_50": max(patient.age-50, 0),
                "r_match_ageover_65": max(patient.age-65, 0),
                "donor_age": donor.age,
                "donor_ageunder_20": max(20-donor.age, 0),
                "donor_ageover_65": max(donor.age-65, 0),
                "zero_mismatch1": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles) == 0),
                "mmb_hla_a1": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "a") == 1),
                "mmb_hla_a2": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "a") == 2),
                "mmb_hla_b1": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "b") == 1),
                "mmb_hla_b2": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "b") == 2),
                "mms_hla_dr1": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "dr") == 1),
                "mms_hla_dr2": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "dr") == 2),
                "years_on_dial": patient.dialysis_time
                }
                dict_log_odds = self.dict_log_odds_ETKAS

            if et_group == "esp":
                dict_values = {
                "intercept": 1,
                "vpra_percent": patient.vpra,
                "vpra_percentover_25": max(patient.vpra-25,0),
                "vpra_percentover_50": max(patient.vpra-50,0),
                "vpra_percentover_85": max(patient.vpra-85,0),
                "r_match_age": patient.age,
                "r_match_ageunder_70": max(70-patient.age, 0),
                "r_match_ageover_75": max(patient.age-75, 0),
                "donor_age": donor.age,
                "donor_ageunder_70": max(70-donor.age, 0),
                "donor_ageover_75": max(donor.age-75, 0),
                "zero_mismatch": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles) == 0),
                "mmb_hla_a1": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "a") == 1),
                "mmb_hla_a2": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "a") == 2),
                "mmb_hla_b1": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "b") == 1),
                "mmb_hla_b2": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "b") == 2),
                "mms_hla_dr1": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "dr") == 1),
                "mms_hla_dr2": int(self.mmHLA(donor.hla_alleles, patient.hla_alleles, "dr") == 2),
                "years_on_dial": patient.dialysis_time
                }
                dict_log_odds = self.dict_log_odds_ESP

            if set(dict_log_odds.keys()) != set(dict_values.keys()):
                raise ValueError("The keys in the coefficients and values dictionaries do not match.")

            # Extract keys to maintain the correct order
            keys = dict_log_odds.keys()

            # Convert to arrays for dot product calculation, maintaining the correct order
            coefficients_array = np.array([dict_log_odds[key] for key in keys])
            values_array = np.array([dict_values[key] for key in keys])

            # Calculate the dot product (predicted log odds)
            predicted_log_odds = np.dot(coefficients_array, values_array)

            # Calculate the probability
            prob = (1 / (1 + np.exp(-predicted_log_odds)))

            rand_num = self.random.random()

            #print(f"prob: {prob}, random number: {rand_num}")
            if prob >= rand_num:
                return True
            else:
                return False



        def match_donor_to_ptnt(donor, patients, patient_group, et_group = "etkas"):
            if not patients:
                return False  # No patients to match with
            
            # if no dictionary of log odds is defined,
            # the allocation will be done by selecting randomly
            # one of the top ranked patients,
            # up to the by max_rank_pts_organ_offer defined rank
            if self.dict_log_odds_ETKAS == None and et_group == "etkas":
                if len(patients) > self.max_rank_pts_organ_offer:
                    matched_patient = self.random.choice(patients[0:self.max_rank_pts_organ_offer])
                else:
                    matched_patient = self.random.choice(patients)

            elif self.dict_log_odds_ESP == None and et_group == "esp":
                if len(patients) > self.max_rank_pts_organ_offer:
                    matched_patient = self.random.choice(patients[0:self.max_rank_pts_organ_offer])
                else:
                    matched_patient = self.random.choice(patients)
            
            else:
                # acceptance is determined by a piecewise logistic regression model
                matched_patient = None
                if len(patients) > self.max_rank_pts_organ_offer: # defines up to which patient rank organ offers are made
                    i = 0
                    #print(f"patient_number: {i}")
                    for patient in patients[0:self.max_rank_pts_organ_offer]:
                        if organ_acceptance(donor, patient, et_group):
                            matched_patient = patient
                            break
                        i += 1  
                        self.declined_offers_counter += 1  
                else:
                    i = 0
                    for patient in patients:
                        #print(f"patient_number: {i}")
                        if organ_acceptance(donor, patient, et_group):
                            matched_patient = patient
                            break
                        i += 1
                        self.declined_offers_counter += 1
                if matched_patient == None:
                    return False
            

            #matched_pairs.append((matched_patient, donor))
            patient_group.remove(matched_patient)
            donor.declined_offers = self.declined_offers_counter
            donor.status = "transplanted"
            matched_patient.status = "transplanted"
            matched_patient.step_transplanted = self.schedule.time
            matched_patient.linked_donor = donor.unique_id
            matched_patient.hla_mm = self.mmHLA(donor.hla_alleles, matched_patient.hla_alleles)
            return True

        if self.et_groups_bool:

            # Get all non-transplanted patients
            ptnts_bld_grp_prgrm = {
            "A": {"esp": [], "etkas": []},
            "AB": {"esp": [], "etkas": []},
            "B": {"esp": [], "etkas": []},
            "O": {"esp": [], "etkas": []}
            }

            count_transplantable_patients = 0
            
            for agent in self.schedule.agents:
                if agent.type == "patient" and agent.status == "transplantable":
                    ptnts_bld_grp_prgrm[agent.blood_group][agent.et_program].append(agent)
                    count_transplantable_patients += 1
            if not self.quiet:
                print(f"     Organ donors: {len(organ_donors)} transplantable patients: {count_transplantable_patients}")

            if len(organ_donors) > 0:
                
                for donor in organ_donors:
                    donor_matched = False
                    self.declined_offers_counter = 0

                    donor_hla_values = set(donor.hla_alleles.values())
                    
                    if donor.age >= self.esp_age_border:
                        esp_patients = [agent for agent in ptnts_bld_grp_prgrm[donor.blood_group]["esp"]]

                        # Filter ESP patients by unacceptable antigens
                        esp_patients = filter_agents_by_unacceptable_antigens(esp_patients, donor_hla_values)

                        esp_region_patients = [agent for agent in esp_patients if agent.region == donor.region]
                        
                        # organs are first matched to patients from the same subregion
                        esp_subregion_patients = [agent for agent in esp_region_patients if agent.subregion == donor.subregion]
                        
                        if esp_subregion_patients: #only proceed if there are esp_patients
                            esp_subregion_patients.sort(key=lambda x: x.dialysis_time, reverse=True)
                            donor_matched = match_donor_to_ptnt(donor, esp_subregion_patients, ptnts_bld_grp_prgrm[donor.blood_group]["esp"], "esp")
                            if donor_matched:
                                    continue #if matched continue with next organ
                            
                        # Consider region-level patients but exclude those from the subregion (already considered)
                        esp_region_patients_no_subregion = [agent for agent in esp_region_patients if agent.subregion != donor.subregion]
                        if esp_region_patients_no_subregion: #only proceed if there are esp_patients
                            esp_region_patients_no_subregion.sort(key=lambda x: x.dialysis_time, reverse=True)
                            donor_matched = match_donor_to_ptnt(donor, esp_region_patients_no_subregion, ptnts_bld_grp_prgrm[donor.blood_group]["esp"], "esp")
                            if donor_matched:
                                    continue #if matched continue with next organ
                                
                        # This is not the same as it would be by Eurotransplant.
                        # consider all ESP patients but not those already in the region or subregion
                        if self.esp_national_allo:
                            esp_patients_no_region_or_subregion = [agent for agent in esp_patients if agent not in esp_subregion_patients and agent not in esp_region_patients]
                            
                            if esp_patients_no_region_or_subregion: #only proceed if there are esp_patients
                                esp_patients_no_region_or_subregion.sort(key=lambda x: x.dialysis_time, reverse=True)
                                donor_matched = match_donor_to_ptnt(donor, esp_patients_no_region_or_subregion, ptnts_bld_grp_prgrm[donor.blood_group]["esp"], "esp")
                                if donor_matched:
                                        continue #if matched continue with next organ
                        
                        # This is an approach to simulate the extended allocation for ESP organs, that aren't allocated regionally
                        # The organ is offered to the national ETKAS patients, if it was not accepted by the ESP patients in the region.
                        # The ETKAS patients are sorted based on the ETKAS point system.
                        # Organ acceptance is simulated by the "normal" ETKAS acceptance Odds Ratios.        
                        if self.extended_allo:
                            etkas_patients = [agent for agent in ptnts_bld_grp_prgrm[donor.blood_group]["etkas"]]
                            
                            # Filter ETKAS patients by unacceptable antigens
                            etkas_patients = filter_agents_by_unacceptable_antigens(etkas_patients, donor_hla_values)
                            
                            extended_allo_patients = etkas_patients

                            if extended_allo_patients: #only proceed if there are esp_patients
                                extended_allo_patients.sort(key=lambda patient: self.calc_points(patient, donor), reverse=True)
                                donor_matched = match_donor_to_ptnt(donor, extended_allo_patients, ptnts_bld_grp_prgrm[donor.blood_group]["etkas"], "etkas")
                                if donor_matched:
                                        continue #if matched continue with next organ


                    else:       
                        etkas_patients = [agent for agent in ptnts_bld_grp_prgrm[donor.blood_group]["etkas"]]

                        # Filter ETKAS patients by unacceptable antigens
                        etkas_patients = filter_agents_by_unacceptable_antigens(etkas_patients, donor_hla_values)

                        mm000_patients = [agent for agent in etkas_patients if (self.mmHLA(donor.hla_alleles, agent.hla_alleles) == 0)]

                        if mm000_patients: #only proceed if there are mm000_patients
                            mm000_patients_filtered = mm000_patients

                            # In case of fully homozygous donor, rank fully homozygous recipients to fully heterozygous recipients
                            if donor.hla_n_homozygous_loci == 3:
                                # Iterate through homozygosity levels from most to least
                                for homozygosity_level in range(3, -1, -1):
                                    filtered_patients = [agent for agent in mm000_patients_filtered if agent.hla_n_homozygous_loci == homozygosity_level]

                                    # If there are any patients at this level, look for a match and exit the loop if a match was found
                                    if filtered_patients:
                                        filtered_patients.sort(key = lambda patient: self.calc_points(patient, donor), reverse=True)
                                        donor_matched = match_donor_to_ptnt(donor, filtered_patients, ptnts_bld_grp_prgrm[donor.blood_group]["etkas"], "etkas")
                                        if donor_matched:
                                            break
                                        # exclude already considered patients from the list
                                        mm000_patients_filtered = [agent for agent in mm000_patients_filtered if agent not in filtered_patients]

                                if donor_matched:
                                    continue #if matched continue with next organ
                            else:
                                mm000_patients_filtered.sort(key = lambda patient: self.calc_points(patient, donor), reverse=True)
                                donor_matched = match_donor_to_ptnt(donor, mm000_patients_filtered, ptnts_bld_grp_prgrm[donor.blood_group]["etkas"], "etkas")
                                if donor_matched:
                                    continue #if matched continue with next organ
                        
                        if donor.age < 18: # if donor age is < 18 paediatric patients are evaluated first
                            paed_patients = [agent for agent in etkas_patients if (agent.age < 18 and agent.blood_group == donor.blood_group and agent not in mm000_patients)]
                            
                            if paed_patients: #only proceed if there are paed_patients
                                paed_patients.sort(key = lambda patient: self.calc_points(patient, donor), reverse=True)
                                donor_matched = match_donor_to_ptnt(donor, paed_patients, ptnts_bld_grp_prgrm[donor.blood_group]["etkas"], "etkas")
                                if donor_matched:
                                    continue #if matched continue with next organ
                        
                            #exclude for the next step already considered patients
                            etkas_filtered = [agent for agent in etkas_patients if agent not in paed_patients and agent not in mm000_patients]

                        elif donor.age > 18:
                            etkas_filtered = [agent for agent in etkas_patients if agent not in mm000_patients]

                        if etkas_filtered: #only proceed if there are etkas_patients
                            etkas_filtered.sort(key = lambda patient: self.calc_points(patient, donor), reverse=True)
                            donor_matched = match_donor_to_ptnt(donor, etkas_filtered, ptnts_bld_grp_prgrm[donor.blood_group]["etkas"], "etkas")
                            if donor_matched:
                                    continue #if matched continue with next organ
                    
                    # if no match was found, the organ is removed from the list
                    if donor_matched == False:
                        donor.status = "removed"
                        donor.declined_offers = self.declined_offers_counter
            

        else:
            patients_by_blood_group = {"A": [], "AB": [], "B": [], "O": []}

            for agent in self.schedule.agents:
                if agent.type == "patient" and agent.status == "transplantable":    
                    patients_by_blood_group[agent.blood_group].append(agent)

            if len(organ_donors) > 0:
                
                for donor in organ_donors:
                    donor_matched = False
                    self.declined_offers_counter = 0

                    donor_hla_values = set(donor.hla_alleles.values())

                    patients_blood_grp = [agent for agent in patients_by_blood_group[donor.blood_group]]

                    # Filter by unacceptable antigens
                    patients_blood_grp = filter_agents_by_unacceptable_antigens(patients_blood_grp, donor_hla_values)
                    
                    if patients_blood_grp: #only proceed if there are patients
                            patients_blood_grp.sort(key = lambda patient: self.calc_points(patient, donor), reverse=True)
                            donor_matched = match_donor_to_ptnt(donor, patients_blood_grp, patients_by_blood_group[donor.blood_group], "etkas")

                    if donor_matched == False:
                        donor.status = "removed"
                        donor.declined_offers = self.declined_offers_counter

    def step(self):
        """
        Execute a single step of the model.
        """

        st = time.time()
        if (self.datacollector_interval is None) or (self.datacollector_interval == 0):
            self.datacollector.collect(self)  # Collect data from agents at each step
        elif (self.schedule.steps % self.datacollector_interval) == 0:
            self.datacollector.collect(self)
        end_timer(st, "    Datacollector time: ", self.quiet)

        st = time.time()
        self.schedule.step()  # Execute one step of the model's schedule
        end_timer(st, "    Step execution time: ", self.quiet)

        st = time.time()
        n_pts_to_add = next(self.n_pts_add_per_step)
        self.add_patient(n_pts_to_add)  # Add new patients to the model
        end_timer(st, "    Adding patients time: ", self.quiet)
        
        st = time.time()
        n_organ_donors_to_add = next(self.n_organ_donors_step)
        self.add_organ_donor(n_organ_donors_to_add)  # Add new organ donors to the model
        end_timer(st, "    Adding donors time: ", self.quiet)

        st = time.time()
        self.allocate() # Allocate organs to transplantable recipients
        end_timer(st, "    Allocation time: ", self.quiet)