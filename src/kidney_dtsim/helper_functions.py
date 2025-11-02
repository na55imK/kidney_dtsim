# helper_functions.py

# Data visualization tools.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Data manipulation and analysis.
import pandas as pd

def plot_donor_ages_at_step(df, step):
    sns.set_style("whitegrid")
    # Filter out the data for the specified step
    step_data = df.xs(step, level="Step")

    # Filter out the donor age data
    donor_ages = step_data[(step_data["agent_type"] == "organ_donor")]["age"]

    # Check if there are donor ages to plot
    if not donor_ages.empty:
        # Plotting
        sns.histplot(donor_ages, binwidth=1)
        sns.despine(top=True, left=True)
        plt.title(f"Additive distribution of donor ages at step {step}")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()


def plot_age_at_removal_at_step(df, step):
    sns.set_style("whitegrid")
    # Filter out the data for the specified step
    step_data = df.xs(step, level="Step")

    ages = step_data[(step_data["agent_type"] == "patient") & (step_data["status"] == "removed")]["age"]

    # Check if there are ages to plot
    if not ages.empty:
        # Plotting
        sns.histplot(ages, binwidth=1)
        sns.despine(top=True, left=True)
        plt.title(f"Additive distribution of age at removal from waiting list at step {step}")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()

def plot_age_at_transplant_at_step(df, step):
    sns.set_style("whitegrid")
    # Filter out the data for the specified step
    step_data = df.xs(step, level="Step")

    ages = step_data[(step_data["agent_type"] == "patient") & (step_data["status"] == "transplanted")]["age"]

    # Check if there areages to plot
    if not ages.empty:
        # Plotting
        sns.histplot(ages, binwidth=1)
        sns.despine(top=True, left=True)
        plt.title(f"Additive distribution of age at transplantation at step {step}")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()

def plot_waitinglist_time_at_step(df, step):
    sns.set_style("whitegrid")
    # Filter out the data for the specified step
    step_data = df.xs(step, level="Step")

    # Filter out the waiting time of transplanted patients
    waitinglist_time = step_data[(step_data["agent_type"] == "patient") & (step_data["status"] == "transplanted")]["waitinglist_time"]
    age = step_data[(step_data["agent_type"] == "patient") & (step_data["status"] == "transplanted")]["age"]

    # Check if there are waiting times to plot
    if not waitinglist_time.empty:
        #Plotting
        sns.scatterplot(x = age, y = waitinglist_time, s = 4, alpha=0.2)
        sns.despine(top=True, left=True)
        plt.ylabel("Waitinglist time")
        plt.xlabel("Age")
        plt.title(f"Waiting time for transplanted patients up to step {step}")
        plt.show()

def plot_rec_vs_donor_age(df):
    """
    Plots the Recipients versus the donor age.
    Input has to be the merged_patients_donor dataframe
    or a dataframe with the columns 'agent_type', 'status_recipient', 'age_recipient', and 'age_donor'.
    """
    sns.set_style("whitegrid")
    
    filtered_df = df[(df["agent_type"] == "patient") & (df["status_recipient"] == "transplanted")].loc[:, ["age_recipient", "age_donor"]]
    
    sns.scatterplot(data=filtered_df,x="age_recipient", y="age_donor", s = 4, alpha=0.2)
    sns.despine(top=True, left=True)
    plt.title(f"Recipient vs. donor age")
    plt.xlabel("Recipient age")
    plt.ylabel("Donor age")
    plt.show()

def plot_ridge(df, col_label):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(11, rot = -.25, light = .7)
    g = sns.FacetGrid(df, row="Step", hue="Step", palette=pal, aspect=10, height=.6)
    
    g.map(sns.kdeplot, col_label,
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, col_label, clip_on=False, color="w", lw=2, bw_adjust=.5)
    
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, col_label)
    
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
    
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    return(g)

def concat_agent_vars_df(df):
    steps = df.index.get_level_values("Step").unique()

    list = []
    for step in steps:
        tmp_df = df.xs(step, level="Step").copy(deep = True)
        tmp_df.loc[:,"Step"] = step
        list.append(tmp_df)

    return(pd.concat(list, ignore_index=True))