import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def clean(path):

    data = pd.read_csv(path)
    # converting the input into data frames of columns and rows
    sel_data = pd.DataFrame(data, columns=['Match', 'Innings', 'Runs.Remaining', 'Wickets.in.Hand', 'Over', 'Runs',
                                           'Total.Runs', 'Innings.Total.Runs'])
    sel_data = sel_data[sel_data['Innings'] == 1]
    sel_data = sel_data.set_index('Match')

    Matches = []
    for x in list(sel_data.index):
        if x not in Matches:
            Matches.append(x)

    for match in Matches:
        l = len(sel_data.loc[match, 'Total.Runs'])
        if sel_data.loc[match, 'Wickets.in.Hand'].to_numpy()[-1] == 0:
            sel_data.loc[match, 'Total.Overs'] = 50
        else:
            sel_data.loc[match, 'Total.Overs'] = len(sel_data.loc[match, 'Total.Runs'])

    sel_data = sel_data[sel_data['Total.Overs'] == 50]

    Matches = []
    for x in list(sel_data.index):
        if x not in Matches:
            Matches.append(x)

    new_df = pd.DataFrame(columns=['Innings', 'Runs.Remaining', 'Wickets.in.Hand', 'Over', 'Runs', 'Total.Runs', 'Innings.Total.Runs'])

    for match in Matches:
        dummy = sel_data.loc[match].iloc[0, :].copy()
        dummy['Over'] = 0
        dummy['Runs.Remaining'] = dummy['Innings.Total.Runs']
        dummy['Wickets.in.Hand'] = 10
        dummy['Runs'] = 0
        new_df = new_df.append(dummy)
        new_df = new_df.append(sel_data.loc[match, :])

    sel_data = new_df

    sel_data['Over'] = 50 - sel_data['Over']
    sel_data = sel_data.reset_index()
    sel_data = sel_data.rename(columns={'index': 'Match'})

    return sel_data


# function to find the average of all the runs at a given wicket available
def getMeanRunByWicket(sel_data, w):
    selWicket = sel_data['Wickets.in.Hand'] == w
    df = sel_data[selWicket]
    return np.mean(df.groupby(['Match'])['Runs.Remaining'].max())


# Model for 1st question
def model1(Z0, b, u):
    return Z0 * (1 - np.exp(-b * u))


# Model for 2nd question
def model2(Z0, u, L):
    return Z0 * (1 - np.exp(-L / Z0 * u))


# Function to get average of maximum runs by overs and wicket
def getAverageMaxRun(sel_data):
    Z0_list = []
    for w in np.arange(10):
        Z0_list.append(getMeanRunByWicket(sel_data, w + 1))
    return Z0_list


def errorFunc1(params, train_data):
    opt_Z0 = params[0]
    opt_b = params[1]
    train_data_run = train_data[0]
    train_data_over = train_data[1]
    loss = 0
    for i in range(len(train_data_run)):
        predicted_run = model1(opt_Z0, opt_b, train_data_over[i])
        loss += (predicted_run - train_data_run[i]) ** 2
    return loss


# The sum of squared errors loss function, summed across overs and wickets, that we indent to minimize to get the optimum results
def errorFunc2(params, train_data):
    opt_L = params[-1]
    opt_Z0 = params[:-1]
    train_data_run = train_data[0]
    train_data_over = train_data[1].astype('int32')
    train_data_wicket = train_data[2].astype('int32')
    loss = 0
    for i in range(len(train_data_run)):
        predicted_run = model2(opt_Z0[train_data_wicket[i] - 1], train_data_over[i], opt_L)
        loss += (predicted_run - train_data_run[i]) ** 2
    return loss


def DuckworthLewis20Params(CSV_file_name):
    sel_data = clean(CSV_file_name)
    n = len(sel_data)

    # initial guess for Z0 parameters
    # initial_Z0 = getAverageMaxRun()
    initial_Z0 = [14.4176399, 30.17553671, 58.15210963, 91.93851205, 118.16310953, 155.57062764, 186.39519971, 232.43576966, 264.3617314, 307.68280687]
    initial_b = [0.71047785, 0.33946087, 0.17614862, 0.11141592, 0.08668876, 0.06584414, 0.05495535, 0.04406987, 0.03874772, 0.03329212]

    # Use the Scipy Optimize library's Minimize function ob the target function, with parameters of Z0 & L, and the data required as arguments, with BFGS algorithm
    opt_Z0 = [0] * 10
    opt_b = [0] * 10
    min_error = [0] * 10
    for i in range(10):
        print(f"opt for wicket {i + 1} started")
        sel_data_wicketwise = sel_data[sel_data['Wickets.in.Hand'] == i + 1]
        sol = minimize(errorFunc1, [initial_Z0[i], initial_b[i]],
                       args=[sel_data_wicketwise['Runs.Remaining'].values, sel_data_wicketwise['Over'].values],
                       method='L-BFGS-B')
        min_error[i] = sol.fun
        opt_Z0[i] = sol.x[0]
        opt_b[i] = sol.x[1]

    print("\n\nThe minimized per point error = " + str(sum(min_error) / n))
    return opt_Z0, opt_b


def DuckworthLewis11Params(CSV_file_name):
    sel_data = clean(CSV_file_name)
    n = len(sel_data)
    # initial guess for Z0 parameters
    initial_Z0 = getAverageMaxRun(sel_data)
    initial_L = 10
    initial_params = initial_Z0 + [initial_L]

    # Use the Scipy Optimize library's Minimize function ob the target function, with parameters of Z0 & L, and the data required as arguments, with BFGS algorithm
    sol = minimize(errorFunc2, initial_params, args=[sel_data['Runs.Remaining'].values, sel_data['Over'].values,
                                                     sel_data['Wickets.in.Hand'].values], method='L-BFGS-B')

    opt_Z0 = sol.x[:-1]
    opt_L = sol.x[-1]

    min_error = sol.fun
    print("\n\nThe minimized per point error = " + str(min_error / n))
    return opt_Z0, opt_L


def plot2(Z0, L):
    # Plot the data of fraction of resources available predicted with optimized values
    fig = plt.figure(1)
    plt.xlabel('Overs used')
    plt.ylabel('Resource remaining %')

    # This is the maximum possible resource available prediction of Z
    Z50 = model2(Z0[-1], 50, L)

    over_axis = np.arange(51)

    # For each wicket, we plot graphs using the function model to predict with optimized values over wickets
    for i in range(10):
        y = 100 * model2(Z0[i], 50.0 - over_axis, L) / Z50
        zf = "{:.0f}".format(Z0[i])
        plt.plot(over_axis, y, label='Z(' + str(i + 1) + ') = ' + str(zf))
        plt.legend()

    slope = -2 * over_axis + 100
    plt.plot(over_axis, slope, 'black')

    fig.suptitle('Question 2', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.85)

    plt.savefig('Harshit_Samani_Assignment2_output_plot2.png')
    plt.show()


def plot1(Z0, b):
    # Plot the data of fraction of resources available predicted with optimized values
    fig = plt.figure(1)
    plt.xlabel('Overs used')
    plt.ylabel('Resource remaining %')

    # This is the maximum possible resource available prediction of Z
    Z50 = model1(Z0[-1], 50, b[-1])

    over_axis = np.arange(51)

    # For each wicket, we plot graphs using the function model to predict with optimized values over wickets
    for i in range(10):
        y = 100 * model1(Z0[i], 50.0 - over_axis, b[i]) / Z50
        zf = "{:.0f}".format(Z0[i])
        plt.plot(over_axis, y, label='Z(' + str(i + 1) + ') = ' + str(zf))
        plt.legend()


    slope = -2 * over_axis + 100
    plt.plot(over_axis, slope, 'black')

    fig.suptitle('Question 1', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.85)
    plt.savefig('Harshit_Samani_Assignment2_output_plot1.png')
    plt.show()


path = "04_cricket_1999to2011.csv"

opt_Z01, opt_b = DuckworthLewis20Params(path)
plot1(opt_Z01, opt_b)

opt_Z02, opt_L = DuckworthLewis11Params(path)
plot2(opt_Z02, opt_L)

# opt_Z02= [14.547958367271676,
#  31.591956658169426,
#  56.99560247075761,
#  91.93851370819979,
#  118.1631097964584,
#  151.2539092051594,
#  186.39519963124718,
#  232.43576517880828,
#  264.3617313798436,
#  307.68280690355465]
#
# opt_b = [0.6625481969971553,
#  0.2666528781723382,
#  0.19150901789097197,
#  0.10986059944510322,
#  0.08623644423307075,
#  0.07069444946386981,
#  0.05519035598169698,
#  0.044321448512187264,
#  0.03885035140413219,
#  0.03331663467949012]
#
# opt_Z01 = [ 14.4176399 ,  30.17553671,  58.15210963,  91.93851205,
#        118.16310953, 155.57062764, 186.39519971, 232.43576966,
#        264.3617314 , 307.68280687]
#
# opt_L = 10.24341380846697


