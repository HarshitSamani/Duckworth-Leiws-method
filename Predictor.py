from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from math import sin, cos, sqrt, tan, radians, pi, degrees
import datetime as dt
from scipy.optimize import fsolve
from tqdm import tqdm


def cart2pol(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = ((np.arctan2(y, x) * 180 / np.pi)) % 360
    return (r, theta)


def pol2cart(r, theta):
    x = r * np.cos(theta * np.pi / 180)
    y = r * np.sin(theta * np.pi / 180)
    return np.array([x, y])


def get_intesection(circle, line):
    center, radius = circle
    center = pol2cart(center[0], center[1])

    point, angle = line
    point = pol2cart(point[0], point[1])

    func = lambda x: [(x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2 - radius ** 2,
                      (x[1] - point[1]) - np.tan(angle * np.pi / 180) * (x[0] - point[0])]
    init = [point[0] + radius * np.cos(angle * np.pi / 180), point[1] + radius * np.sin(angle * np.pi / 180)]

    sol = fsolve(func, init)
    sol = cart2pol(sol[0], sol[1])
    return sol


def MarsEquantModel(c, r, e1, e2, z, s, oppositions):
    orbit = ((1, c), r)
    equant = (e1, (e2 + z))

    equant_oppositions_angles = (z + oppositions[:, 0] * s)

    equant_oppositions_cords = []

    for angle in equant_oppositions_angles:
        equant_oppositions_cords.append(get_intesection(orbit, (equant, angle))[1])

    equant_oppositions_cords = np.array(equant_oppositions_cords)

    errors = equant_oppositions_cords - oppositions[:, 1]

    return errors, max(abs(errors))


def plotModel(c, r, e1, e2, z, s, oppositions):
    equant = (e1, (e2 + z) % 360)
    cart_equant = pol2cart(equant[0], equant[1])

    equant_oppositions_angles = (z + oppositions[:, 0] * s) % 360

    figure, axes = plt.subplots()
    Drawing_circle = plt.Circle((cos(radians(c)), sin(radians(c))), r, fill=False)
    plt.plot(cart_equant[0], cart_equant[1], 'ro', markersize=2)
    plt.plot(cos(radians(c)), sin(radians(c)), 'bo', markersize=2)
    plt.plot(0, 0, 'go', markersize=2)
    axes.set_aspect(1)
    # plt.grid()
    axes.add_artist(Drawing_circle)
    plt.xlim(-r - 2, r + 2)
    plt.ylim(-r - 2, r + 2)

    for i in range(12):
        intersect1 = get_intesection(((1, c), r), ((0, 0), oppositions[i, 1]))
        intersect1 = pol2cart(intersect1[0], intersect1[1])
        plt.plot([0, intersect1[0]], [0, intersect1[1]], linewidth=0.7)

        intersect2 = get_intesection(((1, c), r), (equant, equant_oppositions_angles[i]))
        intersect2 = pol2cart(intersect2[0], intersect2[1])
        plt.plot([cart_equant[0], intersect2[0]], [cart_equant[1], intersect2[1]], linestyle='dashed', linewidth=0.7)

    # fig1 = plt.gcf()
    # fig1.savefig('plot', dpi=200)
    plt.show()


def bestOrbitInnerParams(r, s, oppositions):
    best_maxError = float('inf')
    for c in np.linspace(145.88, 145.89, 11):
        for e1 in np.linspace(1.90, 1.91, 11):
            for e2 in np.linspace(93, 93.5, 11):
                for z in np.linspace(55.8, 55.9, 11):
                    Errors, maxError = MarsEquantModel(c, r, e1, e2, z, s, oppositions)
                    if maxError < best_maxError:
                        best_maxError = maxError
                        best_Errors = Errors
                        best_c = c
                        best_e1 = e1
                        best_e2 = e2
                        best_z = z
    return best_c, best_e1, best_e2, best_z, best_Errors, best_maxError


def bestR(s, oppositions):
    best_maxError = float('inf')
    for r in tqdm(np.linspace(10.2, 10.3, 11)):
        c, e1, e2, z, Errors, maxError = bestOrbitInnerParams(r, s, oppositions)
        if maxError < best_maxError:
            best_maxError = maxError
            best_Errors = Errors
            best_c = c
            best_e1 = e1
            best_e2 = e2
            best_z = z
            best_r = r
    return best_r, best_Errors, best_maxError


def bestS(r, oppositions):
    best_maxError = float('inf')
    for s in 360 / np.linspace(686.5, 687.5, 10):
        c, e1, e2, z, Errors, maxError = bestOrbitInnerParams(r, s, oppositions)
        if maxError < best_maxError:
            best_maxError = maxError
            best_Errors = Errors
            best_c = c
            best_e1 = e1
            best_e2 = e2
            best_z = z
            best_s = s
    return best_s, best_Errors, best_maxError


def bestMarsOrbitParams(oppositions):
    best_maxError = float('inf')
    for r in tqdm(np.linspace(10.25, 10.35, 11)):
        for s in 360 / np.linspace(686.91, 686.92, 11):
            c, e1, e2, z, Errors, maxError = bestOrbitInnerParams(r, s, oppositions)
            if maxError < best_maxError:
                best_maxError = maxError
                best_Errors = Errors
                best_c = c
                best_e1 = e1
                best_e2 = e2
                best_z = z
                best_r = r
                best_s = s
    return best_r, best_s, best_c, best_e1, best_e2, best_z, best_Errors, best_maxError


data = pd.read_csv('01_data_mars_opposition_updated.csv')

ref = dt.datetime(data['Year'][0], data['Month'][0], data['Day'][0], data['Hour'][0], data['Minute'][0])
for i in range(12):
    dummy = dt.datetime(data['Year'][i], data['Month'][i], data['Day'][i], data['Hour'][i], data['Minute'][i]) - ref
    data.loc[i, 'diff'] = dummy.days + dummy.seconds / (3600 * 24)
    data.loc[i, 'angle'] = (data.loc[i, 'ZodiacIndex'] * 30 + data.loc[i, 'Degree'] + data.loc[i, 'Minute.1'] / 60 +
                            data.loc[i, 'Second'] / 3600) % 360

oppositions = data[['diff', 'angle']].to_numpy()

best_Errors, best_maxError = MarsEquantModel(c=145.887, r=10.28, e1=1.906, e2=93.05, z=55.87, s=360/686.918, oppositions=oppositions)
best_Errors = [round(x,4) for x in best_Errors]
best_maxError = round(best_maxError,ndigits=4)
print(best_Errors, best_maxError)
plotModel(c=145.887, r=10.28, e1=1.906, e2=93.05, z=55.87, s=360/686.918, oppositions=oppositions)


# x = bestMarsOrbitParams(oppositions)
# print(x)
# best_r,best_s,best_c,best_e1,best_e2,best_z,best_Errors,best_maxError = x
# plotModel(c=best_c, r=best_r, e1=best_e1, e2=best_e2, z=best_z, s=best_s, oppositions=oppositions)
#
#
# with open('report','a') as f:
#     f.write(f"best_r : {best_r}\n"
#             f"best_s : {best_s}\n"
#             f"best_c : {best_c}\n"
#             f"best_e1 : {best_e1}\n"
#             f"best_e2 : {best_e2}\n"
#             f"best_z : {best_z}\n"
#             f"best_Errors : {best_Errors}\n"
#             f"best_maxError : {best_maxError}\n\n")

with open("aal_report", "w") as file:
    file.write(f"num_hidden_channels={num_hidden_channels}"
               f"hidden_channels_dims={hidden_channels_dims}"
               f"Activation={Activation}"
               f"Patience={Patience}"
               f"Dropout={Dropout}"
               f"Learning_rate={Learning_rate}"
               f"seeds={seeds}"
               f"epochs={epochs}"
               f"Threshold percentage={threshold_percentage}%"
               f"threshold_pos={th_p:0.2f}"
               f"threshold_neg={th_n:0.2f}"
               f"train_avg_accuracy={stat.mean(final_train_accs)*100:0.2f}%"
               f"train_accuracy_stdev={stat.stdev(final_train_accs)*100:0.2f}%"
               f"val_avg_accuracy={stat.mean(final_val_accs)*100:0.2f}%"
               f"val_accuracy_stdev={stat.stdev(final_val_accs)*100:0.2f}%"
               f"test_avg_accuracy={stat.mean(final_test_accs)*100:0.2f}%"
               f"test_accuracy_stdev={stat.stdev(final_test_accs)*100:0.2f}%\n")
