# ENEL525 Lab 5
# Athena McNeil-Roberts

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Assume a car operates using five (5) gears and the gear selection depends on the instantaneous 
# speed of the car and the incline of the road with reference to the gravitational horizontal.

# In this lab, you will be exploring the fuzzy inference method using scikit-fuzzy (skfuzzy) library to 
# implement the gear selector based on the 2 inputs. For this goal, you need to follow the steps below 
# for membership functions and decision rules definition.

# create Antecedent for both inputs and Consequent for gear output
speed = ctrl.Antecedent(np.arange(0, 81), 'speed')
incline = ctrl.Antecedent(np.arange(-15, 16), 'incline')
gear = ctrl.Consequent(np.arange(1, 6), 'gear')

# set defuzzification method as 'mom'
gear.defuzzify_method = 'mom'

#  membership functions for speed input
speed['rolling'] = fuzz.trapmf(speed.universe, [0, 0, 10, 11])
speed['slow'] = fuzz.trapmf(speed.universe, [10, 11, 25, 26])
speed['medium'] = fuzz.trapmf(speed.universe, [25, 26, 50, 51])
speed['fast'] = fuzz.trapmf(speed.universe, [50, 51, 65, 66])
speed['speeding'] = fuzz.trapmf(speed.universe, [65, 66, 80, 80])

#  membership functions for incline input
incline['steep'] = fuzz.trapmf(incline.universe, [-15, -15, -10, -9])
incline['slope'] = fuzz.trapmf(incline.universe, [-11, -10, -1, 0])
incline['flat'] = fuzz.trapmf(incline.universe, [-1, 0, 0, 1])
incline['up'] = fuzz.trapmf(incline.universe, [0, 1, 10, 11])
incline['climb'] = fuzz.trapmf(incline.universe, [9, 10, 15, 15])


#  membership functions for gear output
gear['1'] = fuzz.trimf(gear.universe, [1, 1, 2])
gear['2'] = fuzz.trimf(gear.universe, [1, 2, 3])
gear['3'] = fuzz.trimf(gear.universe, [2, 3, 4])
gear['4'] = fuzz.trimf(gear.universe, [3, 4, 5])
gear['5'] = fuzz.trimf(gear.universe, [4, 5, 5])

speed.view()
incline.view()
gear.view()


# rules based on table
rule1 = fuzz.control.Rule((speed['rolling'] & incline['steep']), gear['2'])
rule2 = fuzz.control.Rule((speed['rolling'] & incline['slope']), gear['1'])
rule3 = fuzz.control.Rule((speed['rolling'] & incline['flat']), gear['1'])
rule4 = fuzz.control.Rule((speed['rolling'] & incline['up']), gear['1'])
rule5 = fuzz.control.Rule((speed['rolling'] & incline['climb']), gear['1'])

rule6 = fuzz.control.Rule((speed['slow'] & incline['steep']), gear['3'])
rule7 = fuzz.control.Rule((speed['slow'] & incline['slope']), gear['2'])
rule8 = fuzz.control.Rule((speed['slow'] & incline['flat']), gear['2'])
rule9 = fuzz.control.Rule((speed['slow'] & incline['up']), gear['1'])
rule10 = fuzz.control.Rule((speed['slow'] & incline['climb']), gear['1'])

rule11 = fuzz.control.Rule((speed['medium'] & incline['steep']), gear['4'])
rule12 = fuzz.control.Rule((speed['medium'] & incline['slope']), gear['4'])
rule13 = fuzz.control.Rule((speed['medium'] & incline['flat']), gear['3'])
rule14 = fuzz.control.Rule((speed['medium'] & incline['up']), gear['3'])
rule15 = fuzz.control.Rule((speed['medium'] & incline['climb']), gear['2'])

rule16 = fuzz.control.Rule((speed['fast'] & incline['steep']), gear['5'])
rule17 = fuzz.control.Rule((speed['fast'] & incline['slope']), gear['5'])
rule18 = fuzz.control.Rule((speed['fast'] & incline['flat']), gear['4'])
rule19 = fuzz.control.Rule((speed['fast'] & incline['up']), gear['4'])
rule20 = fuzz.control.Rule((speed['fast'] & incline['climb']), gear['4'])

rule21 = fuzz.control.Rule((speed['speeding'] & incline['steep']), gear['5'])
rule22 = fuzz.control.Rule((speed['speeding'] & incline['slope']), gear['5'])
rule23 = fuzz.control.Rule((speed['speeding'] & incline['flat']), gear['5'])
rule24 = fuzz.control.Rule((speed['speeding'] & incline['up']), gear['4'])
rule25 = fuzz.control.Rule((speed['speeding'] & incline['climb']), gear['4'])


controlSystem = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                   rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19,
                                   rule20, rule21, rule22, rule23, rule24, rule25])

gear_selector = ctrl.ControlSystemSimulation(controlSystem)

# test inputs
gear_selector.input['speed'] = 15
gear_selector.input['incline'] = -12
gear_selector.compute()
print("Gear selection number (test case 1): ", gear_selector.output['gear'])

gear_selector.input['speed'] = 80
gear_selector.input['incline'] = 5
gear_selector.compute()
print("Gear selection number (test case 2): ", gear_selector.output['gear'])

gear_selector.input['speed'] = 45
gear_selector.input['incline'] = 0
gear_selector.compute()
print("Gear selection number (test case 3): ", gear_selector.output['gear'])

gear_selector.input['speed'] = 21
gear_selector.input['incline'] = 10
gear_selector.compute()
print("Gear selection number (test case 4): ", gear_selector.output['gear'])

gear_selector.input['speed'] = 75
gear_selector.input['incline'] = -3
gear_selector.compute()
print("Gear selection number (test case 5): ", gear_selector.output['gear'])
plt.show()
