from gradient_descent import get_local_gradient
from tools import gradient_aggregate, generate_data_range, generate_constant, generate_slope, get_dataset

#get_dataset(isFirst = True, slope = None, constant = None, data_range = None)

number_locations = 5
theta0_list = []
theta1_list = []

set_data_range = generate_data_range()
set_slope = generate_slope()
set_constant = generate_constant(set_data_range)


# get the datasets for each
for location in range(number_locations):
    # general the x and y dataset for the location
    local_x, local_y = get_dataset(slope = set_slope, constant = set_constant, data_range = set_data_range)

    theta0, theta1, min_error = get_local_gradient(x = local_x, y = local_y)
    theta0_list += [theta0]
    theta1_list += [theta1]

theta0, theta1 = gradient_aggregate(theta0_list, theta1_list)
print("Theta 0: %s"%theta0)
print("Theta 1: %s"%theta1)