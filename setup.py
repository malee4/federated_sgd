# import packages
import random

def print_dataset(x, y):
    # print the x dataset
    print("x")
    for x_val in x:
        print(x_val)
    
    # print the y dataset
    print("y")
    for y_val in y:
        print(y_val)
    return 

# generate the dataset
def get_dataset(isFirst = True, slope = None, constant = None):
    if not isFirst and not (slope and constant):
        return None, None

    # number of items in the set
    num_data_points = int(random.random() * 50) + 50
    print("Total data points: %s"%num_data_points)

    # get a random range, greater than or equal to 10
    data_range = int(random.random() * 90) + 10

    if isFirst:
        # generate an approximate constant theta
        constant = int(random.random() * data_range)
        print("Approximate constant: %s"%constant)
        
        # generate an approximate slope theta1
        slope = int(random.random() * 10) + 1 # randomly generate a slope
        print("Approximate slope: %s"%slope)

    random_error = int(random.random() * data_range * 0.1) 

    x = []
    y = [] 

    for item in range(num_data_points):
        x_value = int(random.random() * data_range)
        
        y_value = x_value * slope + constant + random_error

        x += [x_value]
        y += [y_value]

    # print_dataset(x, y)
    if isFirst:
        return x, y, slope, constant
    
    return x, y