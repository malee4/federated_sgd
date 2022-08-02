import random
from linear_tools import get_dataset

# calculate the sum of the difference
def sum_of_differences(h, x, y, notConstant = False):
    sum = 0
    # sum the difference between the predicted and actual value
    for i in range(len(x)):
        temp = h(x[i]) - y[i]
        if notConstant:
            sum += temp * x[i]
        else:
            sum += temp
    return sum

# get the dataset 
def get_linear_gradient(isFirst = False, slope = None, constant = None, x = None, y = None, a = 0.1):
    # if dataset is not given, randomly generate a dataset
    # NOTE: THIS IS NOT CORRECT/DOES NOT COVER ALL CASES
    if (x or y) and len(x) != len(y):
        raise Exception("Please pass in a complete dataset with matching lengths.")
    elif not x and not y:
        if isFirst:
            x, y, slope, constant = get_dataset(isFirst = isFirst, slope = slope, constant = constant)
        else:
            x, y = get_dataset(isFirst = isFirst, slope = slope, constant = constant)
    
    # probably not the best version of generating a dataset, but it's fine :)
    m = len(x)

    temp_theta0 = 0
    temp_theta1 = 0

    # arbitrary theta0 and theta1 starting values
    theta0 = 0 # constant term
    theta1 = 0 

    h = lambda x: theta0 + theta1*x

    min_error = sum_of_differences(h, x, y) ** 2
    curr_error = sum_of_differences(h, x, y) ** 2

    while curr_error <= min_error:
        min_error = curr_error
        # simultaneously update the values
        temp_theta0 = theta0 - (1 / float(m)) * a * sum_of_differences(h, x, y)
        temp_theta1 = theta1 - (1 / float(m)) * a * sum_of_differences(h, x, y, notConstant = True)
        theta0 = temp_theta0
        theta1 = temp_theta1

        h = lambda x: theta0 + theta1*x

        # calculate the new error with the values
        curr_error = sum_of_differences(h, x, y) ** 2

    if isFirst: # that is, if no information was passed in
        return theta0, theta1, min_error, x, y, slope, constant
    
    return theta0, theta1, min_error

