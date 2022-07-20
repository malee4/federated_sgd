import random

# arbitrary theta0 and theta1 starting values
theta0 = 0 # constant term
theta1 = 0 

a = 0.1 # step 

# create a sample dataset
def get_dataset():
    # generate dataset
    # number of items in the set
    num_data_points = int(random.random() * 50) + 50
    print("Total data points: %s"%num_data_points)

    # get a random range, greater than or equal to 10
    data_range = int(random.random() * 90) + 10

    # generate an approximate constant theta0
    approximate_constant = int(random.random() * data_range)
    print("Approximate constant: %s"%approximate_constant)
    
    # generate an approximate slope theta1
    approximate_slope = int(random.random() * 10) + 1 # randomly generate a slope
    print("Approximate slope: %s"%approximate_slope)

    x = []
    y = [] 

    for item in range(num_data_points):
        x_value = int(random.random() * data_range)
        random_error = int(random.random() * data_range * 0.2)
        y_value = x_value * approximate_slope + approximate_constant + random_error

        x += [x_value]
        y += [y_value]
    
    # print the full generated datasets
    # print("x set: %s"%x)
    # print("y set: %s"%y)
    print("X")
    for x_val in x:
        print(x_val)
    
    print("Y")
    for y_val in y:
        print(y_val)

    return x, y

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

# h function
# lambda x: theta0 + theta1*x

# get the dataset 
x, y = get_dataset()

if len(x) != len(y):
    raise Exception("x and y dataset lengths do not match")

m = len(x)

temp_theta0 = 0
temp_theta1 = 0

h = lambda x: theta0 + theta1*x

min_error = sum_of_differences(h, x, y) ** 2
curr_error = sum_of_differences(h, x, y) ** 2

while curr_error <= min_error:
    
    min_error = curr_error
    # simultaneously update
    temp_theta0 = theta0 - (1 / float(m)) * a * sum_of_differences(h, x, y)
    temp_theta1 = theta1 - (1 / float(m)) * a * sum_of_differences(h, x, y, notConstant = True)
    theta0 = temp_theta0
    theta1 = temp_theta1

    h = lambda x: theta0 + theta1*x

    curr_error = sum_of_differences(h, x, y) ** 2

print("Theta 0 = " + str(theta0))
print("Theta 1 = " + str(theta1))

