import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def calculate_diameter(points):
    
    diameter = 0.0
    num_points = len(points)
    for i in range(num_points):
        for j in range(i+1, num_points):
            distance = euclidean_distance(points[i], points[j])
            if distance > diameter:
                diameter = distance
    return diameter

######################### offline algorithm to produce offline OPT distance #################################
def offline_k_center(points, k):
    # Initialize the first center randomly
    centers = [points[np.random.randint(len(points))]]
    
    while len(centers) < k:
        # Find the point that is the farthest from any center
        next_center = max(points, key=lambda point: min(euclidean_distance(point, center) for center in centers))
        centers.append(next_center)
    
    return centers

def max_distance_to_centers(points, centers):
    max_dist = 0
    for point in points:
        min_dist_to_center = min(euclidean_distance(point, center) for center in centers)
        max_dist = max(max_dist, min_dist_to_center)
    return max_dist

def plot_points_and_centers(points, centers):
    plt.figure(figsize=(8, 6))
    x, y = zip(*points)
    cx, cy = zip(*centers)
    
    plt.scatter(x, y, color='blue', label='Points')
    plt.scatter(cx, cy, color='red', s=100, label='Centers', edgecolors='black')
    plt.title("Offline k-Center Clustering")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()


# Generate random points
np.random.seed(42)
data_points = np.random.rand(100, 2) * 100  # 100 points in a 100x100 grid

# Number of centers
k = 5

# Solve the offline k-center problem
centers = offline_k_center(data_points, k)

# Calculate the maximum distance to the nearest center
# This value used as input parameter in the online problem
max_dist = max_distance_to_centers(data_points, centers)

diam = calculate_diameter(data_points)

# Plot the points and the selected centers
#plot_points_and_centers(data_points, centers)

print("Selected Centers:", centers)
print("Maximum distance to nearest center:", max_dist)



####################################### online positive-body chasing for k-center ##################################################


# define the radius for the ball to outline the covering/packing constraints
beta = 1.5
r = beta * max_dist

epsilon = 0.25


####################################### functions needed for main method ############################################################

# initialization for online_k_center
def initialization(points):
    x = np.zeros(len(points))

    # start with a set of candidate points of randomly selected 10 points from input
    #indices = np.random.choice(points.shape[0], size = 10, replace=False)
    #candidates = points[indices, :]
    
    print("\n")
    #print("candidate indices:", indices)
    #print("center candidates:", candidates)

    indices = []
    candidates = []

    # initialize constraint matrix
    constraint_matrix = []

    return x, indices, candidates, constraint_matrix

# update the covering constraints to accommodate insertion and deletion of clients
def update_constraints(C,s, n):
    # update the constraint matrix that for each client, Cx >= 1
    # where row t in C for client t has 0s at entries in s
    new_row = np.zeros(n)
    new_row[s] = 1
    C.append(new_row)
    
    return C

# find all points within the radius of the current client
# this step can be thought of as updating the covering constraint
def find_candidates(candidate_indices, points, client, radius):
    s = []
    for index in candidate_indices:
        #print("index in candidate_indicies:", index)
        if euclidean_distance(points[int(index)], client) <= radius:
            s.append(int(index))
    
    print("candidates within radius:", s)
    return s

# check if new client's constraint is satisfied
def check_covering_feasibility(s, x):
    # check if the covering constraint is satisfied
    covering_sum = np.sum(x[s])
    print("total weight in radius:", covering_sum)
    if covering_sum >= 1:
        return True
    else:
        return False


# covering objective function for solving x's
def covering_objective(x, x_old, s, epsilon):

    # we only deal with the x_i's whose coefficient c_i is nonzero
    prev_x_hat = np.zeros(len(s))

    for i in range(len(prev_x_hat)):
        prev_x_hat[i] = x_old[s[i]] + epsilon/(4*len(s))
    
    #prev_x = np.zeros(len(s))
    return np.sum((x + epsilon/(4*len(s))) * np.log((x + epsilon/(4*len(s)))/prev_x_hat) - (x + epsilon/(4*len(s))))


# packing objective function for solving x's
def packing_objective(x, x_old):

    # avoiding divide by zero
    c = 1e-10
    x_old = np.maximum(x_old, c)
    
    return np.sum(x * np.log(x/x_old) - x)


# covering constraint for solving x's
def covering_constraint(x):
    return np.sum(x) - 1


# packing constraint for solving x's
def packing_constraint(x, epsilon, k):
    return (1 + epsilon) * k - np.sum(x) 


# optimize and update the values of x's when a covering constraint not satisfied
def update_covering_variables(x, s, epsilon):
    # start with x's at t-1
    x0 = np.zeros(len(s))
    for i in range(len(s)):
        x0[i] = x[s[i]]

    # set up constraint for scipy solver for x's
    cons = {'type': 'ineq', 'fun':covering_constraint}

    result = minimize(covering_objective, x0, args=(x, s, epsilon), constraints=cons, method='SLSQP')

    print("updated fractional solutions:", result.x)
    
    # finally, update the values in the original x vector
    x_new = x
    for i in range(len(s)):
        x_new[s[i]] = result.x[i]

    return x_new


# optimize and update the values of x's when a packing constraint not satisfied
def update_packing_variables(x, epsilon, k):
    
    cons = {'type': 'ineq', 'fun':packing_constraint, 'args':(epsilon, k)}

    x0 = x

    result = minimize(packing_objective, x0, args=(x), constraints=cons, method='SLSQP')

    print("updated fraction solutions after packing violation:", result.x)

    return result.x

######################################### methods for k-center rounding ###########################################

# set the parameters
alpha = 3 + 3 * np.sqrt(2)
delta = np.sqrt(2)

# testing on batches of data points
#data_points = data_points[:50].copy()

#print("radius:", r)

# subroutine to find the balls B_i and B_hait_i for a given center_index
# this subroutine is called whenever the set S is updated
def find_balls(data_points, clients, center_index, radius, radius_hat):

    print("inside find_ball function")
    B_i = []
    B_i_hat =[]

    for i in range(len(clients)):
        #print("current client:", j)
        #print("current center:", s[i])

        #print("client coordinate:", data_points[j])
        #print("center coordinate:", data_points[s[i]])
        #print("distance:", euclidean_distance(data_points[j], data_points[s[i]]))
        client_coordinate = clients[i][1]
        client_index = clients[i][0]
        print("distance between client and center:", euclidean_distance(client_coordinate, data_points[center_index]))
        if euclidean_distance(client_coordinate, data_points[center_index]) <= radius:
            B_i.append(client_index)
            
        if euclidean_distance(client_coordinate, data_points[center_index]) <= radius_hat:
            B_i_hat.append(client_index)

    print("points in B_i:", B_i)
    print("points in B_i_hat:", B_i_hat)
    
    return B_i, B_i_hat

########################################### the main rounding routine #############################################


'''
def k_center_rounding(x, data_points):
    
    # collect the indices of points whose x values are not zero
    s = []
    s = np.nonzero(x)
    s = list(s[0])

    print("initial set of centers:", s)

    B, B_hat = find_balls(data_points, s, r)

    for i in range(len(s)):
        points_in_reach = B[i]
        print("current center for mass:", s[i])
        print("mass for current center ball:", np.sum(x[points_in_reach]))

        if np.sum(x[points_in_reach]) < 1 - epsilon:
            s.remove(s[i])
            B_hat.remove(B_hat[i])
    
    print("updated set of centers:", s)
    
    ball_set = set(item for sublist in B_hat for item in sublist)

    print("points covered:", ball_set)

    while len(ball_set) < len(data_points):
        # find the clients that are not covered by the current set of centers
        uncovered = set(range(0, len(data_points))) - ball_set

        j = next(iter(uncovered))
        s.append(j)
        #ball_set.append(j)

        for center_index in s:
            if center_index != j and euclidean_distance(data_points[center_index], data_points[j]) <= 2 * r + delta * r:
                s.remove(center_index)

        # update the balls
        B, B_hat = find_balls(data_points, s, r)
        ball_set = set(item for sublist in B_hat for item in sublist)

    return s
'''


############################## main method for online positive-body chasing for k-center ##############################

def online_k_center(points, k, r):

    recourse = 0

    # initialize the vector x with all 0s of dimension len(points)
    x, candidate_index, candidates, constraint_mat = initialization(points)

    # store each client as a tuple (index in points, coordinates)
    clients = []
    # the coordinates of active clients
    client_points = []
    client_indices = []


    set_of_centers = []
    radius_of_centers = np.zeros(len(x))


    for t in range(10):

        x_old = copy.deepcopy(x)
        
        # at each iteration if the request is an insertion,
        # add new client to the set of points that are known 
        candidate_index = np.append(candidate_index, int(t))
        candidates = np.append(candidates, points[t])

        # variables for rounding
        current_client = (t, points[t])
        clients.append(current_client)
        client_points.append(points[t])
        client_indices.append(t)

        print("\n")
        
        print("time t:", t)

        print("current client:", points[t])

        s = find_candidates(candidate_index, points, points[t], r)
        # update the covering constraint matrix
        #constraint_mat = update_constraints(constraint_mat, s, len(points))

        # check if covering constraint is violated
        # if so, update the values of x's
        print("covering feasibility?", check_covering_feasibility(s, x))
        if check_covering_feasibility(s, x) == False:
            # covering constraint not satisfied, need to update values of x's in s
            x_new = update_covering_variables(x, s, epsilon)

        # check if packing constraint is violated
        # if so, update the values of x's
        if (np.sum(x) > (1+epsilon) * k):
            print("packing constraint violated!")
            x_new = update_packing_variables(x, epsilon, k)

        
        # update total recourse in l1-norm
        print("recourse in this round:", np.sum(np.abs(x_new - x_old)))

        recourse += np.sum(np.abs(x_new - x_old))
        x = x_new

        
        #################################### rounding procedure begins from here ####################################
        # integrate rounding at each round
        # maintain a set S of open centers at each t
        # calculate for each i in S it's radius = min(beta * OPT(t_i), diag)
        # OPT(t_i) is calculated using the offline algorithm

        # calculate OPT_offline for current active clients
        centers_offline = offline_k_center(client_points, k)
        current_OPT_dist = max_distance_to_centers(client_points, centers_offline)
        # diam = calculate_diameter(client_points)

        # find the points whose x value changed from 0 to 1 in this round
        # This means this point is added as a center
        # calculate its radius r(t, i) = min(beta * current_OPT_dist, Delta)
        '''
        for i in range(len(x)):
            if x_new[i] - x_old[i] == 1:
                set_of_centers.append(i)
                current_r = np.min(beta * current_OPT_dist, diam)
                radius_of_centers[i] = current_r
        '''
        
        # identify the balls/set of points that are B_i and B_i_hat for each i in set_of_centers
        # while building the balls B_i, drop any i from set_of_centers if it has mass less than 1-epsilon
        #list_of_B_i = []

        print("\n")
        print("Rounding begins")
        print("current set of centers:", set_of_centers)

        #list_of_B_i = []
        list_of_B_i_hat = []
        S = copy.deepcopy(set_of_centers)
        for center in S:
            
            print("\n")
            print("for center:", center)
            print("r_i of this center:", radius_of_centers[center])
            print("r_i_hat of this center:", alpha * min(beta * current_OPT_dist, diam))
            
            B_i, B_i_hat = find_balls(points, clients, center, radius_of_centers[center], alpha * min(beta * current_OPT_dist, diam))
            
            # drop any B_i whose mass is too small
            mass = 0
            for index_of_point in B_i:
                mass += x[index_of_point]
            
            print("mass for cent", mass)

            if mass < 1 - epsilon:
                set_of_centers.remove(center)
                print("center dropped from set")
            else:
                list_of_B_i_hat.append(B_i_hat)
        
        covered_points = set(item for sublist in list_of_B_i_hat for item in sublist)
        print("covered points:", covered_points)
        print("all clients:", client_indices)
        while len(covered_points) < len(client_indices):
        # find the clients that are not covered by the current set of centers
            uncovered = set(client_indices)- covered_points
            print("uncovered clients:", uncovered)

            j = next(iter(uncovered))
            set_of_centers.append(j)
            #ball_set.append(j)
            # record current_radius
            current_r = min(beta * current_OPT_dist, diam)
            radius_of_centers[j] = current_r

            print("new center {j} added to set", j)

            for center_index in set_of_centers:
                
                if center_index != j and euclidean_distance(points[center_index], points[j]) <= radius_of_centers[j] + radius_of_centers[center_index] + delta * min(radius_of_centers[j], radius_of_centers[center_index]):
                    set_of_centers.remove(center_index)
                    print("center {center_index} dropped", center_index)

            # update the the set of B_i_hat
            list_of_B_i_hat = []
            for center in set_of_centers:
                B_i, B_i_hat = find_balls(points, clients, center, radius_of_centers[center], alpha * min(beta * current_OPT_dist, diam))
                # ball_set = set(item for sublist in B_hat for item in sublist)
                list_of_B_i_hat.append(B_i_hat)
            covered_points = set(item for sublist in list_of_B_i_hat for item in sublist)
        print("selected centers for this round:", set_of_centers)

    return x, recourse, set_of_centers


#####################################################################################################################
################################################### main ############################################################
##################################################################################################################### 

fractional_sol, recourse, centers = online_k_center(data_points, k, r) 

print("\n")
print("final fractional solution:", fractional_sol)
print("number of centers:", np.sum(fractional_sol))
print("total recourse:", recourse)
print("final selected centers:", centers)






