import numpy as np
import math
import copy
import time
import matplotlib.pyplot as plt


def compute_error(x, M):
    return np.dot(M, x)-x

################################################
# build stochastic matrix
# question1 and question3
###############################################
def stochasticmatrix(matrix):
    Hermit = 0
    for i in range(n):  #
        children = 0  # the number of children of the node/ the output degree
        for j in range(n):
            if matrix[i][j] == 1:
                children += 1
        if children != 0:  # dangling node
            # matrix[i] *= (1/children)
            for j in range(n):
                if matrix[i][j] == 1:
                    matrix[i][j] = (1 / children)
        else:
            for j in range(n):  # normalization
                matrix[i][j] = (1 / n)
            Hermit = 1

    matrix = np.matrix(matrix)
    if Hermit == 1:
        matrix = matrix.H
    else:
        matrix = matrix.T  # transpose

    return matrix


###############################################
# using power method to iterate
# queston2
###############################################
def power_method(A, n, round):
    time_start = time.time()
    e = np.ones((n, 1))  # column vector full with 1
    e_t = e.T
    M = d * stochasticmatrix(A) + (1 - d) * ((e * e_t) / n)  # calculate M
    # print("\n the M is:")
    # print(M)
    x = np.random.rand(n, 1)  # random primal x
    # x = np.ones((n,1))
    # a = sum(x)
    # for i in range(n):
    #	x[i] = x[i] / a
    # print("\n the primal x is:")
    # print(x)
    error_array = np.array([])
    x_old = 0
    for k in range(round):  # interate
        x = np.dot(M, x)
        x /= (sum(x))
        # error = compute_error(x, M)
        # error_array = np.append(error_array, np.var(error))
        # error = x - x_old
        # error_array = np.append(error_array, np.mean(error))
        # error_array = np.append(error_array, np.var(x))
        # x_old = x
        a = np.dot(M, x)
        error_array = np.append(error_array, abs(a.sum() - x.sum()))
    # x /= (sum(x))
    # print("\n solve Mx=x, x is:")
    # print(x)
    # print('\n')
    time_end = time.time()
    # error = compute_error(x, M)
    # error_array = 0
    # a = np.dot(M, x)
    return time_end-time_start, error_array


###############################################
# using power method to iterate
# question1
###############################################
def generate_graph(n):
    A = np.random.rand(n, n)
    for i in range(n):
        for j in range(n):
            if A[i][j] <= 0.5:
                A[i][j] = 0
            else:
                A[i][j] = 1
    # print("A is:")
    # print(A)
    # print("\n")
    return A


############################################
# generate symmetric matrix
# question2
###########################################
def generate_graph_symmetric(n):
    # A = np.ones((n,n))
    valid = 0
    while valid == 0:  # if symmetrix valid == 0, then return
        valid = 1
        A = np.random.rand(n, n)
        for i in range(n):
            for j in range(n):
                if A[i][j] <= 0.5:
                    A[i][j] = 0
                else:
                    A[i][j] = 1

        for i in range(n):  #
            children = 0  # the number of children of the node/ the output degree
            for j in range(n):
                if A[i][j] == 1:
                    children += 1
            if children == 0:  # dangling node
                for j in range(n):  # normalization
                    A[i][j] = (1 / n)

        A = np.triu(A)  # build diagonal matrix
        A += A.T - np.diag(A.diagonal())

        for i in range(n):
            A[i] /= sum(A[i])  # normaliztion

        for i in range(n):  # check diagonal or not
            for j in range(i, n):
                if A[i][j] != A[j][i]:
                    valid = 0
    # print("\n original matrix is:")
    # print(A)
    # print('\n')
    return A


############################################
# use power method to iterate in question2
############################################
def power_method_realmat(A, n):
    time_start = time.time()
    e = np.ones((n, 1))
    e_t = e.T
    M = d * A + (1 - d) * ((e * e_t) / n)

    # M = np.triu(M)
    # M += M.T - np.diag(M.diagonal())
    # print("M is :")
    # print(M)
    # print('\n')
    x = np.random.rand(n, 1)
    x /= sum(x)
    '''
	x = np.triu(x)
	x += x.T - np.diag(x.diagonal())
	'''
    # x = np.ones((n,n))
    # a = sum(x)
    # for i in range(n):
    #	x[i] = x[i] / a
    # print("\n the primal is:")
    # print(x)
    for k in range(200):
        x = np.dot(M, x)
    # print('\n')
    x /= sum(x)
    # print("\n the final is:")
    # print(x)
    time_end = time.time()
    # error = compute_error(x, M)
    error = 0
    return time_end-time_start, np.mean(error)


############################################
# triangle algotirhm
# question3
############################################
def triangle(A, n):
    print("\n the input matrix A is:")
    print(A)
    theta = 0.00001
    e = np.ones((n, 1))
    e_t = e.T
    M = d * stochasticmatrix(A) + (1 - d) * ((e * e_t) / n)
    print("\n M is:")
    print(M)
    M = M - np.eye(n)  # solve Mx = x -> (M-I)x = 0
    e = np.ones((n, 1))
    P = np.zeros((n, 1))  # P is zero vector
    pivot = []
    index = 0  # index for pivot
    print("\n M is:")
    print(M)

    alfa = np.random.rand(n, 1)  # primal guess
    alfa /= sum(alfa)
    P_prime = np.dot(M, alfa)  # primal P_primal
    number = float('inf')
    for i in range(n):  # find the closest vector
        if np.linalg.norm(P - M[:, i]) < number:
            # if np.dot((P - P_prime).T, M[:,i]) >= (1/2) * (math.pow(np.linalg.norm(P), 2) - math.pow(np.linalg.norm(P_prime), 2)):
            number = np.linalg.norm(P - M[:, i])
            pivot = M[:, i]
            index = i

    # print(np.linalg.norm(P - P_prime))
    # print(np.linalg.norm(P - pivot))
    count = 0  # interation times

    while np.linalg.norm(P - P_prime) > theta * np.linalg.norm(P - pivot):
        count += 1
        # number = float('inf')
        for i in range(n):
            if np.dot((P - P_prime).T, M[:, i]) >= (1 / 2) * (
                    math.pow(np.linalg.norm(P), 2) - math.pow(np.linalg.norm(P_prime), 2)):
                # if np.linalg.norm(P - M[:,i]) < number:
                # number = np.linalg.norm(P - M[:,i])
                pivot = M[:, i]
                index = i
                break

        al_star = (np.dot((P - P_prime).T, (pivot - P_prime))) / (
            math.pow(np.linalg.norm(pivot - P_prime), 2))  # build al_star
        # P_pp = (1 - al_star) * P_prime + al_star * pivot
        for i in range(n):  # update alfa
            if i != index:
                alfa[i] = (1 - al_star) * alfa[i]
            else:
                alfa[i] = (1 - al_star) * alfa[i] + al_star

        P_pp = np.dot(M, alfa)
        P_prime = P_pp  # replace P_prime with P_pp

    # print(al_star)
    # print(P_pp)
    a = np.dot(M + np.eye(n), alfa)
    print("\nthe final result is:")
    print(alfa)

    print("\n the number of interation is:")
    print(count)

    print("\n M * alfa is:")
    print(a)
    # print(sum(np.dot(M + np.eye(n), alfa)))
    distance = 0
    for i in range(n):
        distance += abs(a[i][0] - alfa[i][0])

    print("\n the error is:")
    print(distance)


############################################
# comparison Jacobi
############################################
def Jacobi(A, n, it):  # Jacobi algorithm
    e = np.ones((n, 1))
    e_t = e.T
    M = d * stochasticmatrix(A) + (1 - d) * ((e * e_t) / n)

    M = M - np.eye(n)
    x = np.random.rand(n)  # primal guess
    x /= sum(x)
    y = np.zeros(n)  # intermedate value
    count = 0  # number of iteration

    while count < it:
        count += 1
        for i in range(n):
            temp = 0
            for j in range(n):
                if i != j:
                    temp = temp - (x[j] * M[i, j])
            y[i] = temp / M[i, i]
        x = copy.deepcopy(y)

    print(f"\n the result of {count} iteration is:")
    print(y)
    result = np.dot(M, x)
    distance = 0
    for i in range(n):
        distance += abs(result[0, i])

    print(f"the error of Jacobi iteration is {distance}")


############################################
# comparison Gauss-Seidel
############################################
def Gauss_Seidel(A, n, it):  # Gauss_Seidel algorithm
    e = np.ones((n, 1))
    e_t = e.T
    M = d * stochasticmatrix(A) + (1 - d) * ((e * e_t) / n)

    M = M - np.eye(n)
    x = np.random.rand(n)  # primal guess
    x /= sum(x)
    y = np.zeros(n)  # intermedate value
    count = 0  # number of iteration

    while count < it:
        count += 1
        for i in range(n):
            temp = 0
            for j in range(n):
                if i != j:
                    temp = temp - (x[j] * M[i, j])
            x[i] = temp / M[i, i]

    print(f"\n the result of {count} iteration is:")
    print(x)
    result = np.dot(M, x)
    distance = 0
    for i in range(n):
        distance += abs(result[0, i])

    print(f"the error of Gauss_Sdidel is {distance}")


############################################
# comparison SOR
############################################
def SOR(A, n, it):
    w = 0.7  # relaxation factor
    e = np.ones((n, 1))
    e_t = e.T
    M = d * stochasticmatrix(A) + (1 - d) * ((e * e_t) / n)

    M = M - np.eye(n)
    x = np.random.rand(n)  # primal guess
    x /= sum(x)
    y = np.zeros(n)  # intermediate value
    count = 0  # number of iteration

    while count < it:
        count += 1
        for i in range(n):
            temp = 0
            for j in range(n):
                if i != j:
                    temp = temp - (x[j] * M[i, j])
            y[i] = (1 - w) * y[i] + w * (temp / M[i, i])
        x = copy.deepcopy(y)

    print(f"\n the result of {count} iteration is:")
    print(y)
    result = np.dot(M, x)
    distance = 0
    for i in range(n):
        distance += abs(result[0, i])

    print(f"the error of SOR iteration is {distance}")


############################################

############################################
n = 5
d = 0.85
iteration = 20
# a = np.matrix('1 1 0; 1 0 1; 0 0 0')
G = np.array([[0, 1 / 2, 1 / 2, 0, 0, 0],
              [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
              [1 / 3, 1 / 3, 0, 0, 1 / 3, 0],
              [0, 0, 0, 0, 1 / 2, 1 / 2],
              [0, 0, 0, 1 / 2, 0, 1 / 2],
              [0, 0, 0, 1, 0, 0]])

print("\n############# This is question 1 #################\n")
# A = generate_graph(n)
# B = copy.deepcopy(A)
# power_method(A, n)

# Test for running time

time_array = np.array([])
error_array = np.array([])
matrix_size = 50 #50
iteration_num = 1
iter_round = 10
for i in range(iteration_num):
    A = generate_graph(matrix_size)
    B = copy.deepcopy(A)
    time_diff, error_array = power_method(A, matrix_size, iter_round)
    # print('time cost: ', time_diff)
    time_array = np.append(time_array, time_diff)
    # error_array = np.append(error_array, error)

# plt.figure(1)
# plt.plot(range(iteration_num), time_array, 'r')
# plt.plot(range(iteration_num), np.ones(iteration_num) * np.mean(time_array), 'y')
# plt.ylim(0.01, 0.03)
# plt.text(0.4 * iteration_num, 0.015, 'Power Method, Random Matrix')
# plt.text(0.4 * iteration_num, 0.014, 'mean: ' + str(np.mean(time_array)) + ' s')
# plt.text(0.4 * iteration_num, 0.013, 'variance: ' + str(np.var(time_array)))
# plt.xlabel('Iteration Number')
# plt.ylabel('Running Time (S)')
# plt.title('Running Time VS Iteration')

time_array = np.array([])
error_array2 = np.array([])
matrix_size = 50 #50
iteration_num = 1000
iter_round = 10
for i in range(iteration_num):
    A = generate_graph(matrix_size)
    B = copy.deepcopy(A)
    time_diff, error_array2 = power_method(A, matrix_size, iter_round)
    # print('time cost: ', time_diff)
    time_array = np.append(time_array, time_diff)
    # error_array2 = np.append(error_array2, error2)

time_array = np.array([])
error_array3 = np.array([])
matrix_size = 50 #50
iteration_num = 1000
iter_round = 10
for i in range(iteration_num):
    A = generate_graph(matrix_size)
    B = copy.deepcopy(A)
    time_diff, error_array3 = power_method(A, matrix_size, iter_round)
    # print('time cost: ', time_diff)
    time_array = np.append(time_array, time_diff)
    # error_array3 = np.append(error_array3, error3)

time_array = np.array([])
error_array4 = np.array([])
matrix_size = 50 #50
iteration_num = 1000
iter_round = 10
for i in range(iteration_num):
    A = generate_graph(matrix_size)
    B = copy.deepcopy(A)
    time_diff, error_array4 = power_method(A, matrix_size, iter_round)
    # print('time cost: ', time_diff)
    time_array = np.append(time_array, time_diff)
    # error_array4 = np.append(error_array4, error4)

time_array = np.array([])
error_array5 = np.array([])
matrix_size = 50 #50
iteration_num = 1000
iter_round = 10
for i in range(iteration_num):
    A = generate_graph(matrix_size)
    B = copy.deepcopy(A)
    time_diff, error_array5 = power_method(A, matrix_size, iter_round)
    # print('time cost: ', time_diff)
    time_array = np.append(time_array, time_diff)
    # error_array5 = np.append(error_array5, error5)

plt.figure(2)
plt.plot(range(iter_round), error_array, 'b')
plt.plot(range(iter_round), error_array, 'bo')
plt.plot(range(iter_round), error_array2, 'r')
plt.plot(range(iter_round), error_array2, 'ro')
plt.plot(range(iter_round), error_array3, 'y')
plt.plot(range(iter_round), error_array3, 'yo')
plt.plot(range(iter_round), error_array4, 'g')
plt.plot(range(iter_round), error_array4, 'go')
plt.plot(range(iter_round), error_array5, 'm')
plt.plot(range(iter_round), error_array5, 'mo')
plt.xlabel('Iteration Number')
plt.ylabel('Error')
plt.title('Error VS Iteration')
plt.show()

print('average time cost: ', np.mean(time_array))

# # Test for matrix size
# time_array = np.array([])
# time_mean = np.array([])
# time_var = np.array([])
# error_array = np.array([])
# error_mean = np.array([])
# iteration_num = 100
# matrix_size = 5
# matrix_size_limit = 200
# start_size = matrix_size
# time_limit = 0.1 #1
# time_now = 0
# size_step = 1
# # while time_now <= time_limit:
# while matrix_size <= matrix_size_limit:
#     print('matrix size: ', matrix_size)
#     time_array = np.array([])
#     error_array = np.array([])
#     for i in range(iteration_num):
#         A = generate_graph(matrix_size)
#         B = copy.deepcopy(A)
#         time_diff, error = power_method(A, matrix_size)
#         # print('time cost: ', time_end - time_start)
#         time_array = np.append(time_array, time_diff)
#         error_array = np.append(error_array, error)
#     time_mean = np.append(time_mean, np.mean(time_array))
#     time_var = np.append(time_var, np.mean(time_array))
#     error_mean = np.append(error_mean, np.mean(error_array))
#     matrix_size += size_step
#     time_now = np.mean(time_array)
#
# plt.figure(2)
# plt.plot(range(start_size, matrix_size, size_step), time_mean, 'r')
# plt.ylim(0, time_limit)
# plt.xlabel('Matrix Size')
# plt.ylabel('Average Running Time (S)')
# plt.title('Running Time VS Matrix Size')
# plt.show()

print("\n############# This is question 1 #################\n\n")

# print("\n############# This is question 2 #################\n")
# A_Real = generate_graph_symmetric(matrix_size)
# power_method_realmat(A_Real, matrix_size)
#
# time_array = np.array([])
# error_array = np.array([])
# # iteration_num = 100
# for i in range(iteration_num):
#     A_Real = generate_graph_symmetric(matrix_size)
#     time_diff, error = power_method_realmat(A_Real, matrix_size)
#     # print('time cost: ', time_diff)
#     time_array = np.append(time_array, time_diff)
#     error_array = np.append(error_array, error)
#
# plt.figure(1)
# plt.plot(range(iteration_num), time_array, 'g')
# plt.plot(range(iteration_num), np.ones(iteration_num) * np.mean(time_array), 'b')
# plt.text(0.4 * iteration_num, 0.004, 'Power Method, General Real Matrix')
# plt.text(0.4 * iteration_num, 0.003, 'mean: ' + str(np.mean(time_array)) + ' s')
# plt.text(0.4 * iteration_num, 0.002, 'variance: ' + str(np.var(time_array)))
#
# plt.figure(2)
# plt.plot(range(iteration_num), error_array, 'g')
#
# plt.show()
#
# print('average time cost: ', np.mean(time_array))

# Test for matrix size
# time_array = np.array([])
# time_mean = np.array([])
# time_var = np.array([])
# error_array = np.array([])
# error_mean = np.array([])
# time_now = 0
# size_step = 1
# matrix_size = 5
# while time_now <= time_limit:
#     print('matrix size: ', matrix_size)
#     time_array = np.array([])
#     error_array = np.array([])
#     for i in range(iteration_num):
#         A_Real = generate_graph_symmetric(matrix_size)
#         time_diff, error = power_method_realmat(A_Real, matrix_size)
#         # print('time cost: ', time_end - time_start)
#         time_array = np.append(time_array, time_diff)
#         error_array = np.append(error_array, error)
#     time_mean = np.append(time_mean, np.mean(time_array))
#     time_var = np.append(time_var, np.mean(time_array))
#     error_mean = np.append(error_mean, np.mean(error_array))
#     matrix_size += size_step
#     time_now = np.mean(time_array)
#
# plt.figure(3)
# plt.plot(range(start_size, matrix_size, size_step), time_mean, 'r')
# plt.ylim(0, time_limit)
# plt.xlabel('Matrix Size')
# plt.ylabel('Average Running Time (S)')
# plt.title('Running Time VS Matrix Size')
# plt.show()

# print("\n############# This is question 2 #################\n")

# print("\n############# This is question 3 #################\n\n")
# triangle(B, n)
# print("\n############# This is question 3 #################")
#
# print("\n############# This is Jacobi #################\n")
# Jacobi(A, n, iteration)
# print("\n############# This is Jacobi #################\n\n")
#
# print("\n############# This is Gauss_Seidel #################\n")
# Gauss_Seidel(A, n, iteration)
# print("\n############# This is Gauss_Seidel #################\n\n")
#
# print("\n############# This is SOR #################\n")
# SOR(A, n, iteration)
# print("\n############# This is SOR #################\n\n")
