import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import time


################################################
#build stochastic matrix
#question1 and question3
###############################################
def stochasticmatrix(matrix): 
    Hermit = 0
    for i in range(n):    #
        children = 0    #the number of children of the node/ the output degree
        for j in range(n):
            if matrix[i][j] == 1:
                children += 1
        if children != 0:    #dangling node
            #matrix[i] *= (1/children)
            for j in range(n):
                if matrix[i][j] == 1:
                    matrix[i][j] = (1/children)
        else:
            for j in range(n):    #normalization
                matrix[i][j] = (1/n)
            Hermit = 1

    matrix = np.matrix(matrix)    
    if Hermit == 1:
        matrix = matrix.H
    else:
        matrix = matrix.T     #transpose

    return matrix



###############################################
# using power method to iterate
#queston2
###############################################
def poewr_method(A,n,it):
    e = np.ones((n,1)) #column vector full with 1
    e_t = e.T
    M = d * stochasticmatrix(A) + (1-d)*((e*e_t)/n)    #calculate M
    #print("\n the M is:")
    #print(M)
    x = np.random.rand(n,1)        #random primal x
    #x = np.ones((n,1))
    #a = sum(x)
    #for i in range(n):
    #    x[i] = x[i] / a

    #print("\n the primal x is:")
    #print(x)

    for k in range(it):    #interate
        x = np.dot(M,x)

    # print("\n solve Mx=x, x is:")
    # print(x)
    # print('\n')

    # distance = 0
    # a = np.dot(M, x)
    # print("a is:")
    # print(a)
    # return abs(a.sum() - x.sum())



###############################################
# using power method to iterate
#question1
###############################################
def generate_graph(n):
    A = np.random.rand(n,n)
    for i in range(n):
        for j in range(n):
            if A[i][j] <= 0.5:
                A[i][j] = 0
            else:
                A[i][j] = 1
    '''
    print("A is:")
    print(A)
    print("\n")
    '''
    return A



############################################
# generate symmetric matrix
#question2
###########################################
def generate_graph_symmetric(n):
    #A = np.ones((n,n))
    valid = 0    
    while valid == 0: # if symmetrix valid == 0, then return
        valid = 1
        A = np.random.rand(n,n)
        for i in range(n):
            for j in range(n):
                if A[i][j] <= 0.5:
                    A[i][j] = 0
                else:
                    A[i][j] = 1

        for i in range(n):    #
            children = 0    #the number of children of the node/ the output degree
            for j in range(n):
                if A[i][j] == 1:
                    children += 1
            if children == 0:    #dangling node
                for j in range(n):    #normalization
                    A[i][j] = (1/n)

        A = np.triu(A)    # build diagonal matrix
        A += A.T - np.diag(A.diagonal())
        
        for i in range(n):
            A[i] /= sum(A[i])    #normaliztion

        for i in range(n):    #check diagonal or not
            for j in range(i,n):
                if A[i][j] != A[j][i]:
                    valid = 0
    # print("\n original matrix is:")
    # print(A)
    # print('\n')
    return A

############################################
#use power method to iterate in question2
############################################
def power_method_realmat(A,n):
    e = np.ones((n,1))
    e_t = e.T
    M = d * A + (1-d)*((e*e_t)/n)
    
    #M = np.triu(M)
    #M += M.T - np.diag(M.diagonal())
    # print("M is :")
    # print(M)
    # print('\n')
    x = np.random.rand(n,1)
    x /= sum(x)
    '''
    x = np.triu(x)
    x += x.T - np.diag(x.diagonal())
    '''
    #x = np.ones((n,n))
    #a = sum(x)
    #for i in range(n):
    #    x[i] = x[i] / a
    # print("\n the primal is:")
    # print(x)
    for k in range(200):
        x = np.dot(M,x)
    # print('\n')
    x /= sum(x)
    # print("\n the final is:")
    # print(x)


############################################
#triangle algotirhm
#question3
############################################
def triangle(A,n):
    # print("\n the input matrix A is:")
    # print(A)
    theta = 0.00001 
    e = np.ones((n,1))
    e_t = e.T
    M = d * stochasticmatrix(A) + (1-d)*((e*e_t)/n)
    # print("\n M is:")
    # print(M)
    M = M - np.eye(n) #solve Mx = x -> (M-I)x = 0
    e = np.ones((n,1))
    P = np.zeros((n,1)) # P is zero vector
    pivot = []
    index = 0    #index for pivot
    # print("\n M is:")
    # print(M)

    alfa = np.random.rand(n,1)    #primal guess
    alfa /= sum(alfa)
    P_prime = np.dot(M,alfa)     #primal P_primal
    number = float('inf')
    for i in range(n):    #find the closest vector
        if np.linalg.norm(P - M[:,i]) < number:
        #if np.dot((P - P_prime).T, M[:,i]) >= (1/2) * (math.pow(np.linalg.norm(P), 2) - math.pow(np.linalg.norm(P_prime), 2)):
            number = np.linalg.norm(P - M[:,i])
            pivot = M[:,i]
            index = i

    #print(np.linalg.norm(P - P_prime))
    #print(np.linalg.norm(P - pivot))
    count = 0    #interation times

    while np.linalg.norm(P - P_prime) > theta * np.linalg.norm(P - pivot):
        count += 1
        #number = float('inf')
        for i in range(n):
            if np.dot((P - P_prime).T, M[:,i]) >= (1/2) * (math.pow(np.linalg.norm(P), 2) - math.pow(np.linalg.norm(P_prime), 2)): 
            #if np.linalg.norm(P - M[:,i]) < number:
                #number = np.linalg.norm(P - M[:,i])
                pivot = M[:,i]
                index = i
                break

        al_star = (np.dot((P - P_prime).T, (pivot - P_prime)))/(math.pow(np.linalg.norm(pivot - P_prime), 2))    #build al_star
        #P_pp = (1 - al_star) * P_prime + al_star * pivot
        for i in range(n):    #update alfa
            if i != index:
                alfa[i] = (1 - al_star) * alfa[i]
            else:
                alfa[i] = (1 - al_star) * alfa[i] + al_star

        P_pp = np.dot(M , alfa)    
        P_prime = P_pp    #replace P_prime with P_pp

    #print(al_star)    
    #print(P_pp)
    a = np.dot(M + np.eye(n), alfa)
    # print("\nthe final result is:")
    # print(alfa)
    #
    # print("\n the number of interation is:")
    # print(count)

    # print("\n M * alfa is:")
    # print(a)
    #print(sum(np.dot(M + np.eye(n), alfa)))
    # distance = 0
    # for i in range(n):
    #     distance += abs(a[i][0] - alfa[i][0])
    #
    # # print("\n the error is:")
    # # print(distance)
    # return distance





############################################
#comparison Jacobi
############################################
def Jacobi(A,n,it):    #Jacobi algorithm
    e = np.ones((n,1))
    e_t = e.T
    M = d * stochasticmatrix(A) + (1-d)*((e*e_t)/n)
    M = M - np.eye(n)
    x = np.random.rand(n)    #primal guess
    x /= sum(x)
    y = np.zeros(n)    #intermedate value
    count = 0    #number of iteration
    while count < it:
        count += 1
        for i in range(n):
            temp = 0
            for j in range(n):
                if i != j:
                    temp =  temp - (x[j] * M[i,j])
            y[i] = temp / M[i,i]
        x = copy.deepcopy(y)

    # print(f"\n the result of {count} iteration is:")
    # print(y)
    # result = np.dot(M,x)
    # distance = 0
    # for i in range(n):
    #     distance += abs(result[0,i])
    # # print(f"the error of Jacobi iteration is {distance}")
    # return distance

############################################
#comparison Gauss-Seidel
############################################
def Gauss_Seidel(A,n,it):    #Gauss_Seidel algorithm
    e = np.ones((n,1))
    e_t = e.T
    M = d * stochasticmatrix(A) + (1-d)*((e*e_t)/n)

    M = M - np.eye(n)
    x = np.random.rand(n)    #primal guess
    x /= sum(x)
    y = np.zeros(n) #intermedate value
    count = 0    #number of iteration

    while count < it:
        count += 1
        for i in range(n):
            temp = 0
            for j in range(n):
                if i != j:
                    temp =  temp - (x[j] * M[i,j])
            x[i] = temp / M[i,i]

    # print(f"\n the result of {count} iteration is:")
    # print(x)
    # result = np.dot(M,x)
    # distance = 0
    # for i in range(n):
    #     distance += abs(result[0,i])
    #
    # # print(f"the error of Gauss_Sdidel is {distance}")
    # return distance





############################################
#comparison SOR
############################################
def SOR(A,n,it):
    w = 0.7    #relaxation factor
    e = np.ones((n,1))
    e_t = e.T
    M = d * stochasticmatrix(A) + (1-d)*((e*e_t)/n)

    M = M - np.eye(n)
    x = np.random.rand(n)    #primal guess
    x /= sum(x)
    y = np.zeros(n)    #intermediate value
    count = 0    #number of iteration

    while count < it:
        count += 1
        for i in range(n):
            temp = 0
            for j in range(n):
                if i != j:
                    temp =  temp - (x[j] * M[i,j])
            y[i] = (1 - w) * y[i] + w * (temp / M[i,i])
        x = copy.deepcopy(y)

    # print(f"\n the result of {count} iteration is:")
    # print(y)
    # result = np.dot(M,x)
    # distance = 0
    # for i in range(n):
    #     distance += abs(result[0,i])
    #
    # # print(f"the error of SOR iteration is {distance}")
    # return distance





############################################

############################################
n = 10
d = 0.85
iteration = 20
#a = np.matrix('1 1 0; 1 0 1; 0 0 0')

# print("\n############# This is question 1 #################\n")
# a = np.arange(1, 100, 2)
# #a = [1,10,20,30,40,50,60,70,80,90,100]
# error = []
# for i in range(len(a)):
#     sum = 0
#     for j in range(50):
#         A = generate_graph(n)
#         #B = copy.deepcopy(A)
#         b = poewr_method(copy.deepcopy(A), n, a[i])
#         print("\nthe return error is:")
#         print(b)
#         sum += b
#     print("\naverage error is:")
#     print(sum/50)
#     error.append(sum/50)
# print("\nerror is:")
# print(error)
# l1 = plt.plot(a, error, color = 'blue', label = 'linear line', marker = "*")
# plt.xlabel("iteration step")
# plt.ylabel("error")
# plt.title("iteration step vs error")
# #plt.legend(loc='upper right', labels = ['advanced algorithm', 'baseline algorithm'])
# plt.show()
# print("\n############# This is question 1 #################\n\n")
#
#
#
# print("\n############# This is question 2 #################\n")
# A_Real = generate_graph_symmetric(n)
# power_method_realmat(A_Real,n)
# print("\n############# This is question 2 #################\n")



print("\n############# This is question 3 #################\n\n")
# triangle(B, n)

iter = 50
time_now = 0
time_limit = 1
dim = 20
start_dim = dim
dim -= 1
time_array1 = np.array([])
error_array1 = np.array([])
time_array2 = np.array([])
error_array2 = np.array([])
time_array3 = np.array([])
error_array3 = np.array([])
time_array4 = np.array([])
error_array4 = np.array([])
var_array1 = np.array([])
var_array2 = np.array([])
var_array3 = np.array([])
var_array4 = np.array([])
log_array1 = np.array([])
log_array2 = np.array([])
log_array3 = np.array([])
log_array4 = np.array([])
error = 0
iter_num = 0
while time_now<=time_limit and iter_num<=20:
    iter_num += 1
    dim += 1
    time_tem1 = np.array([])
    time_tem2 = np.array([])
    time_tem3 = np.array([])
    time_tem4 = np.array([])
    error_tem1 = np.array([])
    error_tem2 = np.array([])
    error_tem3 = np.array([])
    error_tem4 = np.array([])
    for i in range(100):
        n = dim
        A = generate_graph(n)
        B = copy.deepcopy(A)
        time_start1 = time.time()
        # triangle(B, n)
        poewr_method(A, n, iteration)
        time_end1 = time.time()
        time_start2 = time.time()
        Jacobi(A,n,iteration)
        time_end2 = time.time()
        time_start3 = time.time()
        Gauss_Seidel(A,n,iteration)
        time_end3 = time.time()
        time_start4 = time.time()
        SOR(A,n,iteration)
        time_end4 = time.time()
        # print('time: ', time_end1-time_start1)
        time_tem1 = np.append(time_tem1, time_end1-time_start1)
        # error_tem1 = np.append(error_tem1, error1)
        time_tem2 = np.append(time_tem2, time_end2 - time_start2)
        # # error_tem2 = np.append(error_tem2, error2)
        time_tem3 = np.append(time_tem3, time_end3 - time_start3)
        # # error_tem3 = np.append(error_tem3, error3)
        time_tem4 = np.append(time_tem4, time_end4 - time_start4)
        # # error_tem4 = np.append(error_tem4, error4)
    print('dim: ', dim)
    time_array1 = np.append(time_array1, np.mean(time_tem1))
    var_array1 = np.append(var_array1, np.var(time_tem1))
    log_array1 = np.append(log_array1, np.mean(np.log(time_tem1)))
    # print('time_tem1: ', time_tem1)
    # error_array1 = np.append(error_array1, np.sum(error_tem1)/20)
    time_array2 = np.append(time_array2, np.mean(time_tem2))
    var_array2 = np.append(var_array2, np.var(time_tem2))
    log_array2 = np.append(log_array2, np.mean(np.log(time_tem2)))
    # # error_array2 = np.append(error_array2, np.sum(error_tem2)/20)
    time_array3 = np.append(time_array3, np.mean(time_tem3))
    var_array3 = np.append(var_array3, np.var(time_tem3))
    log_array3 = np.append(log_array3, np.mean(np.log(time_tem3)))
    # # error_array3 = np.append(error_array3, np.sum(error_tem3)/20)
    time_array4 = np.append(time_array4, np.mean(time_tem4))
    var_array4 = np.append(var_array4, np.var(time_tem4))
    log_array4 = np.append(log_array4, np.mean(np.log(time_tem4)))
    # # error_array4 = np.append(error_array4, np.sum(error_tem4)/20)
    # dim += 1
    time_now = np.mean(time_tem1)
plt.figure(1)
plt.plot(range(start_dim, len(time_array1)+start_dim), time_array1, 'r')
# plt.plot(range(start_dim, len(time_array1)+start_dim), time_array1, 'ro')
plt.plot(range(start_dim, len(time_array2)+start_dim), time_array2, 'g')
# # plt.plot(range(start_dim, len(time_array2)+start_dim), time_array2, 'go')
plt.plot(range(start_dim, len(time_array3)+start_dim), time_array3, 'b')
# # plt.plot(range(start_dim, len(time_array3)+start_dim), time_array3, 'bo')
plt.plot(range(start_dim, len(time_array4)+start_dim), time_array4, 'y')
# # plt.plot(range(start_dim, len(time_array4)+start_dim), time_array4, 'yo')
plt.xlabel('Matrix Size')
plt.ylabel('Running Time (S)')
plt.title('Running Time VS Matrix Size')
plt.figure(2)
plt.plot(range(start_dim, len(time_array1)+start_dim), var_array1, 'r')
plt.plot(range(start_dim, len(time_array2)+start_dim), var_array2, 'g')
plt.plot(range(start_dim, len(time_array3)+start_dim), var_array3, 'b')
plt.plot(range(start_dim, len(time_array4)+start_dim), var_array4, 'y')
# plt.plot(range(start_dim, len(time_array1)+start_dim), var_array, 'ro')
# plt.plot(range(start_dim, len(time_array2)+start_dim), error_array2, 'g')
# # plt.plot(range(start_dim, len(time_array2)+start_dim), error_array2, 'go')
# plt.plot(range(start_dim, len(time_array3)+start_dim), error_array3, 'b')
# # plt.plot(range(start_dim, len(time_array3)+start_dim), error_array3, 'bo')
# plt.plot(range(start_dim, len(time_array4)+start_dim), error_array4, 'y')
# # plt.plot(range(start_dim, len(time_array4)+start_dim), error_array4, 'yo')
plt.xlabel('Matrix Size')
plt.ylabel('Variance')
plt.title('Variance VS Matrix Size')
plt.figure(3)
plt.plot(range(start_dim, len(time_array1)+start_dim), log_array1, 'r')
# plt.plot(range(start_dim, len(time_array1)+start_dim), time_array1, 'ro')
plt.plot(range(start_dim, len(time_array2)+start_dim), log_array2, 'g')
# # plt.plot(range(start_dim, len(time_array2)+start_dim), time_array2, 'go')
plt.plot(range(start_dim, len(time_array3)+start_dim), log_array3, 'b')
# # plt.plot(range(start_dim, len(time_array3)+start_dim), time_array3, 'bo')
plt.plot(range(start_dim, len(time_array4)+start_dim), log_array4, 'y')
# # plt.plot(range(start_dim, len(time_array4)+start_dim), time_array4, 'yo')
plt.xlabel('Matrix Size')
plt.ylabel('log(Running Time (S))')
plt.title('log(Running Time) VS Matrix Size')
plt.show()




print("\n############# This is question 3 #################")



# print("\n############# This is Jacobi #################\n")
# Jacobi(A,n,iteration)
# print("\n############# This is Jacobi #################\n\n")
#
#
# print("\n############# This is Gauss_Seidel #################\n")
# Gauss_Seidel(A,n,iteration)
# print("\n############# This is Gauss_Seidel #################\n\n")
#
#
# print("\n############# This is SOR #################\n")
# SOR(A,n,iteration)
# print("\n############# This is SOR #################\n\n")



