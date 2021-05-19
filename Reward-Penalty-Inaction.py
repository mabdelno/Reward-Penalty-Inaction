#Name: Mary Abdelnour
#Student number: 500648395
import numpy as np
import random
import math
import time

#random in terms of time
random.seed(time.time())

#Q_a = list of estimated probabilities of arms 
#q_a = actual probabilites
Q_a, q_a = np.empty(10), np.empty(10)

#resets the estimated probabilities given amount of arms
def initialize(arms): 
    Q_a = np.empty(arms)
    ip = 100/arms*0.01
    Q_a.fill(ip)
    return Q_a

#initializes actual probabilities of each arm and stores in q_a
def environment(arms):
    n = 1 #steps
    q_a = np.random.random((1,arms))[0]
    return q_a, arms

#returns whether there is a reward or not given which arm is pulled
def reward(percent): 
    if random.randrange(100)/100 < percent:
        return 1
    else:
        return 0

#function that increases probability of arm when achieving a reward
#and decreases probability of all other arms uniformly
def rewarding(Q_a, arms, arm, alpha = 0.1):
    for i in range(arms):
        if i != arm:
            Q_a[i] = (1 - alpha)*Q_a[i]
        else:
            Q_a[i] = Q_a[i] + alpha*(1-Q_a[i])
    return Q_a

#function that decreases probability of arm when no reward received
#and increases probability of all other arms uniformly
def penalty(Q_a, arms, arm, beta = 0.1):
    for i in range(arms):
        if i == arm:
            Q_a[i] = (1 - beta)*Q_a[i]
        else:
            Q_a[i] = (beta/(arms-1)) + (1-beta)*Q_a[i]
    return Q_a

#the action of pulling the arm
#randompick = random probability
#checks if randompick lies between interval of estimated probability
#chooses upper bound limit
def pullarm(Q_a, Q_cum, arms, arm):
    randompick = random.randrange(100)/100
    if 0 < randompick < Q_a[0]: 
        arm = 0
    else:
        Q_cum = Q_a[0]
    for i in range(1, arms):
        if Q_cum < randompick < (Q_cum + Q_a[i]):
            arm = i
            break
        else:
            Q_cum = Q_cum + Q_a[i]
            i += 1
    return arm

#pulls arm, checks for reward
#if reward is achieved, increase probability of reward with the arm
#and uniformly decrease reward probability of all other arms
#if reward is not achieved, decrease probability of reward with the arm
#and uniformly increase reward probability of all other arms
def reward_penalty(Q_a, q_a, arms, iters = 5000):
    Q_a = initialize(arms)
    n = 0
    arm = 0
    oa_count = 0
    Q_cum = 0
    #reward count
    rewards = 0
    for a in range(iters):
        arm = pullarm(Q_a, Q_cum, arms, arm)
        action_reward = reward(q_a[arm])
        if action_reward == 1:
            rewards += 1
            Q_a = rewarding(Q_a, arms, arm)
        else:
            Q_a = penalty(Q_a, arms, arm)
        n += 1
        if np.argmax(q_a) == arm:
            oa_count += 1
        if n%100 == 0:
            print("Times optimal action chosen: %d\n" % oa_count)
            rewardpercentages = (rewards/n)*100
            print("Average reward: %d\n" % rewardpercentages)
    #eoa = estimated optimal action
    eoa = np.argmax(Q_a) 
    #aoa = actual optimal oction
    aoa = np.argmax(q_a)


    return eoa, aoa, rewardpercentages

#pulls arm, checks for reward
#if reward is achieved, increase probability of reward with the arm
#and uniformly decrease reward probability of all other arms
#if reward is not achieved, do nothing (inaction)
def reward_inaction(Q_a, q_a, arms, iters = 5000):
    Q_a = initialize(arms)
    n = 0
    oa_count = 0
    arm = 0
    Q_cum = 0
    rewards = 0
    rewardpercentages = 0
    Q_a = np.empty(arms)
    for a in range(iters):
        arm = pullarm(Q_a, Q_cum, arms, arm)
        action_reward = reward(q_a[arm])
        if action_reward == 1:
            rewards += 1
            Q_a = rewarding(Q_a, arms, arm)
        n += 1
        if np.argmax(q_a) == arm:
            oa_count += 1
        
        #print average of reward percentages 
        #and number of times optimal action chosen
        if n%100 == 0:
            print("Times optimal action chosen: %d\n" % oa_count)
            rewardpercentages = (rewards/n)*100
            print("Average reward: %d\n" % rewardpercentages)
    #eoa = estimated optimal action
    eoa = np.argmax(Q_a) 
    #aoa = actual optimal oction
    aoa = np.argmax(q_a)

    return eoa, aoa, rewardpercentages

arms = 0
#total reward percentages
rp_ra, ri_ra = 0, 0
#current reward percentages
rp_rp, ri_rp = 0 ,0 
#count of number of times optimal action chosen correctly
rp_oa_chosen, ri_oa_chosen = 0, 0

for i in range(100):
    q_a, arms = environment(10)
    print("Environment #%d\n" % (i+1))
    print("Reward Penalty\n")
    rp_eoa, rp_aoa, rp_rp = reward_penalty(Q_a, q_a, arms)
    print("Reward Inaction\n")
    ri_eoa, ri_aoa, ri_rp = reward_inaction(Q_a, q_a, arms)
    ri_ra = ri_rp + ri_ra
    rp_ra = rp_rp + rp_ra

    if rp_eoa == rp_aoa:
        rp_oa_chosen += 1

    if ri_eoa == ri_aoa:
        ri_oa_chosen += 1
    print("----------------------\n")
    
rp_ra = rp_ra/100
ri_ra = ri_ra/100

print("Reward Penalty: ")
print("Total reward percentage average with 100 different environments: %d\n" % rp_ra)
print("Total times estimated optimal action = actual optimal action:%d\n" % rp_oa_chosen)

print("Reward Inaction: ")
print("Total reward percentage average with 100 different environments: %d\n" % ri_ra)
print("Total times estimated optimal action = actual optimal action: %d\n" % ri_oa_chosen)