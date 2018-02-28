from numpy import *
import time
import timeit

def softmax_vec(z):  # computes the whole vector 
    ''' The softmax activation function '''
    sum_exp = sum(exp(z)) 
    
    # Keep some exploration open, by clipping the values going into the sigmoid    
    #a = clip(a, -3, 3)
    return exp(z) / sum_exp

def softmax(Theta, obs, index):  # computes an entry of the softmax corresponding to the action(index)
    ''' The softmax activation function '''
    sum_exp = 0
    dotprod = zeros(Theta.shape[1])
    for k in range(Theta.shape[1]):
        dotprod[k] = dot(obs,Theta[:,k])
        
    correction = max(dotprod)
    
    for k in range(Theta.shape[1]):
        sum_exp += exp(dotprod[k] - correction )
    if sum_exp == float('+inf'):
        print("sum_exp is infinity !!!! ", sum_exp)
    return exp(dotprod[index] - correction )/ sum_exp

    


class REINFORCE_agent(object):
    '''
        A Policy-Search Method.
    '''
    def allowed_actions(self):  # returns an array indicating which actions are allowed        
        a_allowed = zeros(self.dim_action)
        for i in range(self.dim_action):
            if self.possible_action(self.discrete2continuous[i]):
                a_allowed[i] = 1
        #a_allowed = self.is_possible_action(self.discrete2continuous)



        #action_space = self.env.action_space().tolist()
        #pointer1 = 0
        #pointer2 = 0
        #while pointer1 < len(action_space) and pointer2 < len(self.discrete2continuous):
        #    np1 = action_space[pointer1]
        #    np2 = self.discrete2continuous[pointer2]
        #    if array_equal(np1, np2):
        #        a_allowed[pointer2] = 1
        #        pointer1 += 1
        #        pointer2 += 1
        #    elif self.smaller_or_equals_for_action_space(np1, np2):
         #       pointer1 += 1
         #   else:
         #
        #        pointer2 += 1

        #action_space = self.env.action_space()
        #for i in range(self.dim_action ):
            #if self.discrete2continuous[i]  in  action_space:
                #a_allowed[i]= 1
           # for act in action_space:
               # if (self.discrete2continuous[i] == act).all():
                   # a_allowed[i]= 1
                    
            #if action_space.__contains__(self.discrete2continuous[i]):
                #a_allowed[i]= 1
            #if a_allowed[i] == 0:
            #    print("not allowed action: ", self.discrete2continuous[i])
        #for i in range(len(a_allowed)):
        #    if a_allowed[i] == 0:
        #        print("Not allowed actions is: ", self.discrete2continuous[i])
        if(sum(a_allowed)) <1:
            print("Warning we have a low action space!!!!!!!!!")
    #       print("action space: ", action_space)
            print("my proposed actions", self.discrete2continuous)
           
        return a_allowed

    def smaller_or_equals_for_action_space(self, np1, np2):
        '''
        returns if the first np array is smaller than the second using [1:] order and then checking [0] because of the order in action space
        :param np1:
        :param np2:
        :return:
        '''
        if len(np1) != len(np2):
            print("ERROR! The two np array are not of the same size")
        for i in range(1, len(np1)):
            if np1[i] < np2[i]:
                return True
            if np1[i] > np2[i]:
                return False
        if np1[0] < np2[0]:
            return True
        if np1[0] > np2[0]:
            return False

        return True

    def choose_action(self, obs, allowed_actions, epsilon= 0.5):  # returns one of the allowed actions
        sum_exp = 0
        count_allowed_a = int(sum(allowed_actions))        
        prob = zeros(count_allowed_a)        
        actions_ind = zeros(count_allowed_a)   # keep an array of all allowed actions
        dotproduct   = zeros(count_allowed_a)  # save an array with the computed dotproduct of theta and obs
        counter= 0
        for k in range(self.Theta.shape[1]):
            if allowed_actions[k]==1:
                actions_ind[counter]=k             
                dotproduct[counter] = dot(obs,self.Theta[:,k])
                counter +=1
        # epsilon greedy        
        if random.rand() < epsilon:
            action = random.choice(len(prob), size = None)
        else:        
            correction = max(dotproduct)        

            counter= 0
            for k in actions_ind:
                prob[counter]= exp(dotproduct[counter] - correction)
                counter +=1        
            if sum(prob) > 10000000 or sum(prob) < 0.0000001:
                print("Warning : sum_exp = ", sum_exp )
            prob = prob/sum(prob)
            
                
            action = random.choice(len(prob), size = None, p = prob)
            if action != argmax(prob):
                print("Warning, not the action with the highest probability was chosen.")
                print(action, argmax(prob))
                
            # try argmax policy
            action = argmax(prob)
        return actions_ind[action]
                                       
    def __init__(self,environment, obs_space, action_dim, max_steps): # to do: dim_action/action_space calculation, discretization of action space, especially discrete2continous, set episode length, discrete2continous
        
        self.epsilon = 1
        self.t0 = time.time()
        self.timeforallowedact = 0
        self.max_steps = max_steps
        self.env = environment
        self.dim_state = obs_space
        # self.dim_action = action_space.shape[0]      # has to be changed to output basically the number of stores
        self.dim_action = action_dim**(self.env.n_stores+1)  # number of actions to choose
        
        # set random weights
        self.Theta = random.randn(self.dim_state+1 , self.dim_action ) * 0.1  # create weightmatrix with columns as vectors for the individual softmax inputs

        # The step size for the gradient
        self.alpha = 0.001    # has to definitely be updated

        # To store an episode
        self.episode_allowed_actions = zeros((self.max_steps+1,self.dim_action)) # for storing the allowed episodes
        self.episode = zeros((self.max_steps+1,1+self.dim_state+1)) # for storing (a,s,r) 
        self.t = 0                                   # for counting 
        
        available_actions = zeros((3,self.env.n_stores+1 ))   # define a matrix that lists the possible actions for each store
        available_actions[:,0] = [0,int(self.env.max_prod/2),self.env.max_prod]
        for i in range(self.env.n_stores):
            available_actions[:,i+1] = [0,self.env.cap_truck,self.env.cap_truck*2]
        
        # Discretize the action space: compute all action combinations
        self.discrete2continuous = []
        # find other method to compute all permutations
        for i in range(available_actions.shape[0]):
            for j in range(available_actions.shape[0]):
                for k in range(available_actions.shape[0]):
                    for l in range(available_actions.shape[0]):
                        self.discrete2continuous.append( array([int(available_actions[l,0]), int(available_actions[i,1]), int(available_actions[j,2]), int(available_actions[k,3])]))
                        # We use the l for the a0 so we have then ordered by store action and then by production. So it matches the action space order

  #      self.is_possible_action = vectorize(self.possible_action)

    def possible_action(self, action):
        if sum(action[1:]) > self.env.s[0]:
            return False
        if self.env.s[0] + action[0] - sum(action[1:]) > self.env.cap_store[0]:
            return False
        for i in range(1, len(action)):
            if self.env.cap_store[i] - self.env.s[i] < action[i]:
                return False
        return True

    def get_action(self,obs):
    #def get_action(self,obs,reward,action_space,done=False):
        """
            Act.

            Parameters
            ----------

            obs : numpy array
                the state observation
            reward : float
                the reward obtained in this state

            Returns
            -------

            numpy array
                the action to take
        """
        
        # Save some info to a episode
        self.episode[self.t,1:self.dim_state+1] = obs   # set observations in log

        
        # save the allowed actions for that state
        
        t0 = time.time()
        allowed_actions = self.allowed_actions()
        #allowed_actions =  self.is_possible_action(self.discrete2continuous)
        self.timeforallowedact += time.time()-t0

        self.episode_allowed_actions[self.t,:] = allowed_actions
        
        self.t = self.t + 1
        
        x = ones(self.dim_state + 1)
        x[1:] = obs   
        
        # choose new action:
        action = int(self.choose_action( x, allowed_actions, self.epsilon ))
        
        # Save some info to a episode
        self.episode[self.t-1,0] = action

        # Return the action to take
        return array(self.discrete2continuous[action])

    def __str__(self):
        ''' Return a string representation (e.g., a label) for this agent '''
        # This will appear as label when we click on the bug in ALife
        return ("RF. alpha=%3.2f" % (self.alpha))
    
    def update(self,state, action, reward, state_new, action_new):
        
        self.episode[self.t-2,-1] = reward
        
        # End of episode ?
        if self.t == self.max_steps+1:
            
            self.epsilon = self.epsilon * 0.999 # change epsilon parameter
            #print("Update:",self.episode )
            tupdate = time.time()
            grad = zeros(( self.dim_action, self.dim_state + 1 ))
            for ts in range(self.t-1):  
                Dt = sum(self.episode[ts:,-1])  # sum up all rewards
                action = int(self.episode[ts,0])
                
                x = ones(self.dim_state + 1)
                x[1:] = self.episode[ts,1:-1]
                softmaxvalue = softmax(self.Theta,x,action)
                for i in range(self.dim_action): # update every column in Theta individually
                    # Add the bias term (for our model)  
                    if i == action:  # different gradient for the weight of the action that was performed                         
                        grad[i,:] = grad[i,:] + (1 - softmaxvalue) * x
                        #print("for i equal to j: grad = ", grad)
                    else:                        
                        grad[i,:] = grad[i,:] - softmaxvalue * x
                        #print("for i not equal to j: grad = ", grad)
                    #if self.episode_allowed_actions[ts,i] == 1:   # we don't update the policy if the action is not allowed
            #print("Theta 1 before is : ", self.Theta[:,1])
            #print("gradientsum = ", sum(sum(grad)))
            for i in range(self.dim_action):
                self.Theta[:,i] = self.Theta[:,i] + self.alpha *  grad[i,:]  * Dt 
            #print("Theta 2 after is : ", self.Theta[:,2])
            # after episode, set everything to zero!
            self.t = 0
            self.episode_allowed_actions = zeros((self.max_steps+1,self.dim_action)) # for storing the allowed episodes
            self.episode = zeros((self.max_steps+1,1+self.dim_state+1)) # for storing (a,s,r) 
            #print("Algorithm time: ", time.time()- self.t0, " seconds!")
            #print("timeforallowedact = ", self.timeforallowedact)
            #print("time for update = ", time.time() - tupdate )
            self.timeforallowedact = 0
        
        return 


