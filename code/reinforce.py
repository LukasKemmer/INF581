from numpy import *
import time

def softmax_vec(z):  # computes the whole vector 
    ''' The softmax activation function '''
    sum_exp = sum(exp(z)) 
    
    # Keep some exploration open, by clipping the values going into the sigmoid    
    #a = clip(a, -3, 3)
    return exp(z) / sum_exp

def softmax(Theta, obs, index):  # computes an entry of the softmax corresponding to the action(index)
    ''' The softmax activation function '''
    sum_exp = 0
    for k in range(Theta.shape[1]):
        sum_exp += exp(dot(obs,Theta[:,k]) )
    if sum_exp == float('+inf'):
        print("sum_exp is infinity !!!! ", sum_exp)
    # Keep some exploration open, by clipping the values going into the sigmoid    
    #a = clip(a, -3, 3)
    # print("softmax = ",exp(dot(obs,Theta[:,index]))/ sum_exp)
    return exp(dot(obs,Theta[:,index]))/ sum_exp

    


class REINFORCE_agent(object):
    '''
        A Policy-Search Method.
    '''
    def allowed_actions(self):  # returns an array indicating which actions are allowed        
        a_allowed = zeros(self.dim_action)
<<<<<<< Updated upstream
        action_space = self.env.action_space().tolist()
        pointer1 = 0
        pointer2 = 0
        while pointer1 < len(action_space) and pointer2 < len(self.discrete2continuous):
            np1 = action_space[pointer1]
            np2 = self.discrete2continuous[pointer2]
            if array_equal(np1, np2):
                a_allowed[pointer2] = 1
                pointer1 += 1
                pointer2 += 1
            elif self.smaller_or_equals_for_action_space(np1, np2):
                pointer1 += 1
            else:
                pointer2 += 1

# TODO delete this code -- Droche 28/02
 #       for i in range(self.dim_action ):
  #          #if self.discrete2continuous[i]  in  action_space:
  #              #a_allowed[i]= 1
  #          for act in self.env.action_space():
  #              if (self.discrete2continuous[i] == act).all():
  #                  a_allowed[i]= 1
=======
        action_space = self.env.action_space()
        for i in range(self.dim_action ):
            #if self.discrete2continuous[i]  in  action_space:
                #a_allowed[i]= 1
            for act in action_space:
                if (self.discrete2continuous[i] == act).all():
                    a_allowed[i]= 1
                    
            #if action_space.__contains__(self.discrete2continuous[i]):
                #a_allowed[i]= 1
>>>>>>> Stashed changes
            #if a_allowed[i] == 0:
            #    print("not allowed action: ", self.discrete2continuous[i])
        if(sum(a_allowed)) <1:
            print("Warning we have a low action space!!!!!!!!!")
            print("action space: ", action_space)
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



    
    def choose_action(self, obs, allowed_actions):  # returns one of the allowed actions
        sum_exp = 0
        count_allowed_a = int(sum(allowed_actions))
        actions_ind = zeros(count_allowed_a)   # keep an array of all allowed actions
        prob = zeros(count_allowed_a)
        counter= 0
        for k in range(self.Theta.shape[1]):
            if allowed_actions[k]==1:
                actions_ind[counter]=k
                sum_exp += exp(dot(obs,self.Theta[:,k]))
                prob[counter]= exp(dot(obs,self.Theta[:,k]))
                counter +=1
        prob = prob/sum_exp
        #print("sum_exp = ", sum_exp )
        #print("probabilities = ", prob)
        #print("sum of probabilities = ", sum(prob))
        action = random.choice(len(prob), size = None, p = prob)
        return actions_ind[action]
                                       
    def __init__(self,environment, obs_space, action_dim, max_steps): # to do: dim_action/action_space calculation, discretization of action space, especially discrete2continous, set episode length, discrete2continous
        self.max_steps = max_steps
        self.env = environment
        self.dim_state = obs_space
        # self.dim_action = action_space.shape[0]      # has to be changed to output basically the number of stores
        self.dim_action = action_dim**(self.env.n_stores+1)  # number of actions to choose
        
        # set random weights
        self.Theta = random.randn(self.dim_state+1 , self.dim_action ) * 0.1  # create weightmatrix with columns as vectors for the individual softmax inputs

        # The step size for the gradient
        self.alpha = 0.0001    # has to definitely be updated

        # To store an episode
        self.episode_allowed_actions = zeros((self.max_steps+1,self.dim_action)) # for storing the allowed episodes
        self.episode = zeros((self.max_steps+1,1+self.dim_state+1)) # for storing (a,s,r) 
        self.t = 0                                   # for counting 
        
        available_actions = zeros((3,self.env.n_stores+1 ))   # define a matrix that lists the possible actions for each store
        available_actions[:,0] = [0,4,8]               # warehouse can produce more 0,5 or 10
        for i in range(self.env.n_stores):
            available_actions[:,i+1] = [0,1,2]          # shops can order 0,1 or 2 
        
        # Discretize the action space: compute all action combinations
        self.discrete2continuous = []
        # find other method to compute all permutations
        for i in range(available_actions.shape[0]):
            for j in range(available_actions.shape[0]):
                for k in range(available_actions.shape[0]):
                    for l in range(available_actions.shape[0]):
                        self.discrete2continuous.append( array([int(available_actions[l,0]), int(available_actions[i,1]), int(available_actions[j,2]), int(available_actions[k,3])]))
                        # We use the l for the a0 so we have then ordered by store action and then by production. So it matches the action space order
        print("number of actions: ", len(self.discrete2continuous))
        

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
        allowed_actions = self.allowed_actions()
        self.episode_allowed_actions[self.t,:] = allowed_actions

        
        self.t = self.t + 1
        
        x = ones(self.dim_state + 1)
        x[1:] = obs   
        
        # choose new action:
        action = int(self.choose_action( x, allowed_actions ))
        
        # Save some info to a episode
        self.episode[self.t-1,0] = action

        # Return the action to take
        return array(self.discrete2continuous[action])

    def __str__(self):
        ''' Return a string representation (e.g., a label) for this agent '''
        # This will appear as label when we click on the bug in ALife
        return ("RF. alpha=%3.2f" % (self.alpha))
    
    def update(self,state, action, reward, state_new, action_new):
        '''
            update function not required for q-s-policy
        '''
            
        self.episode[self.t-2,-1] = reward
        
        # End of episode ?
        if self.t == self.max_steps+1:
            print("Update:",self.episode )
            for ts in range(self.t-1):  
                Dt = sum(self.episode[ts:,-1])  # sum up all rewards
                #print("Dt = ", Dt)
                action = int(self.episode[ts,0])
                #print("action at ts=", ts,  " is ", action)
                x = ones(self.dim_state + 1)
                x[1:] = self.episode[ts,1:-1]
                for i in range(self.dim_action): # update every column in Theta individually
                    # Add the bias term (for our model)  
                    if i == action:  # different gradient for the weight of the action that was performed                         
                        grad = (1 - softmax(self.Theta,x,action,self.correction[i])) * x
                        #print("for i equal to j: grad = ", grad)
                    else:                        
                        grad = - softmax(self.Theta,x,action,self.correction[i]) * x
                        #print("for i not equal to j: grad = ", grad)
                    if self.episode_allowed_actions[ts,i] == 1:   # we don't update the policy if the action is not allowed
                        self.Theta[:,i] = self.Theta[:,i] + self.alpha *  grad  * Dt 
                        if dot(self.Theta[:,i],x ) > self.correction[i]:  # update the correction factor
                            self.correction[i] = dot(self.Theta[:,i],x )
                #print("Theta =", self.Theta)
            # after episode, set everything to zero!
            self.t = 0
            self.episode_allowed_actions = zeros((self.max_steps+1,self.dim_action)) # for storing the allowed episodes
            self.episode = zeros((self.max_steps+1,1+self.dim_state+1)) # for storing (a,s,r) 
            print("Algorithm time: ", time.time()- self.t0, " seconds!")
        
        return 


