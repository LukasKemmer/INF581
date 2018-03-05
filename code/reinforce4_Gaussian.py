from numpy import *
import time

    # not used:
def softmaxvec(Theta, obs):  # computes a vector of the softmax
    ''' The softmax activation function '''

    dotprod = zeros(Theta.shape[1])
    for k in range(Theta.shape[1]):
        dotprod[k] = dot(obs,Theta[:,k])
        
    correction = max(dotprod) - 6
    correction_vec = correction * ones(Theta.shape[1])
    prob = exp(dotprod - correction_vec)
    prob = prob/ sum(prob)
    return prob

def softmax_f(Theta, obs, allowed_actions):
    count_allowed_a = int(sum(allowed_actions))        
    prob = zeros(count_allowed_a)        
    actions_ind = zeros(count_allowed_a).astype(int)   # keep an array of all allowed actions
    dotproduct  = zeros(count_allowed_a)  # save an array with the computed dotproduct of theta and obs
    counter= 0
    count = 0
    for k in range(Theta.shape[1]):
        if allowed_actions[k]==1: 
            actions_ind[count] = k
            dotproduct[count] = dot(obs,Theta[:,k])
            count += 1

    correction = max(dotproduct) - 6 # choose a correction factor       
    correction_vec = correction * ones(count_allowed_a)
    
    exp_dot = exp(dotproduct - correction_vec)
    exp_dot = exp_dot / sum(exp_dot)
    
    prob = zeros(Theta.shape[1])
    prob[actions_ind] = exp_dot 
    #print("softmax_vec = ", prob)
    #print("allowed_actions = ", allowed_actions )
    #print("softmax_vec sum = ", sum(prob))
    return prob

            
    # not used:
def softmax(Theta, obs, index):  # computes an entry of the softmax corresponding to the action(index)
    ''' The softmax activation function '''
    sum_exp = 0
    dotprod = zeros(Theta.shape[1])
    for k in range(Theta.shape[1]):
        dotprod[k] = dot(obs,Theta[:,k])
        
    correction = max(dotprod) - 6
    
    for k in range(Theta.shape[1]):
        sum_exp += exp(dotprod[k] - correction )
    if sum_exp == float('+inf'):
        print("sum_exp is infinity !!!! ", sum_exp)
    return exp(dotprod[index] - correction )/ sum_exp

def phi_size(n_stores, type_of_phi, n_rbf = 3):
    # give back the size of phi, given which phi design you want:
    if type_of_phi == 1: # use squared stocks
        return n_stores+1+1
    elif type_of_phi == 2: # use rbf
        return n_rbf*(n_stores+1)+1
    elif type_of_phi == 0:
        return 1
    else: 
        print("Error, no valid type of phi was chosen")
        return 0
    
    
def RBF(s,c):
    sigma = 1
    phi = np.zeros(len(c))
    for i in range(len(c)):
        phi[i] = np.exp( - (np.linalg.norm( (s-c[i])**2, ord=2)/(2*sigma)))
    return phi

def RBF_vec(state , centers):
    sigma = 50    
    rbf_vec = zeros((centers.shape)) # shape of centers: nbr_rbf, n_stores+1
    for i in range(centers.shape[0]): # for every rbf 
        rbf_vec[i,:] = exp( - (state - centers[i,:])**2/(2*sigma))
    return hstack(rbf_vec[:,i] for i in range(rbf_vec.shape[1]))






class REINFORCE_agent_Gauss(object):
    '''
        A Policy-Search Method.
    '''    
    def compute_phi(self,obs,type_of_phi,n_stores, n_rbf = 3):
        obs[-n_stores:] = obs[-2*n_stores:-n_stores] - obs[-n_stores:]
        if type_of_phi == 1: # use squared stocks
            squarred = obs[:n_stores+1]**2
            return hstack(([1],obs,squarred))
        elif type_of_phi == 2: # use rbf
            statemin = zeros(self.env.n_stores+1)
            statemax = self.env.cap_store 
            statemid = (statemax + statemin)/2
            centers = array([statemin,statemid,statemax])
            rbf = RBF_vec(obs[:n_stores+1] , centers)
            #print("State = ", hstack(([1],obs,rbf)))
            return hstack(([1],obs,rbf))
        elif type_of_phi == 0:
            return hstack(([1],obs))
        else: 
            print("Error, no type of phi was chosen")
            return 0
    
    
    def allowed_actions(self):  # returns an array indicating which actions are allowed        
        a_allowed = zeros(self.dim_action)
        for i in range(self.dim_action):
            if self.possible_action(self.discrete2continuous[i,:]):
                a_allowed[i] = 1
                
        # warning message        
        if(sum(a_allowed)) <1:
            print("Warning: we have an action space of zero!!!!!!!!!")
           
        return a_allowed

    
    def choose_action_softmax(self, obs, allowed_actions, epsilon= 0):  # returns one of the allowed actions
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
                # print("allowed action is:", self.discrete2continuous[k])
        # epsilon greedy        
        if random.rand() < epsilon:
            action = random.choice(len(prob), size = None)
        else:        
            correction = max(dotproduct) - 6 # choose a correction factor       

            counter= 0
            for k in actions_ind:
                prob[counter]= exp(dotproduct[counter] - correction)
                counter +=1        
            if sum(prob) > 10000000 or sum(prob) < 0.0000001:
                print("Warning : sum_exp = ", sum_exp )
            prob = prob/sum(prob)
            
            # choose action according to probability vector   
            action = random.choice(len(prob), size = None, p = prob)
               
        return actions_ind[action]
    
    
    def choose_action_Gaussian(self, phi, allowed_actions, epsilon =0.5):
        count_allowed_a = int(sum(allowed_actions))                       
        dotproduct   = zeros(self.Theta.shape[1])  # save an array with the computed dotproduct of theta and obs
        
        # calculate the dotproduct for every dimension
        for k in range(self.Theta.shape[1]):  # for every action dimension
            dotproduct[k] = dot(phi,self.Theta[:-1,k])
        
        # calculate the indices of the allowed actions
        actions_ind = zeros(count_allowed_a).astype(int)   # keep an array of all allowed actions                
        counter= 0 
        for k in range(self.dim_action):    
            if allowed_actions[k]==1:
                actions_ind[counter]=k                             
                counter +=1   
                
        # print("allowed actions:", self.discrete2continuous[actions_ind,:])
        
        prob = ones(count_allowed_a)
        #softmax:
        if random.rand() < epsilon:
            action = random.choice(len(prob), size = None)
        else:  
            #correction = max(dotproduct) - 6 # choose a correction factor       
            # prob = Gauss_vec(Theta,  )
            prob = ones(count_allowed_a)
            for j in range(self.env.n_stores+1):
                prob *= (1/self.Theta[-1,j]) * exp(-(self.discrete2continuous[actions_ind,j] - dotproduct[j] *ones(count_allowed_a))**2/ (2 * self.Theta[-1,j]**2))            
            
            if sum(prob) > 10000000 or sum(prob) < 0.0000001:
                print("Warning : sum_exp = ", sum(prob) )
            
            prob = prob / sum(prob)      
        
            if self.output % int(self.output_freq/10) == 0 and self.t == self.max_steps-1:
                print("probabilities:" , prob)
        # choose action according to probability vector   
            action = random.choice(len(prob), size = None, p = prob)   
        return actions_ind[action]        
        
        
        
        
        
    def compute_gradient(self,phi,allowed_actions, action_vec):
        Gradient = zeros((self.Theta.shape))
        
        #compute some usefull terms
        ######################################        
        sigma = self.Theta[-1,:]
        count_allowed_a = int(sum(allowed_actions)) 
        # indices of allowed actions
        actions_ind = zeros(count_allowed_a).astype(int)               
        counter= 0 
        for k in range(self.dim_action):    
            if allowed_actions[k]==1:
                actions_ind[counter]=k                             
                counter +=1   
        
        # calculate the dotproduct for every dimension 
        dotproduct   = zeros(self.Theta.shape[1])
        for k in range(self.Theta.shape[1]):  # for every action dimension
            dotproduct[k] = dot(phi,self.Theta[:-1,k])                
        
        # probabilities unnormalized for each action 
        prob = ones(count_allowed_a)
        for j in range(self.env.n_stores+1):
            prob *= (1/sigma[j]) * exp(-(self.discrete2continuous[actions_ind,j] - dotproduct[j] *ones(count_allowed_a)**2)/ (2 * sigma[j]**2)) 
        
        
        for j in range(self.Theta.shape[1]): # for factory, wh1, wh2,...
            Gradient[:-1,j] = self.compute_Theta_gradient(phi,allowed_actions, action_vec, index = j,prob = prob,dotproduct = dotproduct,actions_ind = actions_ind,count_allowed_a = count_allowed_a , sigma = sigma )
            Gradient[-1,j] = self.compute_sigma_gradient(phi,allowed_actions, action_vec, index = j,prob = prob,dotproduct = dotproduct,actions_ind = actions_ind,count_allowed_a = count_allowed_a , sigma = sigma ) 
        
        #print("Gradient= ", Gradient)
        return Gradient
        
        
    def compute_Theta_gradient(self, phi ,allowed_actions, action_vec, index,prob,dotproduct,actions_ind ,count_allowed_a, sigma): # compute the gradient for one vector of Theta
        a_phi = action_vec[index] - dotproduct[index]  
        
        # additional factor in denominator
        kfactor = (self.discrete2continuous[actions_ind,index] - ones(count_allowed_a)*dotproduct[index])  / sigma[index]**2
        
        denominator = sum(prob)        
        nominator = sum(kfactor*prob)  
        
        Gradient = ((a_phi /  (sigma[index]**2))  - (nominator/denominator) )* phi
        return Gradient
        
    def compute_sigma_gradient(self, phi ,allowed_actions, action_vec, index,prob,dotproduct,actions_ind ,count_allowed_a, sigma):
        
        a_phi_squarred = (action_vec[index] - dotproduct[index])**2
        
        # term that comes from taking the derivative of the nominator:
        denominator = sum(prob)
        # additional factor 
        kfactor = (self.discrete2continuous[actions_ind,index] - ones(count_allowed_a)*dotproduct[index])**2  / sigma[index]**3
        nominator = sum(prob * kfactor)
        
        Gradient = a_phi_squarred / sigma[index]**3 - 2/ sigma[index] - (nominator/denominator)
        return Gradient
    
        
        
        
        
        
    def __init__(self, environment, actions_per_store, max_steps, output_freq = 100, type_of_phi = 0):         
        # The step size for the gradient
        self.alpha = 0.00001    # very sensitive parameter to choose
        self.alpha_for_sigma = 0.000001
        
        self.epsilon = 1

        self.type_of_phi = type_of_phi
        self.epsilon = 0  # epsilone -greedy (not used at the moment)
        self.t0 = time.time()  # time the algorithm!
        self.max_steps = max_steps  # steps when update is made
        self.env = environment
        
        self.dim_state = self.env.n_stores*3+1 + phi_size(self.env.n_stores, type_of_phi) 
        self.dim_action =  actions_per_store**(self.env.n_stores+1)  # number of actions to choose from
        
        # initialize weights (every action with same probability)
        self.Theta = zeros((self.dim_state+1 , self.env.n_stores+1 ))  # dim: [size_state + 1(sigma)] x [dim in action space]
        self.Theta[-1,:] = ones(self.env.n_stores+1) * 6
        # for softmax:
        # self.Theta = zeros((self.dim_state , self.dim_action ))
        
        
        
        self.output = 0  # print some output every couple of updates
        self.output_freq = output_freq
        
        # To store an episode
        self.episode_allowed_actions = zeros((self.max_steps+1,self.dim_action)) # for storing the allowed episodes
        self.episode = zeros((self.max_steps+1,1+self.dim_state+1)) # for storing (a,s,r) 
        self.t = 0                                   # for counting time steps
        
        # define a matrix that lists the possible actions for each store
        available_actions = zeros( ( actions_per_store ,self.env.n_stores+1 ))   
        available_actions[:,0] = [0,int(self.env.max_prod/2),self.env.max_prod]
        for i in range(self.env.n_stores):
            available_actions[:,i+1] = [0,self.env.cap_truck,self.env.cap_truck*2]
        
        # Discretize the action space: compute all action combinations
        self.discrete2continuous = []
        if self.env.n_stores == 3:
            for i in range(available_actions.shape[0]):
                for j in range(available_actions.shape[0]):
                    for k in range(available_actions.shape[0]):
                        for l in range(available_actions.shape[0]):
                            self.discrete2continuous.append( array([int(available_actions[l,0]), int(available_actions[i,1]), int(available_actions[j,2]), int(available_actions[k,3])]))
                        # We use the l for the a0 so we have then ordered by store action and then by production. So it matches the action space order
        elif self.env.n_stores == 2:
            for i in range(available_actions.shape[0]):
                for k in range(available_actions.shape[0]):
                    for l in range(available_actions.shape[0]):
                        self.discrete2continuous.append( array([int(available_actions[l,0]), int(available_actions[i,1]), int(available_actions[k,3])]))
                        
        elif self.env.n_stores == 1:
            for i in range(available_actions.shape[0]):
                    for l in range(available_actions.shape[0]):
                        self.discrete2continuous.append( array([int(available_actions[l,0]), int(available_actions[i,1])]))
        
        # change list to array:
        self.discrete2continuous = array(self.discrete2continuous)  # differnt actions are in lines, a for factory is in column 1
        return
        

    #   check if an action is allwoed
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
        """
            Act.
            Parameters
            ----------
            obs : numpy array
                the state observation
            Returns
            -------
            numpy array
                the action to take
        """
        
        # compute the new state!
        phi = self.compute_phi(obs, self.type_of_phi, self.env.n_stores)
        self.episode[self.t,1:self.dim_state+1] = phi   # set observations in log
      
        # save the allowed actions for that state        
        allowed_actions = self.allowed_actions()
        self.episode_allowed_actions[self.t,:] = allowed_actions
        
        # next time step
        self.t = self.t + 1
                
        # choose new action:
        action = int(self.choose_action_Gaussian(phi, allowed_actions))
        # for softmax:
        #action = int(self.choose_action_softmax( phi, allowed_actions, self.epsilon ))
        
        # Save some info to a episode
        self.episode[self.t-1,0] = action

        # Return the action to take
        return array(self.discrete2continuous[action])
    
    
    def update(self,state, action, reward, state_new, action_new):  # update the parameters of the agent
        
        # save reward in episode
        self.episode[self.t-2,-1] = reward
        
        # if we reach the end of an episode: update
        if self.t == self.max_steps+1:
            # compute the last reward:
            state_new, reward, done, info = self.env.step(action_new)

            self.episode[self.t-1,-1] = reward

            self.output += 1
            
            self.epsilon = self.epsilon * 0.999
            
            # change epsilon (not used right now)
            # self.epsilon = self.epsilon * 0.9995
            
            grad = zeros((self.Theta.shape))  # initialize empty gradient
            for ts in range(self.t-1):  
                Dt = sum(self.episode[ts:,-1])  # sum up all rewards
                action = int(self.episode[ts,0])
                action_vec = self.discrete2continuous[action,:]
                
                phi = self.episode[ts,1:-1]                      
                # compute the gradient for every time step
                gradient = self.compute_gradient(phi,self.episode_allowed_actions[ts,:], action_vec)                   
                for i in range(self.env.n_stores+1):   # for factory, wh1, wh2...
                    grad[:,i] = grad[:,i] + gradient[:,i] * Dt
                    
            for i in range(self.env.n_stores+1):
                self.Theta[:-1,i] = self.Theta[:-1,i] + self.alpha *  grad[:-1,i] 
                #update sigma not if too small 
                if (self.Theta[-1,i] +  self.alpha_for_sigma *  grad[-1,i])< 0.0001:
                    #self.Theta[-1,i] = 0.01
                    #self.Theta[-1,i] = self.Theta[-1,i]* 0.99995
                    print("Warning, we reached a very low sigma")
                elif (self.Theta[-1,i] +  self.alpha_for_sigma *  grad[-1,i])> 1000:
                    self.Theta[-1,i] = 15
                    print("Warning, we reached a very high sigma")
                    
                else: 
                    #self.Theta[-1,i] = self.Theta[-1,i] + self.alpha_for_sigma *  grad[-1,i]
                    # try decreasing the sigmas:
                    
                    self.Theta[-1,i] = self.Theta[-1,i]* 0.99995
            print("Sigma = ", self.Theta[-1,:])
            # print all desired output here
            if self.output % self.output_freq == 0:
                print("================Episode: ",self.output," ================")
                print("log :",self.episode )
                print("The sum of all gradient entries is: ", sum(sum(absolute(grad))))
                print("The final Theta is:", self.Theta)
                                
                #print("Theta 2 after is : ", self.Theta[:,2])
                print("Algorithm time: ", time.time()- self.t0, " seconds!")
                print("Algorithm time per episode: ", (time.time()- self.t0)/ self.output, " seconds!")
                print("=========================================================")
            
            
            # after episode, set everything to zero!
            self.t = 0
            self.episode_allowed_actions = zeros((self.max_steps+1,self.dim_action)) # for storing the allowed episodes
            self.episode = zeros((self.max_steps+1,1+self.dim_state+1)) # for storing (a,s,r) 
        
        return 
    def create_plots(self, rewards): # plot some useful information about the algorithm:
        print("========================Final Output=====================")
        #print("log :", self.episode)
        #print("The sum of all gradient entries is: ", sum(sum(absolute(grad))))
        #print("Theta 2 after is : ", self.Theta[:,2]) 
        s=[1 ,20, -8, 3 , 0]
        awarehouse = zeros(20)
        for i in range(len(awarehouse)):
            addvec = [0,0,1,0,0]
            s = s + i * addvec
            awarehouse[i] = dot(s,self.Theta[:-1,1])  
            fig10 = plt.figure(figsize=(10, 4), dpi=120)
            plt.plot(awarehouse, 'b', label='action warehouse1')
            plt.show()
        print("Algorithm time: ", time.time()- self.t0, " seconds!")
        print("Algorithm time per episode: ", (time.time()- self.t0)/ self.output, " seconds!")
        print("=========================================================")       
        return



