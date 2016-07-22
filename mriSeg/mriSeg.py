#!/usr/local/bin

"""
A package for brain MRI segmentation
"""

__author__ = "Hao Wang"


"""
MRI class    
"""
from scipy.stats import norm
import numpy as np

class MRI:

    def __init__(self, mri, seg=None, L=4):

        self._mri = np.copy(mri[:,:,0]) # get pixel intensity
        if seg is not None:
            self._seg = np.copy(seg[:,:,0]) # get ground truth
            self.to_label() # turn ground truth into labels
        else:
            self._seg = None
        
        self.L = L # number of classes
        self.mask = (self._mri!=1) # roughly filter out the background
        
        self._bias_corr = False # if bias correction is done
        self._mri_bc = None # corrected intensity
        self.mri_bc = None # corrected png
        self._bias_fd = None # estimated bias field
        self._predict_gs = None # gibbs sampler prediction
        self._predict_bp = None # belief progation prediction
        
        # gibbs distribution parameters
        self._mu = None 
        self._sigma = None 
        
        self._nrow = mri.shape[0]
        self._ncol = mri.shape[1]
    
    def get_intensity(self):
        return np.copy(self._mri)
    
    def get_label(self):
        return np.copy(self._seg)
    
    def get_intensity_bc(self):
        return np.copy(self._mri_bc)
    
    def get_mri_bc(self):
        return np.copy(self.mri_bc)
    
    def get_bias_field(self):
        return np.copy(self._bias_fd)
    
    def get_prediction_gs(self):
        return np.copy(self._predict_gs)
    
    def get_prediction_bp(self):
        return np.copy(self._predict_bp)
    
    
    """
    Bias Correction
    """
    def correct_bias(self, p_sub, K, spacing, Lambda,
                     th_em, th_c, verbose=False):
        """
        p_sub is the percentage of pixels for training
        K is the number of GMM components
        number of interval is for Bspline 
        Lambda controls the penalized Bspline
        th_em and th_c are thresholds for EM and updating c
        verbose controls if the training information is printed
        """
        
        # get log intensity
        m = np.copy(self._mri)
        d = np.log(m).ravel()
        
        # sub sampling 
        sub = np.random.choice(2, len(d), p=[1-p_sub, p_sub])
        sub = (sub==1)
        d_sub = d[sub]
        d_bs = np.arange(len(d))[sub]

        # initialize the cubic bspline
        nrow = self._nrow
        ncol = self._ncol
        x_bs = d_bs%ncol
        y_bs = d_bs/ncol+1
        bs_x = Cubic_Bspline(0, ncol-1, ncol/spacing)
        bs_y = Cubic_Bspline(0, nrow-1, nrow/spacing)
        Phi_x = np.array([bs_x.basis(i) for i in x_bs])
        Phi_y = np.array([bs_y.basis(i) for i in y_bs])
        Psi_x = bs_x.penalty_matrix()
        Psi_y = bs_y.penalty_matrix()
        
        # construct bicubic bspline
        c = np.zeros(Psi_x.shape[0]*Psi_y.shape[0])
        
        # new Phi
        Phi_xy = np.zeros(Phi_x.shape[1]*\
                          Phi_y.shape[1]*\
                          Phi_x.shape[0]).reshape(Phi_x.shape[1],
                                                  Phi_y.shape[1],
                                                  Phi_x.shape[0])

        for i in range(Phi_x.shape[1]):
            for j in range(Phi_y.shape[1]):
                Phi_xy[i,j] = Phi_x[:,i]*Phi_y[:,j]

        Phi_xy = Phi_xy.swapaxes(0,2).swapaxes(1,2).reshape(Phi_x.shape[0],
                                                            Phi_x.shape[1]*\
                                                            Phi_y.shape[1])
        
        # new Psi
        Psi_xy = np.zeros(Phi_x.shape[1]*\
                          Phi_y.shape[1]*\
                          Phi_x.shape[1]*\
                          Phi_y.shape[1]).reshape(Phi_x.shape[1],
                                                 Phi_y.shape[1],
                                                 Phi_x.shape[1],
                                                 Phi_y.shape[1])

        for i in range(Phi_x.shape[1]):
            for j in range(Phi_y.shape[1]):
                for k in range(Phi_x.shape[1]):
                    Psi_xy[i,j,k] = Psi_x[i,k]+Psi_y[j]

        Psi_xy = Psi_xy.reshape(Phi_x.shape[1],
                                Phi_y.shape[1],
                                Phi_x.shape[1]*\
                                Phi_y.shape[1]).reshape(Phi_x.shape[1]*
                                                        Phi_y.shape[1],
                                                        Phi_x.shape[1]*\
                                                        Phi_y.shape[1])
    
        # initialize other parameters
        _min = np.min(d_sub)
        _max = np.max(d_sub)
        h = (_max - _min)*1.0/K
        pi_k = np.ones(K)*1.0/K
        mu_k = np.array([_min + i*h for i in range(K)])
        sigma_k = np.ones(K)*(h)**2
        
        # bias correction
        cost = -np.inf
        c_rounds = 1
        
        while True:
            rounds = 1
            while True:
                # EM
                w = self.bc_update_w(d_sub, Phi_xy, c,
                                     mu_k, sigma_k, pi_k)
                mu_k, sigma_k, pi_k = \
                                     self.bc_maximize(w, d_sub, Phi_xy, c)
                cost_new = self.bc_cost(w, d_sub, Phi_xy,
                                        mu_k, sigma_k, pi_k, 
                                        c, Lambda, Psi_xy, regu=True)

                if cost_new - cost > th_em:
                    if verbose and rounds%1000==0: 
                        print 'EM round: ', rounds
                        print 'cost: ', cost_new
                    rounds+=1
                    cost = cost_new
                else:
                    if verbose: 
                        print 'EM finished. Total rounds: ', rounds
                        print 'cost: ', cost_new
                    break
            
            # update c        
            c = self.bc_update_c(Phi_xy, Lambda, Psi_xy, d_sub,
                                 w, mu_k, sigma_k)        
            cost_new = self.bc_cost(w, d_sub, Phi_xy, mu_k,
                                    sigma_k, pi_k, 
                                    c, Lambda, Psi_xy, regu=True)
            if cost_new - cost > th_c:
                if verbose and c_rounds%1000==0:
                    print 'C update round: ', c_rounds
                    print 'cost: ', cost_new
                c_rounds+=1
                cost = cost_new
            else:
                if verbose: 
                    print "All finished. Total rounds: ", c_rounds
                    print 'cost: ', cost_new
                break
        
        # evaluate bspline basis for all pixels
        x = np.arange(len(d))%ncol
        y = np.arange(len(d))/ncol+1
        Phi_x = np.array([bs_x.basis(i) for i in x])
        Phi_y = np.array([bs_y.basis(i) for i in y])
        
        # new Phi
        Phi = np.zeros(Phi_x.shape[1]*\
                       Phi_y.shape[1]*\
                       Phi_x.shape[0]).reshape(Phi_x.shape[1],
                                               Phi_y.shape[1],
                                               Phi_x.shape[0])
            
        for i in range(Phi_x.shape[1]):
            for j in range(Phi_y.shape[1]):
                Phi[i,j] = Phi_x[:,i]*Phi_y[:,j]

        Phi = Phi.swapaxes(0,2).swapaxes(1,2).reshape(Phi_x.shape[0],
                                                      Phi_x.shape[1]*\
                                                      Phi_y.shape[1])

        
        # update bias corrected image information
        self._bias_corr = True 
        log_bias = np.dot(Phi,c)
        log_m_bc = d - log_bias
        
        # estimated bias field
        bias_fd = np.exp(log_bias).reshape(self._nrow, self._ncol)
        bias_fd[np.invert(self.mask)] = np.nan
        self._bias_fd = bias_fd
        
        # corrected intensity
        m_bc = np.exp(log_m_bc)
        m_bc[m_bc >255] = 255
        _mri_bc = m_bc.reshape(self._nrow, self._ncol)
        self._mri_bc = _mri_bc.astype('uint8')
        
        # corrected png
        temp = np.copy(self._mri_bc)
        temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
        mri_bc = np.append(temp, temp, axis=2)
        mri_bc = np.append(mri_bc, temp, axis=2)
        self.mri_bc = mri_bc
    
    
    """
    MRF with Gibbs Sampler 
    """
    def mrf_gs(self, C=2, c1=80, c2=1, beta=1, delta=[1,-1], 
               iteration=100, verbose=False):
        """ 
        segmentation with Gibbs Sampler
        T(t) = C/log(t+1)
        alpha(t) = c1*0.9**t+c2
        delta = [a, b] so that:
            delta(ys!=yt) = a
            delta(ys==yt) = b
        """

        # if bias correction is performed, use the 
        # corrected intensity
        if self._bias_corr:
            m = np.copy(self._mri_bc)
        else:
            m = np.copy(self._mri)
        
        # initialize random segmentation
        L = self.L
        y = np.random.choice(L, self._nrow*self._ncol)
        y = y.reshape(self._nrow, self._ncol)
        
        # EM
        t = 1
        T = C/np.log(t+1)
        alpha = c1*0.9**t+c2
        
        while t <= iteration:
            if verbose and t%10==0: 
                print "MRF Gibbs Sampling, round: ", t
                print 'class means: ', mu_l
                print 'class variances: ', sigma_l
                
            # E-step
            mu_l = np.array([m[y==l].sum()*1.0/len(y[y==l]) \
                             for l in range(L)])
            sigma_l = np.array([((m[y==l]-mu_l[l])**2).sum()\
                                /len(y[y==l]) \
                               for l in range(L)])

            # M-step
            for row in range(self._nrow):
                for col in range(self._ncol):
                    y_i = y[row, col]
                    x_i = m[row, col]
                    
                    # get neighbors
                    nbs = np.array([y[max(row-1, 0), col], 
                                    y[min(row+1, self._nrow-1), col],
                                    y[row, max(col-1, 0)],
                                    y[row, min(col+1, self._ncol-1)]])
                    # compute sum of delta functions
                    nbs = sum([delta[i] for i in (nbs==y_i)])

                    # calculate sampling distribution
                    sigma_l[sigma_l==0] = 1
                    energy = (1/T)*\
                             (beta*nbs+alpha*(x_i-mu_l)**2/2/sigma_l\
                             +alpha*np.log(np.sqrt(sigma_l)))

                    # avoid exponentiate large negative number
                    with np.errstate(over='raise'):
                        pr = np.exp(-energy)/np.exp(-energy).sum()
                    
                    # make sure probabilities sum to 1 and are non-negative
                    pr[-1] = 1-pr[:-1].sum()
                    pr[pr<0] = 0  

                    # sample y_i by pr
                    y[row, col] = np.random.choice(L,1,p=pr)
            
            # update T, t, alpha
            t+=1
            T = C/np.log(t+1)
            alpha = c1*0.9**t+c2
            
        # reassign labels so that the class means match the greyscale scheme
        l_order = mu_l.argsort()
        for i in range(L):
            y[y==l_order[i]] = -i-1
        y = -y-1
        
        # update Gibbs Sampler prediction and parameters
        self._predict_gs = y
        self._mu = mu_l[l_order]
        self._sigma = sigma_l[l_order]
        
        print "MRF Gibbs Sampling finished."
    
    
    """
    Belief Propagation
    """
    def mrf_bp(self, C=2, c1=80, c2=1, beta=1, delta=[1,-1], 
           iteration=50, verbose=False):
        """
        segmentation with Belief Propagation using
        the parameters estimated through Gibbs Sampling
        """
        
        # if bias correction is performed, use the 
        # corrected intensity
        if self._bias_corr:
            m = np.copy(self._mri_bc)
        else:
            m = np.copy(self._mri)
            
        # init an np.array of shape (nrow,ncol,n_neigbhor,n_label)
        # to store the message from pixel(i,j) to its neighbors
        # for the 3rd dimension, the corresponding neighbor is:
        # 0: up, 1: down, 2: left, 3: right
        L = self.L
        msg = np.full((self._nrow, self._ncol, 4, L), 1.0/L)
        
        # gibbs distribution parameters 
        mu_l = self._mu
        sigma_l = self._sigma
        
        # update messages between pixels following the order:
        # up, down, left, right
        t = 1
        T = C/np.log(t+1)
        alpha = c1*0.9**t+c2
        
        while t <= iteration:
            if verbose and t%10==0: 
                print "MRF Belief Propagation, round: ", t
            
            for row in range(self._nrow):
                for col in range(self._ncol):
                    # update outgoing messages from pixel(row,col)
                    x_i = m[row,col]
                    
                    # up
                    if row>0:
                        self.update_msg(0,1,x_i,alpha,beta,delta,
                                        mu_l,sigma_l,T,row,col,
                                        row-1,col,msg)
                    # down
                    if row<self._nrow-1:
                        self.update_msg(1,0,x_i,alpha,beta,delta,
                                        mu_l,sigma_l,T,row,col,
                                        row+1,col,msg)
                    # left
                    if col>0:
                        self.update_msg(2,3,x_i,alpha,beta,delta,
                                        mu_l,sigma_l,T,row,col,
                                        row,col-1,msg)
                    # right
                    if col<self._ncol-1:
                        self.update_msg(3,2,x_i,alpha,beta,delta,
                                        mu_l,sigma_l,T,row,col,
                                        row,col+1,msg)  
            
            # update gibbs parameters
            T = C/np.log(t+1)
            alpha = c1*0.9**t+c2
            t+=1
        
        print "BP message updating finished."
        
        # generate prediction with messages
        y = np.copy(self._seg)
        
        for row in range(self._nrow):
            for col in range(self._ncol):
                x_i = m[row,col]
                self.bp_predict(x_i,mu_l,sigma_l,T,
                                alpha,row,col,msg,y)
        
        print "BP prediction generated."
            
        # update Belief Propagation prediction
        self._predict_bp = y
    

    
    """
    Evaluation
    """
    def evaluate(self, method='gs', coef='tanimoto'):
        """
        evaluate given prediction with specified method,
        tanimoto coefficient: 
        (A intersect B)/(A + B - (A intersect B))
        or dice coefficient:
        2*(A intersect B)/(A + B)
        """
        L = self.L
        
        if method == 'bp':
            pred = np.copy(self._predict_bp)
        else:
            pred = np.copy(self._predict_gs)
        
        # get (A intersect B) and (A+B) for each class
        intersect = np.array([np.all([self._seg.ravel()==l, 
                                      pred.ravel()==l], axis=0).sum()\
                                      for l in range(L)])
        union = np.array([(self._seg==l).sum()+(pred==l).sum()\
                         for l in range(L)])
        
        if coef == 'dice':
            return 2.0*intersect/union
        else:
            return 1.0*intersect/(union-intersect)
        
            
    """
    Helper Funtions
    """
    def update_msg(self, pos1, pos2, x_i, alpha, beta, delta, 
                   mu, sigma, T, row1, col1, row2, col2, msg):
        # update message from pixel(row1,col1) to its neighbor at pos1
        # for the pixel(row2,col2) the message from pos2 is updated
        # pos: 0-up, 1-down, 2-left, 3-right
        
        # delta evaluation matrix
        delta_m = np.full((self.L,self.L), delta[0]*1.0)
        np.fill_diagonal(delta_m, delta[1])
        delta_m = beta*delta_m
        
        # x energy term
        sigma = np.copy(sigma)
        sigma[sigma==0] = 1 
        energy_x = alpha*((x_i-mu)**2/2/sigma\
                          +np.log(sigma))
        
        # get messeges from neighbors other than the one at pos1
        nbs = range(4)
        # get valid neighbor positions
        if row1==0: nbs.remove(0)
        if row1==self._nrow: nbs.remove(1)
        if col1==0: nbs.remove(2)
        if col1==self._ncol: nbs.remove(3)
        if pos1 in nbs: nbs.remove(pos1)
        # get product of messeges for each class
        msg_prod = msg[row1,col1,nbs,:].prod(axis=0)
        
        # update messages
        energy = (1/T)*(delta_m+energy_x)
        msg_raw = (np.exp(-energy)*msg_prod).sum(axis=1)
        # normalize
        msg[row2,col2,pos2,:] = msg_raw/msg_raw.sum()
        
        
    def bp_predict(self, x_i, mu, sigma, T, alpha,
                   row, col, msg, y):
        # predict the label for pixel(row,col)
        
        # compute energy_x
        sigma = np.copy(sigma)
        sigma[sigma==0] = 1
        energy_x = (1/T)*alpha*((x_i-mu)**2/2/sigma\
                          + np.log(sigma))
        
        # get messeges from neighbors 
        nbs = range(4)
        # get valid neighbor positions
        if row==0: nbs.remove(0)
        if row==self._nrow: nbs.remove(1)
        if col==0: nbs.remove(2)
        if col==self._ncol: nbs.remove(3)
        # message product
        msg_prod = msg[row,col,nbs,:].prod(axis=0)
        
        # sample label by the normalized belief
        belief = np.exp(-energy_x)*msg_prod
        pr = belief/belief.sum()
        pr[-1] = 1-pr[:-1].sum()
        pr[pr<0] = 0  
        y[row, col] = np.random.choice(self.L,1,p=pr)


    def bc_update_c(self, Phi, Lambda, Psi, d, 
                    w, mu_k, sigma_k):
        # update c
        
        # construct S
        s_ik = w/sigma_k
        s_i = s_ik.sum(axis=1)
        S = np.diag(s_i)
        
        # compute d_tilde
        d_tilde = (s_ik*mu_k).sum(axis=1)/s_ik.sum(axis=1) 
        
        c = np.dot(np.dot(Phi.T,S), Phi) 
        c = c + 2*Lambda*Psi
        c = np.linalg.inv(c)
        c = np.dot(c, Phi.T)
        c = np.dot(c, S)
        c = np.dot(c, d-d_tilde)
        
        return c
    
    def bc_update_w(self, d, Phi, c, mu_k, sigma_k, pi_k):
        # the E-step of EM for bias correction
        
        # the bias
        b = np.dot(Phi, c)
        # density of d - b 
        dnorm = np.array([norm.pdf(d-b, mu_k[i], 
                                   np.sqrt(sigma_k[i])) 
                                   for i in range(len(mu_k))]).T
        
        w = (dnorm*pi_k).T
        w = (w/w.sum(axis=0)).T
        
        dnorm = np.array([norm.pdf(d-b, mu_k[i], 
                                   np.sqrt(sigma_k[i])) 
                                   for i in range(len(mu_k))])
        w = dnorm.T*pi_k
        w = w.T/w.T.sum(axis=0)
        w[w==0] = 1e-15
        
        return w.T
    
    def bc_maximize(self, w, d, Phi, c):
        # the M-step of EM for bias correction
        
        # bias subtracted intensity
        d_hat = (d-np.dot(Phi,c)).reshape(len(d),1)
        # update mu_k
        mu_k = (d_hat*w).sum(axis=0)/w.sum(axis=0)
        # update sigma_k
        sigma_k = ((d_hat-mu_k)**2*w).sum(axis=0)/w.sum(axis=0)
        # update pi_k
        pi_k = w.sum(axis=0)/len(d)
        
        return mu_k, sigma_k, pi_k
        
    def bc_cost(self, w, d, Phi, mu_k, sigma_k, pi_k, 
                c, Lambda, Psi, regu=True):
        """
        evaluate the objective function for bias correction,
        if regu=False, the regularization term is not computed
        """
        
        # the bias
        b = np.dot(Phi,c)
        # density of d - b 
        dnorm = np.array([norm.pdf(d-b, mu_k[i], 
                                   np.sqrt(sigma_k[i])) 
                                   for i in range(len(mu_k))])
        # log part in the function
        log_part = dnorm.T*pi_k/w
        log_part[log_part == 0] = 1e-15
        log_part = np.log(log_part)*w

        EM_term = log_part.sum()
        
        if regu:
            regu_term = Lambda*np.dot(np.dot(c.T, Psi), c)
        else:
            regu_term = 0
            
        return (EM_term - regu_term)
    
    def to_label(self):
        # convert the pixel value to the corresponding label 
        labels = np.unique(self._seg)

        for i in range(len(labels)):
            self._seg[self._seg == labels[i]] = i
        
        


"""
Bspline class
"""
from scipy.linalg import toeplitz
import numpy as np

class Cubic_Bspline:
    
    def __init__(self, _min, _max, num_interval):
        
        self._min = _min
        self._max = _max
        self.num_interval = num_interval
        self.h = (_max - _min)*1.0/num_interval
        self.degree = 3
        self.knots = self.get_knots()
        
    def basis(self, xi):
        return self.__basis(xi, self.degree)
    
    def __basis0(self, xi):
        # evaluate basis function of order zero
        return np.where(np.all([self.knots[:-1] <=  xi, 
                                xi < self.knots[1:]],axis=0), 1.0, 0.0)
    
    def __basis(self, xi, degree):
        if degree == 0:
            return self.__basis0(xi)
        else:
            sub_basis = self.__basis(xi, degree-1) # basis of one less order
            
            first_numerator = xi - self.knots[:-degree]
            first_denominator = self.knots[degree:] - self.knots[:-degree]
            
            second_numerator = self.knots[(degree+1):] - xi
            second_denominator = self.knots[(degree+1):] - self.knots[1:-degree]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                first_term = np.where(first_denominator != 0.0, 
                                      first_numerator/first_denominator, 0.0)
                second_term = np.where(second_denominator != 0.0, 
                                       second_numerator/second_denominator, 0.0)
            
            return (first_term[:-1]*sub_basis[:-1] + second_term*sub_basis[1:])
    
    def penalty_matrix(self):
        # penalty matrix by the method of Reinsch,
        # Psi = (Delta.T)inv(W)Delta
        dim = len(self.knots) - self.degree - 1
        
        # construct delta
        delta_row = [1/self.h, 0, 1/self.h] + [0 for i in range(dim-3)] 
        delta_col = [1/self.h] + [0 for i in range(dim-3)]
        delta = toeplitz(delta_col, delta_row)
        
        # construct W
        diag1 = np.diag([self.h/6 for i in range(dim-3)], k=-1)
        diag2 = np.diag([2*self.h/3 for i in range(dim-2)], k=0)
        diag3 = np.diag([self.h/6 for i in range(dim-3)], k=1)
        W = diag1 + diag2 + diag3
        
        return np.dot(np.dot(delta.T, np.linalg.inv(W)), delta)
        
        
    def get_knots(self):
        # generate inner and bound knots
        knots = np.linspace(self._min, self._max, self.num_interval+1)
        
        # padd the knot vector
        knots = np.append(np.repeat(self._min, self.degree), knots)
        knots = np.append(knots, np.repeat(self._max, self.degree))
        
        return knots