import numpy as np
#from 1D_ED import Hubbard_1D

class VMC_1D(object):
    
    def __init__(self, L, N, t, U, g):
        self.Lsite = L
        self.N = N
        self.t = t
        self.U = U
        self.g = g

        H_free_half = -t*np.eye(self.Lsite, k = 1) - t*np.eye(self.Lsite, k = -1)
        H_free_half[0][self.Lsite-1] = -t
        H_free_half[self.Lsite-1][0] = -t
        H_free = np.zeros((self.Lsite*2, self.Lsite*2))
        H_free[:self.Lsite, :self.Lsite] = H_free_half
        H_free[self.Lsite:, self.Lsite:] = H_free_half
        self.H_free = H_free
        
        self.eigvals, self.eigvecs = np.linalg.eig(self.H_free)

    def get_basis(self):
        from itertools import combinations
        up = combinations(range(L), N)
        up = [ sorted(list(xx)) for xx in up ]
        up_basis = up
        #for i, occ in enumerate(up):
        #    up_basis.append(occ)
        down_basis = up_basis
        basis = []
        for x in up_basis:
            for y in down_basis:
                y_new = [ (yy+self.Lsite) for yy in y ] 
                basis.append(x+y_new)
        return basis  

    def get_basis_num(self, num = 20):
        from itertools import combinations
        up = combinations(range(L), N)
        up = [ sorted(list(xx)) for xx in up ]
        up_basis = []
        for i, occ in enumerate(up):
            up_basis.append(occ)
        down_basis = up_basis
        return up_basis

        """
        basis = []
        bb = False
        NN = 0
        for x in up_basis:
            if bb:
                break
            for y in down_basis:
                y_new = [ (yy+self.Lsite) for yy in y ] 
                basis.append(x+y_new)
                if len(basis) >= num:
                    bb = True
                    break 
        assert len(basis) == num, len(basis)
        return basis  
        """


    def Vstate(self):   
        """
        variational state \ket{psi}
        constructed by the N lowest eigenstates of non-interacting Hamiltonian
        """
        E, U = self.eigvals, self.eigvecs
        sort_indice = np.argsort(E)
        U_new = U[:, sort_indice[:self.N*2]] # its k-th column represents the k-th lowest eigenstate
        return U_new  

    def Vstate_up(self):   
        E, U = self.eigvals, self.eigvecs
        sort_indice = np.argsort(E)
        U_new = U[:, sort_indice[:self.N*2]] # its k-th column represents the k-th lowest eigenstate
        tmp = U_new[:, range(0, self.N*2, 2)]
        return tmp[range(self.Lsite), :]

    def Vstate_down(self):   
        E, U = self.eigvals, self.eigvecs
        sort_indice = np.argsort(E)
        U_new = U[:, sort_indice[:self.N*2]] # its k-th column represents the k-th lowest eigenstate
        tmp = U_new[:, range(1, self.N*2, 2)]
        return tmp[range(self.Lsite, self.Lsite*2), :]


    def x_Psi0(self, state):  # < x | \Psi_0 >
        U = self.Vstate()
        assert type(state) == list, type(state)
        assert len(state) == self.N*2
        U = U[state,:]
        det = np.linalg.det(U)
        return det
   
    def hopping(self, state, x, dire): 
        assert dire in [1, -1]
        assert x in range(self.Lsite)
        assert type(state) == list, state
        assert len(state) == self.N, len(state)

        spin = ''
        ##if all( [uu in range(self.Lsite) for uu in state] ):
        if state[0] < self.Lsite:
            spin = 'up'
        ##elif  all( [dd in range(self.Lsite, self.Lsite*2) for dd in state] ):
        else:
            spin = 'down'
        assert spin in ['up', 'down'], spin

        state_new = state.copy()
        parity = None

        if spin == 'down':
            state_new = [ (xx - self.Lsite) for xx in state_new ]

        x1 = x+dire
        x11 = (x+dire) % self.Lsite

        if (x not in state_new) or (x11 in state_new):
            state_new = None
        else:
            alpha = x11
            beta = x
            l = state_new.index(beta) + 1
            parity1 = l-1
            state_new.remove(beta)
            state_new = sorted(state_new + [alpha])
            s = state_new.index(alpha) + 1
            parity2 = s-1 
            #parity3 = 1
            ##if x1 < 0 or x1 >= self.Lx or y1 < 0 or y1 >= self.Ly:
            #if (x1, y1) not in self.idx.values():
            #    parity3 = -1
            import math
            #parity = int(math.pow(-1, parity1 + parity2)*parity3)
            parity = int(math.pow(-1, parity1 + parity2))
            if spin == 'down':
                ##print('plus:')
                ##print('state_new:', state_new)
                state_new = [ (xx + self.Lsite) for xx in state_new ]

        return state_new, parity


    def all_hoppings(self, state):
        assert len(state) == self.N*2
        up = state[:self.N]
        down = state[self.N:]
 
        state_hopped = []
        for x in range(self.Lsite):
            if (x not in up) and ((x+self.Lsite) not in down):
                continue 
            for dire in [1, -1]:
                tmp_up, parity_up = self.hopping(up, x, dire)
                tmp_down, parity_down = self.hopping(down, x, dire)
                ##print('tmp_up:', tmp_up)
                ##print('tmp_down:', tmp_down)
 
                up_hopped, down_hopped  = '', ''
                if tmp_up == None:
                    assert parity_up == None
                    up_hopped = None
                else:
                    up_hopped = tmp_up + down
                if tmp_down == None:
                    assert parity_down == None
                    down_hopped = None
                else:
                    down_hopped = up + tmp_down

                up_hopped = (up_hopped, parity_up)
                down_hopped = (down_hopped, parity_down)

                if up_hopped not in state_hopped:
                    state_hopped.append( up_hopped )
                if down_hopped not in state_hopped:
                    state_hopped.append( down_hopped )

        return state_hopped
 
    def x_H_x(self, state1, state2): # < x | H | x' > 
        assert(type(state1)) == list
        assert(type(state2)) == list
        assert len(state1) == self.N*2
        assert len(state2) == self.N*2
        assert state2 == sorted(state2), state2
        double_occ = [ i for i in state2[:self.N] if (i+self.Lsite) in state2[self.N:] ]
        double_occ = len(double_occ)
        U = double_occ * self.U

        state2_hopped = self.all_hoppings(state2)
        overlap = 0
        for hopped_and_parity in state2_hopped:
            (state_hopped, parity) = hopped_and_parity
            if state_hopped == state1:
                ###overlap -= self.t * parity
                overlap -= self.t
        if state1 == state2:
            overlap += U
        return overlap

    def Gutzwiller_special(self, x, x1):
        """
        | x > = |n_1, ... , n_k, ... , n_l, n_L >
        | x1 > = |n_1, ... , n_k + 1, ... , n_l - 1, n_L >
        """
        #
        l = [ ll for ll in x if ll not in x1 ]
        assert len(l) == 1, l 
        l = l[0]
        k = [ kk for kk in x1 if kk not in x ]
        assert len(k) == 1, k 
        k = k[0]
        # decide if l-site is doubly occupied before hopping
        l_double_occ = False
        if (l+self.Lsite) in x:
            l_double_occ = True
        # decide if k-site will be doubly occupied after hopping
        k_double_occ = False
        if (k+self.Lsite) in x1:
            k_double_occ = True
        G_x, G_x1 = 1, 1
        if l_double_occ:
            G_x = G_x * self.g
        if k_double_occ:
            G_x1 = G_x1 * self.g
        return G_x1/G_x        
 
    def Gutzwiller_weight(self, state):
        """
        The Gutzwiller projector is given by g^D, where g = exp(-1/2 alpha), and D 
        is total number of double occupation
        """
        assert len(state) == self.N*2, state
        double_occ = len( [ xx for xx in state[:self.N] if (xx+self.Lsite) in state[self.N:] ] )
        import math
        return math.pow(self.g, double_occ)

    def nonzero_x_H_x(self, state):
        """
        for a given | x > belonging to basis, 
        find all | x' > such that < x' | H | x > != 0
        """
        assert len(state) == self.N*2, state
        state_hopped = self.all_hoppings(state)
        all_states = [ hopped for (hopped, parity) in state_hopped if hopped != None ]
        assert all( [ sorted(hh) == hh for hh in all_states ] )
        return all_states + [state]


    def fast_update(self, x, x1):
        """
        W_{K,l} = \sum_{a} U_{K,a} * \hat{U}_{a, l}  
        """
        U = self.Vstate()
        U_x = U[x, :]
        #W = np.dot( U, np.linalg.inv(U_x) )
        W = np.dot( U, np.linalg.pinv(U_x) )
        """
        | x > = |n_1, ... , n_k, ... , n_l, n_L >
        | x1 > = |n_1, ... , n_k + 1, ... , n_l - 1, n_L >
        """
        l = [ x.index(ll) for ll in x if ll not in x1 ]
        assert len(l) == 1, l 
        l = l[0]
        k = [ kk for kk in x1 if kk not in x ]
        assert len(k) == 1, k 
        k = k[0]
        return W[k][l] 

    def fast_update_up(self, x, x1):
        U = self.Vstate_up()
        U_x = U[x, :]
        W = np.dot( U, np.linalg.pinv(U_x) )

        l = [ x.index(ll) for ll in x if ll not in x1 ]
        assert len(l) == 1, l 
        l = l[0]
        k = [ kk for kk in x1 if kk not in x ]
        assert len(k) == 1, k 
        k = k[0]
        return W[k][l] 

    def fast_update_down(self, x, x1):
        U = self.Vstate_down()
        x_tmp = [ (xx - self.Lsite) for xx in x ]
        x1_tmp = [ (xx1 - self.Lsite) for xx1 in x1 ]

        U_x = U[x_tmp, :]
        W = np.dot( U, np.linalg.pinv(U_x) )

        l = [ x_tmp.index(ll) for ll in x_tmp if ll not in x1_tmp ]
        assert len(l) == 1, l 
        l = l[0]
        k = [ kk for kk in x1_tmp if kk not in x_tmp ]
        assert len(k) == 1, k 
        k = k[0]
        return W[k][l] 


    def fast_update_seperate(self, x, x1):
        x_up, x_down = x[:self.N], x[self.N:]
        x1_up, x1_down = x1[:self.N], x1[self.N:]
        ratio_up = self.fast_update_up(x_up, x1_up)
        ratio_down = self.fast_update_down(x_down, x1_down)
        return ratio_up * ratio_down

    def fast_update_general(self, x, x1):
        """
        x and x1 are arbitrary states in basis
        suppose there are m different sites between x and x1
        """
        U = self.Vstate()
        U_x = U[x, :]
        l_idx = [ i for i in range(self.N*2) if x[i] != x1[i] ]
        m = len(l_idx)
        K = [ x1[j] for j in l_idx ] 
        l = [ x[k] for k in l_idx ]
        assert len(K) == m
        assert len(l) == m
        #print(l, K)
        A = U[K, :]
        B = np.linalg.pinv(U_x)[:, l_idx]
        W = np.dot(A, B)
        return np.linalg.det(W)


    def Eloc(self, state):
        """
        Evaluated as \sum_{x'} < x |H| x' > <x'|Psi>/<x|Psi>, 
        ###<x|Psi> = <x|P_G|Psi_0> = P_G(x)<x|Psi_0>, where P_G is the Gutzwiller projector
        Here x is state 
        """ 
        eloc = 0
        ##basis = self.get_basis() 
        ##assert state in basis, state
        all_states = self.nonzero_x_H_x(state)
        ##for ba in basis:
        #print('x_Psi0(state):', self.x_Psi0(state))
        for ba in all_states:
            if ba == state:
                eloc += self.x_H_x(state, ba)
            elif ba != state:
                eloc += self.x_H_x(state, ba) * \
                    self.Gutzwiller_weight(ba) / self.Gutzwiller_weight(state) * \
                    self.fast_update(state, ba)
                    #self.x_Psi0(ba) / self.x_Psi0(state)
                    #self.fast_update(state, ba)

        return eloc



    def P_accept(self, x, x1):
        """
        accept = | P(x1) / P(x) |^2
        P(x) = < x | Psi > = G(x) < x | Psi0 >
        """
        # x is the origial "point", x1 is the new one
        detU_x = self.x_Psi0(x)
        detU_x1 = self.x_Psi0(x1)
        Gutz = self.Gutzwiller_weight(x1) / self.Gutzwiller_weight(x)
        P_ratio = Gutz * detU_x1/detU_x
        P_ratio = P_ratio.real**2 + P_ratio.imag**2
        return P_ratio
    

    def P_accept_fast(self, x, x1):
        """
        W_{K,l} = \sum_{a} U_{K,a} * \hat{U}_{a, l}  
        """
        #E, U = self.eigvals, self.eigvecs
        U = self.Vstate()
        U_x = U[x, :]
        W = np.dot( U, np.linalg.pinv(U_x) )
        """
        | x > = |n_1, ... , n_k, ... , n_l, n_L >
        | x1 > = |n_1, ... , n_k + 1, ... , n_l - 1, n_L >
        """
        l = [ x.index(ll) for ll in x if ll not in x1 ]
        assert len(l) == 1, l 
        l = l[0]
        k = [ kk for kk in x1 if kk not in x ]
        assert len(k) == 1, k 
        k = k[0]
        ##print(W.shape)
        Gutz = self.Gutzwiller_weight(x1) / self.Gutzwiller_weight(x)
        P_ratio = W[k][l] * Gutz
        P_ratio = P_ratio.real**2 + P_ratio.imag**2
        return P_ratio

    def nearest_jump(self, state):
        assert len(state) == self.N*2
        state_update = ''
        idx_available = [ ]
        for i in state[:self.N]:
            if (i-1)%self.Lsite not in state:
                idx_available.append( (i, -1) )
            elif (i+1)%self.Lsite not in state:
                idx_available.append( (i, 1) )
        for j in state[self.N:]:
            if ( (j-self.Lsite-1)%self.Lsite + self.Lsite ) not in state:
                idx_available.append( (j, -1) )
            elif ( (j-self.Lsite+1)%self.Lsite + self.Lsite ) not in state:
                idx_available.append( (j, 1) )
        assert idx_available != []
        import random
        (idx, dire) = random.choice(idx_available)
        #dire = random.choice([1, -1])
        idx_new = ''
        if idx in state[:self.N]:
            idx_new = (idx + dire) % self.Lsite
        elif idx in state[self.N:]:
            idx_new = (idx - self.Lsite + dire) % self.Lsite + self.Lsite
        state_new = state.copy()
        state_new.remove(idx)
        state_new.append(idx_new)
        state_new = sorted(state_new)
        return state_new

    def jump(self, state):
        assert len(state) == self.N*2
        up, down = state[:self.N], state[self.N:]

        import random
        spin = random.choice([1, -1])
        l, k = '', ''
        if spin == 1: # spin-up
            l = random.choice(up)
            kk = [ s for s in range(self.Lsite) if s not in up ]
            k = random.choice(kk)
        elif spin == -1: # spin-down
            l = random.choice(down)
            kk = [ s for s in range(self.Lsite, self.Lsite*2) if s not in down ]
            k = random.choice(kk)
        assert l in range(self.Lsite*2)
        assert k in range(self.Lsite*2)
         
        state_update = state.copy()
        state_update.remove(l)
        state_update.append(k)
        state_update = sorted(state_update) 

        return state_update
      
    def random_init(self):
        import random
        from itertools import combinations

        #up = combinations(range(self.Lsite), self.N)
        #up = [ sorted(list(xx)) for xx in up ]
        #up_idx = random.randint(0, len(up)-1)
        #up_init = up[up_idx]

        #down = combinations(range(self.Lsite, self.Lsite*2), self.N)
        #down = [ sorted(list(yy)) for yy in down ]
        #down_idx = random.randint(0, len(down)-1)
        #down_init = down[up_idx]
        up = random.sample(range(self.Lsite), self.N)
        down = random.sample(range(self.Lsite, self.Lsite*2), self.N)
        return sorted(up + down)

    def Metropolis(self, nstep):
        E_tot = [ ]
        num_accept = 0
 
        import random
        #state_init = [ xx*2 for xx in range(self.N) ] + [ (yy*2+self.Lsite+1) for yy in range(self.N) ]
        #state_init = [ 0,  1,  6,  8,  9, 11, 12, 13, 16, 17]
        #state_init = [ 2,  4,  6,  7,  8, 10, 11, 12, 13, 18]
        #state_init = [ 2, 3, 6, 8, 10, 11, 12, 15, 18, 20, 21, 25, 26, \
        #               27, 29, 30, 31, 32, 33, 34, 37, 39, 42, 46, 47, 50, \
        #               51, 52, 53, 58]
        state_init = self.random_init()
        state = state_init.copy()
 
        non_repeat = []
 
        for istep in range(nstep):
            """
            | state > = |n_1, ... , n_k, ... , n_l, n_L >
            | state_new > = |n_1, ... , n_k + 1, ... , n_l - 1, n_L >
            E = \sum_x P(x) Eloc(x)
            P(x) = < x | Psi > = G(x) < x | Psi0 >
            """
            #if istep % 20 == 1:
            #    print(istep, E_tot/(istep+1))             
            """
            l = random.choice(state)
            if l in range(self.Lsite):  # l in spin-up
                k = random.choice([nn for nn in range(self.Lsite) if nn not in state])
            elif l in range(self.Lsite, self.Lsite*2):  # l in spin-down
                k = random.choice([nn for nn in range(self.Lsite, self.Lsite*2) if nn not in state])

            state_new = [ k if mm == l else mm for mm in state ]
            state_new = sorted(state_new)
            """
            #state_new = self.nearest_jump(state)
            state_new = self.jump(state)

            ##P_accept = min(1, self.P_accept(state, state_new))
            P_accept = min(1, self.P_accept_fast(state, state_new))
            eta = random.uniform(0, 1) 
            if eta <= P_accept:
                state = state_new
                num_accept += 1

            if state not in non_repeat:
                non_repeat.append(state)

            if istep in range(100, nstep+50, 50):
                eloc = self.Eloc(state)
                print('istep, eloc:', istep, eloc)
                E_tot.append(eloc)

        #print('len_non_repeat:', len(non_repeat))
        #print('len_bais:', len(basis))

        #for ii in range(nstep):
        #    E_tot += Eloc[ii] * P[ii] / P_sum   
        E_mean = sum(E_tot) / len(E_tot)
        accept_rate = num_accept/nstep
        return state_init, state, E_mean, accept_rate


# test

"""
L, t, U, g = 10, 1., 4., 0.077
N = int(L/2)
model = VMC_1D(L, N, t, U, g)

#print(model.random_init())
#exit()
"""

"""
H_free = model.H_free
print(H_free)
exit()
"""

"""
E = model.eigvals
EE = sorted(E)
for e in EE: 
    print(e)
exit()
"""


"""
basis = model.get_basis()
#for ba in basis:
#    print(ba)
print(len(basis))
import random
a = random.choice(basis)
print(a)
exit()
"""


"""
basis = model.get_basis_num()
#for ba in basis:
#    print(ba)
print(len(basis))
exit()
"""


"""
U = model.Vstate()
print(U)
"""



"""
state = [ 0, 1, 2, 6, 7, 8 ]
x_psi0 = model.x_Psi0(state)
print(x_psi0)
"""


"""
state = [ 6, 7, 8 ]
x = 0
dire = -1
hopped, parity = model.hopping(state, x, dire)
print('state:', state)
print('state_hopped:', hopped)
print('parity:', parity)
exit()
"""

"""
#state = [ 0, 1, 2, 6, 7, 8 ]
state = [ 0, 1, 2, 3,  8, 9, 10, 11 ]
all_hopped = model.all_hoppings(state)
print('state:', state)
for hh in all_hopped:
    print(hh)
"""


"""
state1 = [0, 1, 2, 6, 7, 8]
state2 = [0, 1, 2, 6, 7, 8]

overlap = model.x_H_x(state1, state2)
print(overlap)
"""


"""
state = [0, 1, 2, 6, 7, 8]
Gutz_weight = model.Gutzwiller_weight(state)
print(Gutz_weight)
"""


"""
#state = [0, 1, 2, 6, 7, 8]
state = [0, 2, 4, 6, 8, 10]
out = model.nonzero_x_H_x(state)
print(out)
print(len(out))
exit()
"""


"""
#x = [ 8,  7,  6,  3,  1, 14, 10, 18, 16, 11]
#x1 = [ 8,  7,  9,  3,  1, 14, 10, 18, 17, 11]
x = [ 4,  5,  1,  9,  7, 13, 17, 14, 10, 16]
x1 = [ 4,  5,  2,  9,  7, 13, 17, 14, 18, 16]
x = sorted(x)
x1 = sorted(x1)
ratio = model.x_Psi0(x1) / model.x_Psi0(x)
#ratio_fast = model.fast_update(x, x1)
#ratio_fast_ge = model.fast_update_general(x, x1)
print(ratio**2)
#print(ratio_fast)
#print(ratio_fast_ge)
exit()
"""


"""
#state = [0, 1, 2, 3, 8, 9, 10, 11]
state = sorted( [ 8,  3,  6,  1,  5, 10, 16, 13, 15, 18] )
#state = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
Eloc = model.Eloc(state)
print(Eloc)
exit()
"""


"""
#state1 = [ 4,  2,  3,  6,  0, 13, 17, 11, 19, 15]
#state2 = [ 9,  2,  3,  6,  0, 13, 17, 11, 19, 18]
state1 = [ 2,  3,  5,  7,  9, 10, 16, 18, 14, 15]
state2 = [ 0,  3,  5,  7,  9, 10, 16, 18, 14, 11]
state1 = sorted(state1)
state2 = sorted(state2)

P_accept = model.P_accept(state1, state2)
print(P_accept)
#P_accept_fast = model.P_accept_fast(state1, state2)
#print(P_accept_fast)
exit()
"""




"""
#state = [ xx*2 for xx in range(model.N) ] + [ (yy*2+model.Lsite+1) for yy in range(model.N) ]
#state = [0, 1, 2, 6, 7, 8]
#ss = model.nearest_jump(state)
#print('state:', state)
#print('state_jumped:', ss)


state = [ xx*2 for xx in range(model.N) ] + [ (yy*2+model.Lsite+1) for yy in range(model.N) ]
ss = []
nstep = 2000
for i in range(nstep):
    state = model.nearest_jump(state)
    if state not in ss:
        ss.append(state)
print(len(ss))

exit()

basis = model.get_basis()
absent = []
for bb in basis:
    if bb not in ss:
        absent.append(bb)

print(len(absent), len(basis))

non_repeat = []
for s in ss:
    if s not in non_repeat:
        non_repeat.append(s)
print(non_repeat)

exit()
"""


# run 

"""
import matplotlib.pyplot as plt
"""


"""
L, t, U = 6, 1., 4.
N = int(L/2)
nstep = 2000
print('1D VMC:')
print('L, t, U, Npoints:', L, t, U, nstep)


g = 0.47
model = VMC_1D(L, N, t, U, g)
state_init, state_final, E_tot = model.Metropolis(nstep)
print('E_tot:', E_tot)
print('init_state:', state_init)
print('final_state:', state_final)
print('E_tot_per_site:', E_tot/model.Lsite)
exit()
"""

"""
L, t, U = 30, 1., 4.
N = int(L/2)
nstep = 2000
print('1D VMC:')
print('L, t, U, Npoints:', L, t, U, nstep)

gg = np.arange(0.4, 0.9, 0.1)
E = []
for g in gg:
    model = VMC_1D(L, N, t, U, g)
    state_init, state_final, E_tot = model.Metropolis(nstep)
    E.append(E_tot/model.Lsite)
    print('g:,', g)
    ##print('state_init:', state_init)
    ##print('state_final:', state_final)
    #print(E_tot)
    print('E_tot_per_site:', E_tot/model.Lsite)
    print('\n')

plt.plot(gg, E)
plt.show()
"""



"""
L, t  = 8, 1.
N = int(L/2)
nstep = 3000
print('1D VMC:')
print('L, t, Npoints:', L, t, nstep)

gg = np.arange(0., 1.01, 0.1)

U1 = 2.
E1 = []
for g in gg:
    model = VMC_1D(L, N, t, U1, g)
    state_init, state_final, E_tot = model.Metropolis(nstep)
    E1.append(E_tot/model.Lsite)
    print('g:,', g)
    ##print('state_init:', state_init)
    ##print('state_final:', state_final)
    #print(E_tot)
    print('E_tot_per_site:', E_tot/model.Lsite)
    print('\n')

U2 = 6.
E2 = []
for g in gg:
    model = VMC_1D(L, N, t, U2, g)
    state_init, state_final, E_tot = model.Metropolis(nstep)
    E2.append(E_tot/model.Lsite)
    print('g:,', g)
    ##print('state_init:', state_init)
    ##print('state_final:', state_final)
    #print(E_tot)
    print('E_tot_per_site:', E_tot/model.Lsite)
    print('\n')

U3 = 10.
E3 = []
for g in gg:
    model = VMC_1D(L, N, t, U3, g)
    state_init, state_final, E_tot = model.Metropolis(nstep)
    E3.append(E_tot/model.Lsite)
    print('g:,', g)
    ##print('state_init:', state_init)
    ##print('state_final:', state_final)
    #print(E_tot)
    print('E_tot_per_site:', E_tot/model.Lsite)
    print('\n')


U4 = 14.
E4 = []
for g in gg:
    model = VMC_1D(L, N, t, U4, g)
    state_init, state_final, E_tot = model.Metropolis(nstep)
    E4.append(E_tot/model.Lsite)
    print('g:,', g)
    ##print('state_init:', state_init)
    ##print('state_final:', state_final)
    #print(E_tot)
    print('E_tot_per_site:', E_tot/model.Lsite)
    print('\n')




plt.figure(num = 4, figsize = (8,5))

plt.title('1D VMC, L = 8, t = 1.0, Npoints= 3000')
plt.plot(gg, E1, color = 'red', label = 'U = 2')
plt.plot(gg, E2, color = 'orange', label = 'U = 6')
plt.plot(gg, E3, color = 'blue', label = 'U = 10')
plt.plot(gg, E4, color = 'green', label = 'U = 14')

plt.legend()
plt.xlabel('g')
plt.ylabel('Energy per site')

plt.show()
"""


