import numpy as np
import jax
import jax.numpy as jnp
from functools import partial



# non-interacting Hamiltonian 
class H_free(object):
    def __init__(self, Lsite, N, t):
        self.Lsite = Lsite
        self.N = N
        self.t = t

    def get_matr(self):
        H_free_half = -self.t*jnp.eye(self.Lsite, k = 1) - self.t*jnp.eye(self.Lsite, k = -1)
        H_free_half = H_free_half.at[0, self.Lsite-1].set(-self.t)
        H_free_half = H_free_half.at[self.Lsite-1, 0].set(-self.t)  
       
        H_free = jnp.zeros((self.Lsite*2, self.Lsite*2))
        H_free = H_free.at[ :self.Lsite, :self.Lsite ].set(H_free_half)
        H_free = H_free.at[ self.Lsite:, self.Lsite: ].set(H_free_half)
        return H_free
        
    def get_eig(self):
        eigvals, eigvecs = jnp.linalg.eig(self.get_matr())
        return eigvals, eigvecs
   
    def get_psi0(self):  
        # < x | Psi >, where Psi is the non-interacting ground state
        E, U = self.get_eig()
        sort_indice = jnp.argsort(E)
        U_new = U[:, sort_indice[:self.N*2]] # its k-th column represents the k-th lowest eigenstate
        return U_new
 

# ======================================================================================



def all_hop_spin(state, Lsite):
    # state only contains single-spin electrons
    N = state.size
    hopped = jnp.zeros((nbond*N, N))
    for (i, x) in enumerate(state):
        hopped = hopped.at[i*2,:].set( jnp.sort(jnp.where(state==x, (state+1)%Lsite, state)) )
        hopped = hopped.at[i*2+1,:].set( jnp.sort(jnp.where(state==x, (state-1)%Lsite, state)) )
    return hopped 



def all_hop(state, Lsite):
    (N2, ) = state.shape
    N = int(N2/2)
    up, down = state[:N], state[N:] - Lsite
 
    up_copy = jnp.tile(up, (N*2, 1))
    down_copy = jnp.tile(down, (N*2, 1))
     
    hop_matr = jnp.vstack( (jnp.eye(N), -jnp.eye(N) ) ).astype('int32')

    up_hopped = (up_copy + hop_matr) % Lsite
    down_hopped = (down_copy + hop_matr) % Lsite

    hopped_up = jnp.hstack( (up_hopped, down_copy+Lsite) )
    hopped_down = jnp.hstack( (up_copy, down_hopped+Lsite))
   
    hopped = jnp.vstack( (hopped_up, hopped_down) )
    return hopped


def all_hop_vmap(states, Lsite, nbond = 2):
    (batch, N2) = states.shape
    N = int(N2/2)
    up, down = states[:,:N], states[:,N:] - Lsite

    up_copy = jnp.tile(up, N*2)
    up_copy = jnp.reshape(up_copy, (batch*N*2, N))
    down_copy = jnp.tile(down, N*2)
    down_copy = jnp.reshape(down_copy, (batch*N*2, N))

    hop_matr = jnp.vstack((jnp.eye(N), -jnp.eye(N)))
    hop_matr = jnp.tile(hop_matr, (batch, 1)).astype('int32')

    up_hopped = (up_copy + hop_matr) % Lsite
    down_hopped = (down_copy + hop_matr) % Lsite + Lsite

    up_hopped = jnp.hstack( (up_hopped, down_copy+Lsite) )
    down_hopped = jnp.hstack( (up_copy, down_hopped) )

    final = jnp.hstack((up_hopped, down_hopped))
    final = jnp.reshape(final, (batch, N*2, N*2*2))
    final = jnp.reshape(final, (batch, N*2*2, N*2))

    return final


def logpsi0(state, psi):
    U = psi[list(state),:]
    phase, det = jnp.linalg.slogdet(U)
    #return det + jnp.log(phase)
    #logphi = det + jnp.log(phase)
    #return 2 * logphi.real
    return phase, det



def make_psi0(state, psi):
    """
    Psi_0(x)
    """
    U = psi[list(state),:]
    det = jnp.linalg.det(U)
    return det


def make_psi0_ratio(state0, state1, psi):
    """
    the ration Psi_0(x1) / Psi_0(x0)
    """
    U0 = psi[list(state0),:]
    det0 = jnp.linalg.det(U0)

    U1 = psi[list(state1),:]
    det1 = jnp.linalg.det(U1)

    return det1 / det0


def count_double_occ(state, Lsite):
    N2 = state.size
    N = int(N2/2)
    up = state[None, :N]
    down = state[N:, None]
    diff = up + Lsite - down
    tmp = jnp.where(diff == 0, 1, 0)
    return jnp.sum(tmp)


def Gutzwiller_single(state, Lsite, g):
    """
    compute the Gutzwiller factor g^{D(x)} for a single state x, 
    where D(x) is the number of double occupation
    """
    (N2, ) = state.shape
    assert (N2, ) == state.shape
    N = int(N2/2)

    up = state[None, :N]
    down = state[N:, None]
    diff = up + Lsite - down
    double_occ_state = jnp.sum( jnp.where(diff == 0, 1, 0) )

    return jnp.power( g, double_occ_state ) 


def make_logpsi_square(state, psi, Lsite, g):
    """
    ln |Psi(x)|^2, Psi(x) = Psi_0(x) * g^{D(x)}
    """
    batch, Ne = state.shape
    U = psi[list(state),:]
    det = jnp.linalg.det(U)
 
    Gutzwiller_single_vmapped = jax.vmap(Gutzwiller_single, \
                                   in_axes = (0, None, None), out_axes = 0)
    Gutz = Gutzwiller_single_vmapped(state, Lsite, g)
    Psi = Gutz * det   
    Psi_square = Psi.real**2 + Psi.imag**2
    return jnp.log(Psi_square)



def Gutzwiller_ratio_single(state1, state0, Lsite, g):
    (N2, ) = state1.shape
    assert (N2, ) == state0.shape
    N = int(N2/2)

    up1 = state1[None, :N]
    down1 = state1[N:, None]
    diff1 = up1 + Lsite - down1
    double_occ_state1 = jnp.sum( jnp.where(diff1 == 0, 1, 0) )
     
    up0 = state0[None, :N]
    down0 = state0[N:, None]
    diff0 = up0 + Lsite - down0
    double_occ_state0 = jnp.sum( jnp.where(diff0 == 0, 1, 0) )

    return jnp.power( g, double_occ_state1 ) / jnp.power( g, double_occ_state0 )



def Gutzwiller_ratio_multi(states, state0, Lsite, g):
    (batch, N2) = states.shape
    assert (N2, ) == state0.shape
    N = int(N2/2)
    up = states[:, None, :N]
    down = states[:, N:, None]
    diff = up + Lsite - down
    double_occ_states = jnp.sum( jnp.where(diff == 0, 1, 0), (1,2) )
     
    up0 = state0[None, :N]
    down0 = state0[N:, None]
    diff0 = up0 + Lsite - down0
    double_occ_state0 = jnp.sum( jnp.where(diff0 == 0, 1, 0) )

    return jnp.power( g, double_occ_states ) / jnp.power( g, double_occ_state0 )




def Eloc(state, t, U, psi, Lsite, g):
    """
    Evaluated as \sum_{x'} < x |H| x' > <x'|Psi>/<x|Psi>, 
    """ 
    all_hopped = all_hop(state, Lsite)

    Gutz_ratio = Gutzwiller_ratio_multi(all_hopped, state, Lsite, g)
    #print('all_hopped:')
    #print(all_hopped)
    #print('state:')
    #print(state)
    #print('Gutz_ratio:')
    #print(Gutz_ratio)   


    """ 
    phase_all_hopped, logdet_all_hopped = logpsi(all_hopped, psi)
    phase_state, logdet_state = logpsi(state, psi)
    print('phase_all_hopped:')
    print(phase_all_hopped)
    print('phase_state:')
    print(phase_state)

    phase = phase_all_hopped / phase_state
    print('phase:')
    print(phase)

    logdet = jnp.exp(logdet_all_hopped - logdet_state) 
    Psi = jnp.multiply(phase, logdet)
    """

    Psi = make_psi0(all_hopped, psi) / make_psi0(state, psi)
    #print('Psi')
    #print(Psi)
 
    kinetic = -t * jnp.multiply(Gutz_ratio, Psi)
    kinetic = kinetic.sum(-1)
    #print('kinetic:', kinetic)
 
    potential = U * count_double_occ(state, Lsite)
    #print('potential:', potential)

    eloc = kinetic + potential
    #print('eloc:', eloc)
    return eloc


Eloc_vmapped = jax.vmap(Eloc, in_axes = (0, None, None, None, None, None), out_axes = 0)


def jump(state, Lsite, key):
    """
    change a state x to a new state x', and Psi_(x') must be nonzero.
    """
    tmp = state[None,:] - jnp.arange(Lsite)[:,None]
    pro = abs( jnp.prod(tmp, -1) )
    indice = jnp.argsort(pro)
    indice = indice[int(Lsite/2):]

    key_x, key_proposal = jax.random.split(key, 2)
    x = jax.random.choice(key = key_x, a = state)
    x_proposal = jax.random.choice(key = key_proposal, a = indice)
    state_new = jnp.where(state==x, x_proposal, state)

    return state_new 



def jump_nearest(state, Lsite, psi):
    """
    change a state x to a new state x' by a nearest hopping, and \Psi_(x') must be nonzero.
    """
    states_hopped = all_hop(state, Lsite)
    Nstate, Ne = states_hopped.shape
    #print('states_hopped:')
    #print(states_hopped)
    all_psi = make_psi0(states_hopped, psi) 
    all_psi = all_psi.real**2 + all_psi.imag**2
    #print('all_psi:')
    #print(all_psi)
    max_idx = jnp.argsort(all_psi)[-1]
    return states_hopped[max_idx]



def random_init(batch, Lsite, N):
    """
    randomly initialize a number of batch states for the begining of Metropolis
    """
    import random
    states = []
    for ibatch in range(batch):
        up = random.sample(range(Lsite), N)
        down = random.sample(range(Lsite, Lsite*2), N)
        state = up + down
        states.append(state)
    return jnp.array(states)


def anti_ferro_init(batch, Lsite, N):
    state_init = [ xx*2 for xx in range(N) ] + \
                 [ (yy*2+Lsite+1) for yy in range(N) ]
    state_init = jnp.array(state_init)
    states_init = jnp.tile(state_init, (batch, 1))
    return states_init

 
def Metropolis_single(psi, state_init, Lsite, t, U, g, nthermal, npoints, nacc):
    """
    nthermal: number of steps for thermalization 
    npoints: number of sampled |x> after thermalization
    nacc: number of steps between each sample 
    """
    (Ne, ) = state_init.shape
    state = jnp.array(list(state_init))
    
    nstep = nthermal + npoints * nacc
 
    Elocs = jnp.zeros(npoints)
    key = jax.random.PRNGKey(42)
    n_accept = 0

    states_all = jnp.array(list(state_init))
    num_double_occ = [ count_double_occ(state, Lsite) ]
    
    ipoints = 0
    for istep in range(nstep):
        print('istep:', istep)
        key, key_up, key_down, key_bernoulli, key_accept = jax.random.split(key, 5)
        print('state:')
        print(state)
        
        state_up = state[:int(Ne/2)]
        state_down = state[int(Ne/2):] - Lsite

        state_up_proposal = jump(state_up, Lsite, key_up)
        state_down_proposal = jump(state_down, Lsite, key_down) + Lsite
        #state_up_proposal = jump_nearest(state_up, Lsite, key_up)
        #state_down_proposal = jump_nearest(state_down, Lsite, key_down) + Lsite

        """
        up_proposal = jnp.hstack((state_up_proposal, state_down))
        down_proposal = jnp.hstack((state_up, state_down_proposal))

        up_or_down = jax.random.bernoulli(key_bernoulli, p=0.5)
        #print('up_or_down:', up_or_down)
        state_proposal = jnp.where(up_or_down, up_proposal, down_proposal)
        """

        state_proposal = jnp.hstack((state_up_proposal, state_down_proposal))
 
        ratio_psi = make_psi0(state_proposal, psi) / make_psi0(state, psi)
        ##ratio_psi = jnp.exp( logpsi(state_proposal, psi) - logpsi(state, psi) )
        ratio_Gutz = Gutzwiller_ratio_single(state_proposal, state, Lsite, g)
        ratio = ratio_psi * ratio_Gutz
        ratio = ratio.real**2 + ratio.imag**2
 
        accept = jax.random.uniform(key = key_accept, shape = ratio.shape) < ratio
        if accept:
            n_accept += 1
            states_all = jnp.vstack((states_all, state_proposal))

        state = jnp.where(accept, state_proposal, state)
        print('state_new:')
        print(state)
     
        num_double_occ.append( count_double_occ(state, Lsite) )

        if istep in range(nthermal, nstep+1, nacc):
            eloc = Eloc(state, t, U, psi, Lsite, g)
            print('eloc:', eloc)
            Elocs = Elocs.at[ipoints].set(eloc)
            ipoints += 1
        #print('\n')
  
    return state, Elocs, n_accept/nstep, states_all, num_double_occ



                
def Metropolis(psi, states_init, Lsite, t, U, g, nthermal, npoints, nacc):
    """
    nthermal: number of steps for thermalization 
    npoints: number of sampled |x> after thermalization
    nacc: number of steps between each sample 
    """
    batch, Ne = states_init.shape
    states = jnp.array(list(states_init))
 
    Elocs = jnp.zeros((npoints, batch))
    key = jax.random.PRNGKey(42)

    num_accept = 0
    double_occ = jnp.zeros((nthermal, batch))
    count_double_occ_vmapped = jax.vmap(count_double_occ, in_axes = (0, None), out_axes = 0)

    all_states = jnp.zeros((npoints, batch, Ne))

    jump_vmap = jax.vmap(jump, in_axes = (0, None, 0), out_axes = 0) 
    jump_nearest_vmap = jax.vmap(jump_nearest, in_axes = (0, None, None), out_axes = 0) 
 
    nstep = nthermal + npoints * nacc
     
    ipoints = 0
    for istep in range(nstep):
        print('istep:', istep)
        key, key_up, key_down, key_accept, key_bernoulli = jax.random.split(key, 5)
        #print('states:')
        #print(states)
        
        states_up = states[:, :int(Ne/2)]
        states_down = states[:, int(Ne/2):] - Lsite

        key_ups = jax.random.split(key_up, batch)
        key_downs = jax.random.split(key_down, batch)
        states_up_proposal = jump_vmap(states_up, Lsite, key_ups)
        states_down_proposal = jump_vmap(states_down, Lsite, key_downs) + Lsite

        """
        up_proposal = jnp.hstack((states_up_proposal, states_down))
        down_proposal = jnp.hstack((states_up, states_down_proposal))
        up_or_down = jax.random.bernoulli(key_bernoulli, p=0.5, shape = (batch,))
        #print('up_or_down:', up_or_down)
        states_proposal = jnp.where(up_or_down[:, None], up_proposal, down_proposal)
        """

        states_proposal = jnp.hstack((states_up_proposal, states_down_proposal))
        #states_proposal = jump_nearest_vmap(states, Lsite, psi)

        #print('states_proposal:')
        #print(states_proposal)

        #psi_ratio = jnp.exp( logpsi(states_proposal, psi) - logpsi(states, psi) )
        make_psi0_ratio_vmapped = jax.vmap( make_psi0_ratio, in_axes = (0, 0, None), out_axes = 0 )
        psi_ratio = make_psi0_ratio_vmapped(states, states_proposal, psi) 

        Gutzwiller_ratio_single_vmapped = jax.vmap(Gutzwiller_ratio_single, \
                                             in_axes = (0, 0, None, None), out_axes = 0 )
        Gutz_ratio = Gutzwiller_ratio_single_vmapped(states_proposal, states, Lsite, g)

        ratio = jnp.multiply( psi_ratio, Gutz_ratio )
        ratio = ratio.real**2 + ratio.imag**2
        accept = jax.random.uniform(key = key_accept, shape = ratio.shape) < ratio
        #print('accept:') 
        #print(accept)   
        num_accept += accept.sum()
        states = jnp.where(accept[:, None], states_proposal, states)
        #print('states_new')
        #print(states)

        if istep in range(nthermal):
            double_occ = double_occ.at[istep,:].set( count_double_occ_vmapped(states, Lsite) )

        if istep in range(nthermal, nstep+1, nacc):
            eloc = Eloc_vmapped(states, t, U, psi, Lsite, g)
            assert eloc.shape == (batch,)
            print('eloc:', eloc)
            Elocs = Elocs.at[ipoints,:].set(eloc)
            # 
            all_states = all_states.at[ipoints,:,:].set(states)
            ipoints += 1
            #print('Elocs:')
            #print(Elocs)
            #print('E_mean:', Elocs.mean())
            #print('states_new:')
            #print(states)

    Eloc_mean = Elocs.mean()
    accept_rate = num_accept/(batch*nstep)
    return Eloc_mean, accept_rate, double_occ, all_states



def grad_g(psi, states_init, Lsite, t, U, g, nthermal, npoints, nacc):
    """
    Gradient estimator
    dg ln Psi(x)^2 = 2 dg Psi(x) / Psi(x) = 2 dg G(x) Psi_0(x) / G(x) Psi_0(x) 
    = D(x) g^{-1}, where D(x) is the number of double occupation of | x >
    The full estimator is given by the mean value of D(x) * g^{-1} * ( E(x) -E.mean )
    """
    batch, Ne = states_init.shape
    states = jnp.array(list(states_init))
 
    Elocs = jnp.zeros((npoints, batch))
    #grad_all = jnp.zeros((npoints, batch))  # D(x) * g^{D(x) - 2}
    double_occ = jnp.zeros((npoints, batch))
    auto_grad_logpsi_all = jnp.zeros((npoints, batch))
    num_accept = 0
    
    key = jax.random.PRNGKey(42)

    jump_vmap = jax.vmap(jump, in_axes = (0, None, 0), out_axes = 0) 
    jump_nearest_vmap = jax.vmap(jump_nearest, in_axes = (0, None, 0), out_axes = 0) 

    #make_psi0_ratio_vmapped = jax.vmap( make_psi0_ratio, in_axes = (0, 0, None), out_axes = 0 )
    logpsi_square_grad = jax.jacrev(make_logpsi_square, argnums = -1)
    
    make_psi0_ratio_vmapped = jax.vmap( make_psi0_ratio, in_axes = (0, 0, None), out_axes = 0 )
    Gutzwiller_ratio_single_vmapped = jax.vmap(Gutzwiller_ratio_single, \
                                             in_axes = (0, 0, None, None), out_axes = 0 )
 
    count_double_occ_vmapped = jax.vmap(count_double_occ, in_axes = (0, None), out_axes = 0)

    nstep = nthermal + npoints * nacc
    ipoints = 0
    for istep in range(nstep):
        #print('istep:', istep)
        key, key_up, key_down, key_accept, key_bernoulli = jax.random.split(key, 5)
 
        #print('states:')
        #print(states)
        
        states_up = states[:, :int(Ne/2)]
        states_down = states[:, int(Ne/2):] - Lsite

        keys_up = jax.random.split(key_up, batch)
        keys_down = jax.random.split(key_down, batch)
        up_proposal = jump_vmap(states_up, Lsite, keys_up)
        down_proposal = jump_vmap(states_down, Lsite, keys_down) + Lsite

        """
        states_up_proposal = jnp.hstack((up_proposal, states_down))
        states_down_proposal = jnp.hstack((states_up, down_proposal))
        up_or_down = jax.random.bernoulli(key, p=0.5)
        states_proposal = jnp.where(up_or_down, states_up_proposal, states_down_proposal)
        """
        states_proposal = jnp.hstack((up_proposal, down_proposal))
        #print('states_proposal:')
        #print(states_proposal)

        psi_ratio = make_psi0_ratio_vmapped(states, states_proposal, psi) 
        Gutz_ratio = Gutzwiller_ratio_single_vmapped(states_proposal, states, Lsite, g)

        ratio = jnp.multiply( psi_ratio, Gutz_ratio )
        ratio = ratio.real**2 + ratio.imag**2
 
        accept = jax.random.uniform(key = key_accept, shape = ratio.shape) < ratio
        num_accept += accept.sum()

        states = jnp.where(accept[:, None], states_proposal, states)

        if istep in range(nthermal, nstep+1, nacc):
            eloc = Eloc_vmapped(states, t, U, psi, Lsite, g)
            assert eloc.shape == (batch,)
            print('eloc:', eloc)
            Elocs = Elocs.at[ipoints,:].set(eloc)
            # 
            doubleocc = count_double_occ_vmapped(states, Lsite) 
            double_occ = double_occ.at[ipoints,:].set(doubleocc)
            #
            auto_grad_logpsi = logpsi_square_grad(states, psi, Lsite, g)
            print('auto_grad_logpsi:', auto_grad_logpsi)
            auto_grad_logpsi_all = auto_grad_logpsi_all.at[ipoints,:].set(auto_grad_logpsi)
            #
            ipoints += 1

    E_mean = Elocs.mean()

    grad_logpsi = jnp.multiply( double_occ, jnp.power(g, -1) )
    grad = jnp.multiply( grad_logpsi, Elocs - E_mean )
    grad_mean = grad.mean() * 2


    print('auto_grad_logpsi_all:', auto_grad_logpsi_all)
    auto_grad = jnp.multiply( auto_grad_logpsi_all, Elocs - E_mean )
    auto_grad_mean = auto_grad.mean()
 
    return grad_mean, auto_grad_mean, E_mean



def optimize_g(psi, states_init, Lsite, t, U, nthermal, npoints, nacc, opt_nstep, learning_rate):

    import optax

    optimizer = optax.adam(learning_rate = learning_rate)
    param = jax.random.uniform( key = jax.random.PRNGKey(42), minval = 0.02, maxval = 1 )
    opt_state = optimizer.init(param)
    
    def step(param, opt_state):
        grad, loss = grad_g(psi, states_init, Lsite, t, U, param, nthermal, npoints, nacc)
        updates, opt_state = optimizer.update(grad, opt_state, param)
        param = optax.apply_updates(param, updates)
        return param, opt_state, loss

    for i in range(opt_nstep):
        param, opt_state, loss = step(param, opt_state)
        print('istep, g, E:')
        print(i, param, loss)
 



