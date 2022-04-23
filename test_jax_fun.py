import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax_fun import H_free, all_hop, logpsi, make_psi, make_psi_ratio, count_double_occ, \
                   Gutzwiller_ratio_single, Gutzwiller_ratio_multi, Eloc, Eloc_vmapped, \
                   jump, jump_nearest, random_init, Metropolis_single, Metropolis, \
                   grad_g, optimize_g, make_logpsi_square, Gutzwiller_single


def test_Gutzwiller_single():
    state = jnp.array([0,1,2, 6,7,8])
    states = jnp.array([ [0,1,2, 6,7,8], [0,1,2, 6,7,9] ])
    Lsite = 6
    g = 0.5
    Gutzwiller_single_vmapped = jax.vmap(Gutzwiller_single, in_axes = (0, None, None), out_axes = 0)
    print(Gutzwiller_single_vmapped(states, Lsite, g))

def test_logpsi():
    state = jnp.array([0,1,2, 6,7,8])
    states = jnp.array([ [0,1,2, 6,7,8], [0,1,2, 6,7,9] ])
    states_full = jnp.array([ [[0,1,1, 6,7,8], [0,3,1, 6,7,8]], \
                              [[0,1,3, 6,7,8], [0,2,3, 6,6,10]] ])
    t = 1.
    U = 4.
    Lsite = 6
    g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    phase, det = logpsi(states_full, psi)
    print(phase)
    print(det)


def test_make_logpsi_square():
    state = jnp.array([0,1,2, 6,7,8])
    states = jnp.array([ [0,1,2, 6,7,8], [0,1,2, 9,10,11] ])
    states_full = jnp.array([ [[0,1,1, 6,7,8], [0,3,1, 6,7,8]], \
                              [[0,1,3, 6,7,8], [0,2,3, 6,6,10]] ])
    t = 1.
    U = 4.
    Lsite = 6
    g = 0.5
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    aa = make_logpsi_square(states, psi, Lsite, g)
    logpsi_square_grad = jax.jacrev(make_logpsi_square, argnums = -1)
    #print(aa)
    aa_grad = logpsi_square_grad(states, psi, Lsite, g)
    print(aa_grad)
 

def test_Eloc():
    state = jnp.array([5,2,1, 7,9,9])
    
    t = 1.
    U = 0.
    Lsite = 6
    g = 1.
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    eloc = Eloc(state, t, U, psi, Lsite, g)
    print(eloc)


def test_Eloc_vmapped():
    state1 = jnp.array([5,2,1, 7,9,9])
    state2 = jnp.array([4,2,1, 7,8,9])
    state3 = jnp.array([5,2,3, 6,9,11])
    states = jnp.vstack( (state1, jnp.vstack( (state2, state3) ) ) )   

    t = 1.
    U = 4.
    Lsite = 6
    g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    eloc1 = Eloc(state1, t, U, psi, Lsite, g)
    eloc2 = Eloc(state2, t, U, psi, Lsite, g)
    eloc3 = Eloc(state3, t, U, psi, Lsite, g)

    elocs = Eloc_vmapped(states, t, U, psi, Lsite, g)

    print(eloc1, eloc2, eloc3)
    print(elocs)
    

def test_jump_nearest():
    state = jnp.array([0,2,5, 7,9,10])
    t = 1.
    U = 4.
    Lsite = 6
    g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()
    #print(jump_nearest(state, Lsite, psi))
    jump_nearest_vmapped = jax.vmap(jump_nearest, in_axes = (0, None, None), out_axes = 0)
    states = jnp.array([ [0,2,5, 7,9,10], \
                         [0,1,4, 6,8,11] ])
    print(jump_nearest_vmapped(states, Lsite, psi))
    



def test_jump_vmapped():
    state1 = jnp.array([0,2,5])
    state2 = jnp.array([1,2,5])
    state3 = jnp.array([0,3,4])
    state4 = jnp.array([2,3,5])
    states = jnp.vstack( (state1, jnp.vstack( (state2, jnp.vstack( (state3, state4) ) ) ) ) )   

    Lsite = 6
    #for k in range(20):
    key = jax.random.PRNGKey(42)
    state1_new = jump(state1, Lsite, key)
    state2_new = jump(state2, Lsite, key)
    state3_new = jump(state3, Lsite, key)
    state4_new = jump(state4, Lsite, key)
    
    jump_vmapped = jax.vmap(jump, in_axes = (0, None, 0), out_axes = 0) 
    keys = jax.random.split(key, 4)
    states_new = jump_vmapped(states, Lsite, keys) 

    #print(jnp.hstack( (states, states_new) ))

    print(states)
    print('\n') 
    print(state1_new)
    print(state2_new)
    print(state3_new)
    print(state4_new)
    print(states_new)


def test_make_psi_ratio_vmapped():
    batch = 1
    t = 1.
    U = 4.
    Lsite = 30
    g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    state0 = jnp.array([0,1,4, 6,8,10])
    state1 = jnp.array([5,2,1, 7,9,9])
    state2 = jnp.array([4,2,1, 7,8,9])
    state3 = jnp.array([5,2,3, 6,9,11])
    states = jnp.vstack( (state1, jnp.vstack( (state2, state3) ) ) )   

    make_psi_ratio_vmapped = jax.vmap( make_psi_ratio, in_axes = (0, 0, None), out_axes = 0 )
    states_ratio = make_psi_ratio_vmapped(states, state0, psi) 





def test_Metropolis_single():
    batch = 1
    t = 1.
    U = 4.
    Lsite = 30
    g = 0.15
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    #state_init = [ xx*2 for xx in range(N) ] + \
    #             [ (yy*2+Lsite+1) for yy in range(N) ]
    #state_init = jnp.array(state_init)
    state_init = random_init(batch, Lsite, N)[0]
    #print('state_init:')
    #print(state_init)

    nthermal = 200
    npoints = 200
    nacc = 25
    #nstep = 2000
    state, Elocs, accept_rate, states_all, num_double_occ =\
             Metropolis_single(psi, state_init, Lsite, t, U, g, nthermal, npoints, nacc)
    print('Elocs_mean')
    print(Elocs.mean())
    print('E per site:')
    print(Elocs.mean()/Lsite)
    print('accept_rate:', accept_rate)



def test_Metropolis():
    batch = 2000
    t = 1.
    U = 4.
    Lsite = 30
    g = 0.7
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    #state_init = [ xx*2 for xx in range(N) ] + \
    #             [ (yy*2+Lsite+1) for yy in range(N) ]
    #state_init = jnp.array(state_init)
    state_init = random_init(batch, Lsite, N)
    #print('state_init:')
    #print(state_init)

    #nstep = 1000
    nthermal = 500
    npoints = 1
    nacc = 50
    #Eloc_mean =  Metropolis(psi, state_init, Lsite, t, U, g, nstep)
    Eloc_mean, accept_rate, double_occ, all_states =  \
                   Metropolis(psi, state_init, Lsite, t, U, g, nthermal, npoints, nacc)
    print('Elocs_mean')
    print(Eloc_mean)
    print('E per site:')
    print(Eloc_mean/Lsite)
    print('accept_rate:', accept_rate)

    #iocc = 0
    #for occ in double_occ:
    #     print('iocc, double_occ:', iocc, occ)
    #     iocc += 1 

    #import pickle as pk
    #double_occ_np = np.array(double_occ)
    #fp = open('./double_occ.txt', 'wb')
    #pk.dump(double_occ_np, fp)
    #fp.close()

    #for ss in all_states:
    #     print(ss)
   
    #import pickle as pk
    #all_states_np = np.array(all_states)
    #fp = open('./all_states.txt', 'wb')
    #pk.dump(all_states_np, fp)
    #fp.close()



def test_grad_g():
    batch = 1
    t = 1.
    U = 4.
    Lsite = 8
    g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    #state_init = [ xx*2 for xx in range(N) ] + \
    #             [ (yy*2+Lsite+1) for yy in range(N) ]
    #state_init = jnp.array(state_init)
    state_init = random_init(batch, Lsite, N)
    #print('state_init:')
    #print(state_init)

    nthermal = 200
    npoints = 10
    nacc = 20
    #Eloc_mean =  Metropolis(psi, state_init, Lsite, t, U, g, nstep)
    grad_mean, grad_auto_mean, E_mean = grad_g(psi, state_init, Lsite, t, U, g, nthermal, npoints, nacc)
    print('Elocs_mean:', E_mean)
    print('E per site:', E_mean/Lsite)
    print('grad_mean:', grad_mean)
    print('grad_auto_mean:', grad_auto_mean)




def test_optimize_g():
    batch = 2000
    t = 1.
    U = 4.
    Lsite = 30
    #g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi()

    states_init = random_init(batch, Lsite, N)

    nthermal = 400
    npoints = 1
    nacc = 50
 
    opt_nstep = 20

    optimize_g(psi, states_init, Lsite, t, U, nthermal, npoints, nacc, opt_nstep)




# run tests ==================================================================



#test_Gutzwiller_single()
#test_logpsi()
#test_make_logpsi_square()
#test_Eloc()
#test_Eloc_vmapped()
#test_jump_nearest()
#test_jump_vmapped()
#test_Metropolis_single()
#test_Metropolis()
test_grad_g()
#test_optimize_g()




