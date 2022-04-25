import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax_fun import H_free, all_hop, logpsi0, make_psi0, make_psi0_ratio, count_double_occ, \
                   Gutzwiller_ratio_single, Gutzwiller_ratio_multi, Eloc, Eloc_vmapped, \
                   jump, jump_nearest, random_init, Metropolis_single, Metropolis, \
                   grad_g, optimize_g, make_logpsi_square, Gutzwiller_single, random_init, anti_ferro_init

def test_Gutzwiller_single():
    state = jnp.array([0,1,2, 6,7,8])
    states = jnp.array([ [0,1,2, 6,7,8], [0,1,2, 6,7,9] ])
    Lsite = 6
    g = 0.5
    Gutzwiller_single_vmapped = jax.vmap(Gutzwiller_single, in_axes = (0, None, None), out_axes = 0)
    out_test = Gutzwiller_single_vmapped(states, Lsite, g)
    out = jnp.array( [ 0.125, 0.25 ] )
    assert jnp.allclose(out_test, out)


def test_logpsi0():
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
    psi = hfree.get_psi0()

    phase, logdet = logpsi0(states_full, psi)
    print(phase)
    print(logdet)

    det = make_psi0(states_full, psi)
    assert jnp.allclose( jnp.multiply(phase, jnp.exp(logdet)), det )
 


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
    psi = hfree.get_psi0()

    logpsi_x_square = make_logpsi_square(states, psi, Lsite, g)

    psi_x = make_psi0(states, psi)
    Gutzwiller_single_vmapped = jax.vmap(Gutzwiller_single, in_axes = (0, None, None), out_axes = 0)
    Gutz_weight = Gutzwiller_single_vmapped(states, Lsite, g)
    psi_x = jnp.multiply( psi_x, Gutz_weight )
    psi_x_square = psi_x.real**2 + psi_x.imag**2

    assert jnp.allclose( jnp.exp(logpsi_x_square), psi_x_square )
 


def test_Eloc():
    state = jnp.array([5,2,1, 7,8,9])
    
    t = 1.
    U = 0.
    Lsite = 6
    g = 1.
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    eloc = Eloc(state, t, U, psi, Lsite, g)
    print('eloc:', eloc)
    assert jnp.allclose( eloc, -8.0 )



def test_Eloc_vmapped():
    state1 = jnp.array([5,2,1, 6,7,9])
    state2 = jnp.array([4,2,1, 7,8,9])
    state3 = jnp.array([5,2,3, 6,9,11])
    states = jnp.vstack( (state1, jnp.vstack( (state2, state3) ) ) )   

    t = 1.
    U = 4.
    Lsite = 6
    g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    eloc1 = Eloc(state1, t, U, psi, Lsite, g)
    eloc2 = Eloc(state2, t, U, psi, Lsite, g)
    eloc3 = Eloc(state3, t, U, psi, Lsite, g)

    elocs = Eloc_vmapped(states, t, U, psi, Lsite, g)

    print(eloc1, eloc2, eloc3)
    print(elocs)
    assert jnp.allclose( elocs, jnp.array( [ eloc1, eloc2, eloc3 ] ) )
    
    


def test_make_psi0_ratio_vmapped():
    batch = 1
    t = 1.
    U = 4.
    Lsite = 6
    g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    state0 = jnp.array([0,1,4, 6,8,10])
    state1 = jnp.array([5,2,1, 7,9,10])
    state2 = jnp.array([4,2,1, 7,8,9])
    state3 = jnp.array([5,2,3, 6,9,11])
    states = jnp.vstack( (state1, jnp.vstack( (state2, state3) ) ) )   

    make_psi0_ratio_vmapped = jax.vmap( make_psi0_ratio, in_axes = (None, 0, None), out_axes = 0 )
    states_ratio = make_psi0_ratio_vmapped(state0, states, psi) 

    state1_ratio = make_psi0_ratio(state0, state1, psi) 
    state2_ratio = make_psi0_ratio(state0, state2, psi) 
    state3_ratio = make_psi0_ratio(state0, state3, psi) 

    assert jnp.allclose( states_ratio, jnp.array( [ state1_ratio, state2_ratio, state3_ratio ] ) )





def test_grad_g():
    batch = 1
    t = 1.
    U = 4.
    Lsite = 8
    g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    state_init = random_init(batch, Lsite, N)
    #print('state_init:')
    #print(state_init)

    nthermal = 200
    npoints = 10
    nacc = 20
    grad_mean, grad_auto_mean, E_mean = grad_g(psi, state_init, Lsite, t, U, g, nthermal, npoints, nacc)
    print('Elocs_mean:', E_mean)
    print('E per site:', E_mean/Lsite)
    print('grad_mean:', grad_mean)
    print('grad_auto_mean:', grad_auto_mean)
    assert jnp.allclose(grad_mean, grad_auto_mean)




