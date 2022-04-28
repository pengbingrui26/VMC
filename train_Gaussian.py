import numpy as np
import jax 
import jax.numpy as jnp


def Gaussian_sampler(shape, sigma, key):
    #print('sigma', sigma)
    x = jax.random.normal(key = key, shape = shape)
    return sigma * x


def logp(x, sigma):
    p = 1/(np.sqrt(2*np.pi)*sigma) * jnp.exp(-x**2/(2*sigma**2))
    return jnp.log(p)
    

def make_grad_loss(x, beta, sigma):
    p = logp(x, sigma)
    F = 1/beta * p + 1/2. * x**2 
    F_mean = F.mean()
    grad_logp = -1/sigma + x**2/(sigma**3)

    auto_grad_logp = jax.jacrev(logp, argnums = -1)(x, sigma)

    grad = jnp.multiply(F, grad_logp)
    #auto_grad = jnp.multiply(F, auto_grad_logp)
    auto_grad = jnp.multiply( F-F_mean, auto_grad_logp )

    grad_mean = grad.mean()
    auto_grad_mean = auto_grad.mean()

    #assert jnp.allclose(grad_mean, auto_grad_mean)
    return grad_mean, F_mean
    #return auto_grad_mean, F_mean



def optimize_sigma(batch, beta, key, nstep, learning_rate):
    import optax

    optimizer = optax.adam(learning_rate = learning_rate)
    param = jax.random.uniform( key = jax.random.PRNGKey(42), minval = 0.01, maxval = 1.)
    opt_state = optimizer.init(param)
    
    x = Gaussian_sampler((batch, ), param, key)

    def step(param, opt_state):
        grad, loss = make_grad_loss(x, beta, param)
        updates, opt_state = optimizer.update(grad, opt_state, param)
        param = optax.apply_updates(param, updates)
        return param, opt_state, loss

    for istep in range(nstep):
        param, opt_state, loss = step(param, opt_state)
        print('istep, sigma, 1/sqrt(beta), loss:')
        print(istep, param, 1/jnp.sqrt(beta), loss)
 

batch = 5000
beta = 4.
key = jax.random.PRNGKey(42)
nstep = 300
learning_rate = 1e-2

optimize_sigma( batch, beta, key, nstep, learning_rate )



