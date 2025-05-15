import jax
import jax.numpy as jnp
import optimistix as optx
import optax

#optimizer = optx.OptaxMinimiser(optax.adagrad(learning_rate=sigma/m), rtol=1e-4, atol=1e-4)

def build_restricted_gaussian_oracle(logprob_fn, m, sigma, alpha):
    
    def rgo(y_bar, rng):
        def f_bar(x, *_): return -logprob_fn(x, None) + 0.5* m/ sigma * jnp.sum((x - y_bar) ** 2)
        x_star = optx.minimise(f_bar, optx.BFGS(rtol=1e-4, atol=1e-4), y_bar, max_steps=256, throw=False).value
        fprime = jax.grad(lambda o: logprob_fn(o, None))(x_star)
            
        def propose(params):
            rng_key_x, _, _, count = params
            rng_key_x, rng_proposal_a, rng_proposal_b = jax.random.split(rng_key_x, num=3)
            
            g = lambda u: -logprob_fn(u, None) + jnp.dot(fprime, u)  
            noise = jax.random.normal(rng_proposal_a, shape=y_bar.shape)
            a = x_star + alpha * noise
            noise = jax.random.normal(rng_proposal_b, shape=y_bar.shape)
            b = x_star + alpha * noise
            rho = jnp.exp(g(a) - g(b))
            u = jax.random.uniform(rng_key_x, shape=(1,))
            #jax.debug.print("U is {u} 🤯", u=(u - jnp.exp(-f_bar(z) + f_bar(x_star) + 1/2 * jnp.sum(noise**2))>0))
            return rng_key_x, u - 0.5*rho, a, count + 1
        def cond(params):
            rng_key_x, gap, z, count = params
            return (jnp.sum(gap) > 0.0)
    
        rng, rng_step = jax.random.split(rng)
        _, _, sample, _ = jax.lax.while_loop(cond, propose, (rng_step, jnp.array([10.0]), x_star, 0))
        return sample
    return rgo

def build_multi_prox_kernel(logprob_fn, m, sigma, alpha):
    
    assert m > 1
    
    RGO = jax.jit(build_restricted_gaussian_oracle(logprob_fn, m-1, sigma, alpha))
    jumpRGO = jax.jit(build_restricted_gaussian_oracle(logprob_fn, m, sigma, alpha))
    
    #@jax.jit
    def update(avg_y_s, rng_key):
        rng_key, estimate_key = jax.random.split(rng_key)
        X_guess = RGO(avg_y_s, estimate_key)
        rng_key, noise_key = jax.random.split(rng_key)
        return X_guess + jnp.sqrt(sigma) * jax.random.normal(noise_key, shape=(avg_y_s.shape[0],))
    
    def kernel(rng_key, state):
        
        state = state.copy()
        
        def gibbs_update(carry, y):
            key, y_sum = carry
            #jax.debug.print("🤯 {y} 🤯", y=y_sum)
            avg_y_s = (y_sum - y)/(m-1)
            key, step_key = jax.random.split(key)
            y_val = update(avg_y_s, step_key)
            return (key, y_sum - y + y_val), y_val

        carry , state["y_s"] = jax.lax.scan(gibbs_update, (rng_key, jnp.sum(state["y_s"], axis=0)), state["y_s"])
        rng_key = carry[0]
        rng_key, step_key = jax.random.split(rng_key)
        #jax.debug.print("🤯 {y} 🤯", y=state['y_s'])
        state["x"] = jumpRGO(jnp.sum(state["y_s"], axis=0)/m, step_key)
        rng_key, noise_key = jax.random.split(rng_key)
        state["y_s"] = state["x"] + jnp.sqrt(sigma) * jax.random.normal(noise_key, shape=state["y_s"].shape)
        
        return state
        
    return kernel

class MultiProxSampler():
    
    def __init__(self, logprob_fn, m, sigma, alpha):
        self.kernel = (build_multi_prox_kernel(logprob_fn, m, sigma, alpha))
        
    def sample(self, init_state, rng_key, N):
        keys = jax.random.split(rng_key, N)
        
        def single_step(state, key):
            sample = self.kernel(key, state)
            return sample, sample["x"]
        
        _, samples = jax.lax.scan(single_step, init_state, keys)
        
        return samples