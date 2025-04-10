import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
jax.config.update("jax_explain_cache_misses", True)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


@jax.jit
def f(x):
    y = jnp.ones((2, 2))
    z = x + y
    return z


x = jnp.zeros((2, 2))
f(x)
