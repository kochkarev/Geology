import tensorflow as tf
import tensorflow.keras as K


policy = K.mixed_precision.Policy('mixed_float16')
K.mixed_precision.set_global_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
