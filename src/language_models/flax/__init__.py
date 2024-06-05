
import transformers as tf

from .phi3 import FlaxPhi3ForCausalLM

tf.FlaxAutoModelForCausalLM.register(
    tf.Phi3Config,
    FlaxPhi3ForCausalLM,
    exist_ok=True
    )
