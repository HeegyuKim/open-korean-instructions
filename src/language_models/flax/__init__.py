
import transformers as tf

from .phi3 import FlaxPhi3ForCausalLM

tf.FlaxAutoModelForCausalLM.register(
    tf.Phi3Config,
    FlaxPhi3ForCausalLM,
    exist_ok=True
    )

from .qwen2 import FlaxQwen2ForCausalLM

tf.FlaxAutoModelForCausalLM.register(
    tf.Qwen2Config,
    FlaxQwen2ForCausalLM,
    exist_ok=True
    )