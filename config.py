#This is a config for colossal-ai
from colossalai.amp import AMP_TYPE

NUM_EPOCHS=200
BATCH_SIZES=256
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=16
LOG_PATH='./log'

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)
gradient_accumulation = 4
#clip_grad_norm = 1.0


