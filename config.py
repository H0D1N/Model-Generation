#This is a config for colossal-ai
from colossalai.amp import AMP_TYPE

NUM_EPOCHS=400
BATCH_SIZES=128
LEARNING_RATE=0.1
WEIGHT_DECAY=5e-4
WARMUP_EPOCHS=16
MOMENTUM=0.9
LOG_PATH='./log'

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)
#gradient_accumulation = 16
#clip_grad_norm = 1.0


