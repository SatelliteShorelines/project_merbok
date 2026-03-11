from transformers import TFSegformerForSemanticSegmentation, SegformerImageProcessor
import tensorflow as tf
import numpy as np
from PIL import Image

# 1) Load TF SegFormer
model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
model = TFSegformerForSemanticSegmentation.from_pretrained(model_id)

# 2) Prepare an HWC image (e.g., PIL or NumPy in HWC order)
image = np.random.rand(512, 512, 3).astype("float32")  # HWC

# 3) Use the processor with return_tensors="tf" → NHWC for TF
processor = SegformerImageProcessor.from_pretrained(model_id)
inputs = processor(images=image, return_tensors="tf")   # NHWC

# 4) Run inference (GPU)
outputs = model(**inputs, training=False)
print("logits:", outputs.logits.shape)  # (1, num_labels, 128, 128)