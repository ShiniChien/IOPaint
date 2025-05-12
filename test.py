import pickle
from gradio_client import Client

client = Client("http://0.0.0.0:7860/")

with open("/home/misa/Desktop/Old/IOPaint/tests/image.pkl", "rb") as f:
    image = pickle.load(f)
with open("/home/misa/Desktop/Old/IOPaint/tests/mask.pkl", "rb") as f:
    mask = pickle.load(f)

result = client.predict(
    image,
    mask,
    "resize",
    api_name="/process_inpaint"
)
print(result)
