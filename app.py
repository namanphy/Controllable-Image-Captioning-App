import streamlit as st
from inference import predict_captions
import cv2
from PIL import Image
import copy
import psutil
print(psutil.virtual_memory())
import numpy as np


def generate_captions(image_buffer):
    """

    :param image_file: image io.iobytes buffer file
    :return: results
    """
    beam_size = 10

    agg_results = []
    for len_class in [0, 1, 2]:
        print(f'data_type: INFERENCE, beam size: {beam_size}, length class: {len_class}')
        results = predict_captions(
            beam_size, len_class,
            data_type='INFERENCE',
            n=200, subword=True,
            img_file=image_buffer, inference=True
        )

        if not agg_results:
            agg_results = results
        else:
            for i in range(len(agg_results)):
                agg_results[i].update(results[i])

    return agg_results


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Image Captioning App")
st.write("")

file = st.file_uploader("Upload file", type=["jpg"])

if file:
    file2 = copy.deepcopy(file)
    # image = cv2.imdecode(np.fromstring(file.read(), np.int8), 1) #cv2.IMREAD_COLOR
    image = Image.open(file2)

    st.write("")
    st.write("Just a second...")

    st.image(image, caption='Uploaded Image.', use_column_width=True)

    data = generate_captions(file.read())
    st.write(data)
