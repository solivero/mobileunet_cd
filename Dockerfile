FROM tensorflow/tensorflow
WORKDIR /app
RUN pip install numpy opencv-python keras imageio pydensecrf sklearn tensorflow_io tensorflow_addons
RUN apt install -y libgl1-mesa-glx
RUN ["/bin/bash"]