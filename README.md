# An integration of real-time face recognition pipeline

* Just an integration of face recognition pipeline to pure pytorch version
* References: to be added
* TensorRT inference: to be added
* Many necessary information is missing now (e.g. environment configuration), which will be supplemented later

# Pipeline

- input: a 3-channel image
- Face Detection: use MTCNN to detect all possible bounding boxes of 'faces', each with 5 keypoints
- Image Crop: crop all the faces according to bounding boxes
- Keypoint Alignment: use skimage to project detected keypoints to standard location
- [Function] Face Anti-spoofing: a pytorch model to detect whether the cropped face represents real  faces or not
- Feature Encoding: in local image dataset, execute feature encoding to transfer image into 128 features, all stored in json format
- [Function] Face Recognition: a pytorch model to recognize the corresponding people's name from given face image 

# Advantage

- Newly created repositoty with newly updated packages,  easy to configure the environment
- Friendly to all users, giving explicit coding style and coding process, which is not only serving for ourselves and just storing it on GitHub
- Integrating very small but accurate models, all based on GPU so as to make a slightest mobile face recognition pipeline, or even CPU with still fast inference, with which to construct a face-recognition platform in your own PC at any time and any place
- [to be added] create GUI to help register new people's face, making dataset maintenance easier
- [to be added] apply TensorRT to accelerate inference in some mobile platforms (e.g. NVIDIA Jetson TX2)
- [to be added] more complete pipeline, including sex detection, age detection, etc. All successful appliance will just depend on small model(s)

- [Tested] Now all the models only cover 553MB of space in your GPU.

