# Face Recognition using FaceNet:
1. The algorithm of FaceNet takes an input image, detects the presence and the location of a face in the image but it does not label the face.  
2. While detecting the face, it identifies facial landmarks and alignments of the face present in the image.  
3. Face alignment, as the name suggests, is the process of identifying the geometric structure of the faces and attempting to obtain a canonical alignment of the face based on translation, rotation, and scale. 
4. Once the process of face alignment is done, the face is passed through the neural network.  
<p align="center">
<img src = "/Net/1.png">
</p>
5. The FaceNet deep learning model computes a 128-d embedding that quantifies the face itself. The network calculates the face embeddings using the input and the triplet loss function.  

6. In each input batch there contains a anchor image,a negative image and a positive image. Anchor - current face, positive image - image containing current face, negative image - image not containing current face.  

7. The neural network computes the 128-d embeddings for each face and then tweaks the weights of the network using the triplet loss function such that, the embeddings of the anchor and positive image lie closer together and while at the same time, pushing the embeddings for the negative image father away.  
<p align="center">
<img src = "/Net/2.png">
</p>
