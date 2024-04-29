# Novel-Image-Genertion-via-text-with-VQGAN #

Novel Image Generation via text prompts using VQGAN (Vector Quantized Generative Adversarial Networks). VQGAN is a generative model for image modeling and is an update over VQ-VAE architecture.

## Project Overview ##

Novel image generation helps to create synthetic yet realistic images that are semantically meaningful and diverse. It has multiple applications like synthesizing 3D shapes, and datasets for improved deep learning model training, medical diagnosis, creating and visualizing synthetic 3D images from 2D instances and vice versa for better scene analysis, and modeling objects for animation, games, and VR/AR. Traditional geometric modeling provides impressive results but at the hands of an experienced person. Manual processes require exact and accurate input, and different tools usually have steep learning curves. Hence, creating compelling 3D models can take lots of time. This project aims to provide an alternative approach to geometric modeling with Machine learning. The goal is to use generative modeling for geometric modeling to generalize from training data while generating new designs and shapes via text prompts entered by the user. This project overall combines the efficiency of convolutional approaches with the expressive power of transformers. The dataset used here is COCO (Common Object in Context) from Microsoft. The wide variety of objects in the dataset allows the model to generalize well to the user inputs.


<img src="./model/architecture.png" width = "800" />
<p class="text-justify"> VQGAN (Vector Quantized Generative Adversarial Network) employs a two-stage structure by learning an intermediary representation before feeding it to a transformer. And, instead of downsampling the image, it uses a codebook to represent visual parts.  </p>
<br>
<img src="./model/flowchart.png" width = "700" />
<p>Both the models are separate working in tandem. The way they work is that VQGAN generates the images, while Transformer judges how well an image matches the text prompt. This interaction guides the generator to produce more accurate images.</p>



## Dataset ##
Common Object in Context (COCO) Dataset is used for the model training to increase the model generalization to text prompts. 
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
