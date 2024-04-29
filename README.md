# Novel-Image-Genertion-via-text-with-VQGAN #

Novel Image Generation via text prompts using VQGAN (Vector Quantized Generative Adversarial Networks). VQGAN is a generative model for image modeling and is an update over VQ-VAE architecture.

## Project Overview ##

Novel image generation helps to create synthetic yet realistic images that are semantically meaningful and diverse. It has multiple applications like synthesizing 3D shapes, and datasets for improved deep learning model training, medical diagnosis, creating and visualizing synthetic 3D images from 2D instances and vice versa for better scene analysis, and modeling objects for animation, games, and VR/AR. Traditional geometric modeling provides impressive results but at the hands of an experienced person. Manual processes require exact and accurate input, and different tools usually have steep learning curves. Hence, creating compelling 3D models can take lots of time. This project aims to provide an alternative approach to geometric modeling with Machine learning. The goal is to use generative modeling for geometric modeling to generalize from training data while generating new designs and shapes via text prompts entered by the user. This project overall combines the efficiency of convolutional approaches with the expressive power of transformers. The dataset used here is COCO (Common Object in Context) from Microsoft. The wide variety of objects in the dataset allows the model to generalize well to the user inputs.


![model architecture](https://github.com/chouhanpreeti/Novel-Image-Generation-via-text-with-VQGAN/blob/master/Outputs/architecture.png)
<p class="text-justify"> VQGAN (Vector Quantized Generative Adversarial Network) employs a two-stage structure by learning an intermediary representation before feeding it to a transformer. And, instead of downsampling the image, it uses a codebook to represent visual parts.  </p>
<br>
![system flowchart](https://github.com/chouhanpreeti/Novel-Image-Generation-via-text-with-VQGAN/blob/master/Outputs/flowchart.png)
<p class="text-justify"> Both the models are separate working in tandem. The way they work is that VQGAN generates the images, while Transformer judges how well an image matches the text prompt. This interaction guides the generator to produce more accurate images.</p>

<br>

### Objectives ###

- Implement the GAN architecture pipeline having an encoder and decoder as part of the generator and a corresponding discriminator.
- Train the stage 1 model on the MS COCO dataset to save the weights of the final model trained to be fed to the transformer model for updating the codebook vectors.
- Implement a transformer-based decoder namely the GPT2 architecture to learn and update the codebook vectors which correspond to image representations in the latent space.
- Train/optimize the transformer pipeline to learn the weights of the final model.
- Predict the outputs by using the GAN and transformer weights together.


### Result Samples ###
<br>
![output4](https://github.com/chouhanpreeti/Novel-Image-Generation-via-text-with-VQGAN/blob/master/Outputs/output.gif)
<p class="text-justify">A pink waterfall </p>
<br>

<br>
![output1](https://github.com/chouhanpreeti/Novel-Image-Generation-via-text-with-VQGAN/blob/master/Outputs/res_output1a.gif)
<p class="text-justify"> A bunch of red roses </p>
<br>

![output2](https://github.com/chouhanpreeti/Novel-Image-Generation-via-text-with-VQGAN/blob/master/Outputs/res_output5.gif)
<p class="text-justify">A smiling ghost </p>
<br>

<br>
![output3](https://github.com/chouhanpreeti/Novel-Image-Generation-via-text-with-VQGAN/blob/master/Outputs/res_output2.gif)
<p class="text-justify">A skull with blue eyes </p>
<br>


### Conclusion ###

- Due to limited computational resources, the model trained couldn't provide the perceived level of realistic objects. The model knowledge was also limited to the dataset used for the training.
- The model is able to capture the noun objects in the text prompt (e.g. zebra, cup, tree, motorcycle, etc.). Even though the generated object was not realistic enough, it shows a good relevance level to the corresponding noun object.
- The model still cannot have common sense reasoning capabilities, e.g. it fails on counting-related and spatial-direction prompts. It seems like the model ignores the information of the number, focusing only on the noun objects


## Dataset ##
Common Object in Context (COCO) Dataset is used for the model training to increase the model generalization to text prompts. 
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
