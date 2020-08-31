# Understanding the cfg file

## Classification

### General

Here you can select the things the program should execute. If you select *calc knowledge quotients* the program will terminate after it calculated the quotients. If *test fine-tune* strategies is selected, *train final model* will not be executed.

### Dataset

Enter the name of the dataset, you want to use.

### Model

Type is the base type of model you want to use. If you saved the whole model to disk, you can reload it with the *from file* and *filepath* option. If you want to load pretrained weights, the *pretrained* and *weightspath* option implement this.

The usual input size for mobilenets is 224x224. If your images of your dataset have a smaller format (f.e. CIFAR10 is 32x32), you can chosse to insert an upsampling layer in front of the network, which scales all images to 224x224. This is controlled through the *scale up input* option.

If you want to change the stride of layers from 2 to 1, because you are using smaller input images, you can choose how many strides of layers you want to change through the *change stride layers* option. Strides of layers with small indexes are preferred to be changed.

### Shunt

The shunt is fully defined by choosing two locations (layer indexes) and an architecture. The easiest way to find your desired layer indexes is to look at the output file, after running the *calc knowledge quotients* mode.

Just as the model, the shunt can also be loaded from a file or pretrained weights can be loaded.

For training and testing the shunt, feature maps of the shunt's location are needed. Please specify your desired path for saved feature maps through the *featuremapspath* option. The program will check, if the needed maps got already saved there. If they are not found, maps will get extracted and saved inside the folder. 

### Training

You can specify different options for the training of the original model, the shunt model and the final (shunt inserted) model. Right now, I am using a two-cycle approach. So I am using two different learning rates for a different amount of epochs.

For the final model, a fine-tune strategy must also be choosen.

### Final Model

You can also load pretrained weights for the final model.