# Steps to read and follow
Implemented our project in tensorflow, the idea behind the project is to find the optimal knowledge that can be tranfered from source to target network.

please download weights of alexnet and put in the same folder. Download url is:http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
please download weighs of VGG16 and put in the same folder Download URL link: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy 
please download caltech101 images from http://www.vision.caltech.edu/Image_Datasets/Caltech101/
please download caltech256 images from http://www.vision.caltech.edu/Image_Datasets/Caltech256/

1. In datasets folder, we have Caltech 101 and Caltech 256 datasets along with train and test txt files, which have paths to Caltech 101 and Caltech 256 images.
2. Since the overall dataset size is not huge, we have included datasets also
3. We have DataInput train and test files for both caltech101 and caltech256 which are named as DataInput_train_caltech101.py DataInput_train_caltech256.py, like wise for test.
4. We have five main files which need to be run independently to understand the knowledge transfer from source to target network
5. If you run main-caltech101-alexnet.py, it uses pretrained weights of alexnet trained on Imagenet and trains only the last layer with high learning rate and other n-1 layers with lower learning rate. Either we can keep the learning rate of n-1 layers to zero or we can keep it low. We would get an accuracy of 82% after 3 epochs.
6. If you run main-caltech256-alexnet.py it also uses pretrained weights of alexnet trained on imagenet and trains only the last layer with hight learning rate and other n-1 layers with lower learning rate, but what you see is the accuracy of caltech256 after taking n-1 layers is not good at all. At first epoch it would be around 10%
7. You can see same results as above with little difference in accuracies when you run other two files, main-vgg16-caltech101.py and main-vgg16-caltech256.py. you would see around 77% and 10%
8. Now if you run main-vgg16-caltech256-n-layer.py, this would give better accuracy than above caltech 256 because We took n-6 layers instead of n-1 layers indicating that caltech 256 features are specific to only certain layers. 
9. We can keep changing the layers that are trainable in main-vgg16-caltech256-n-layer.py to demostrte the idea.  
8 since this project is more on figuring out the optimality, you need to train each model if you want to check the idea behind. Each model would take around 30 minutes to train. 
   


