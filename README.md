# WhereIsWaldo

<img src="https://deadline.com/wp-content/uploads/2016/03/wheres-waldo.jpg?w=600&h=383&crop=1" style="width: 80%;"/>

## Goal

The project is about research and development in computer vision. The aim of the project is to succeed in pointing the character of Waldo from the game "Where is Waldo?" Achieving this task therefore takes into account two aspects: on the one hand, robotics with the handling of the arm and the achievable movements, and on the other hand, the machine learning part and the detection of Waldo's face using the robot's camera.

## Presentation

<img src="http://www.yahboom.net/Public/images/newsimg/61d817018a675.jpg" style="width: 50%;"/>

The arm available is a Yahboom DOFBOT with a JETSON Nano. This makes it possible to work on the arm. The arm is composed of joints and an RGB camera. You can find more details about the robot and it provides a tutorial to make the robot move on the following link : 
http://www.yahboom.net/study/Dofbot-Jetson_nano 

As the arm movements can be performed, it is necessary to complete with the detection of Waldo. In order to recognise Waldo, we turned to a Yolov5 learning model. Yolov5 uses a convolutional neural network (CNN) to retrieve features from an image and an classifier to label the objects. In our case, we only have one class: "Waldo". We therefore used a first dataset available on Kaggle to train this model (https://www.kaggle.com/datasets/residentmario/wheres-waldo). This dataset is composed of photos that we will call "digital", i.e. digital images directly taken from the digital format of a board and not from a photo of a board of "Where is Waldo? The difference lies in the quality of the images and the absence of external factors, such as luminosity, which can be found with a camera.

## Improve the model's performance on real data

To overcome this problem, we created a new dataset, this time from photos taken with the robot's camera. This dataset is therefore a "real" dataset, composed of Waldo's plate with, consequently, particularities such as Waldo's angle, luminosity, contrast, camera resolution and any other external elements that may appear. We then annotated these images and performed data augmentation using Roboflow to add images with a certain angle or contrast. We went from about forty photos to more than 120. And we have launched a learning process with this new dataset. The results are not necessarily satisfactory. One of the reasons is the size of the dataset, which remains small.

To improve this, we tried a training with the new dataset but using frozen layers with the weights of the training with more data. Again, the results are no better and even worse in some situations. The way the robot points at Waldo adds even more uncertainty. Indeed, the robot, first aligning itself with Waldo, while still detecting, before pointing at her, sometimes causes the loss of Waldo's recognition. This is due to poor learning data on particular images of Waldo, with an angle on Waldo's head for example.

The last solution we undertook was to further expand the dataset. To do this, we printed new images with Waldo's head at different sizes, adding angles, some light, to try to get Waldo in more situations. We supplemented the old real dataset with 70 more images, so 110 in all, for 330 when expanded. We ran a new full training (without freezed layers) and this time the results are more satisfactory. However, this model is different from the first one which also had good results. Indeed, the first dataset has images of complete boards while the last one contains images of Waldo closer to the board, without necessarily the whole board and therefore its environment. The two datasets are therefore quite complementary but not necessarily perfect when used independently.

## Possible extension

We found out that we could ensemble methods such as boosting, bagging and stacking. We didn't have the time for implement this type of model. As a quick reminder,
Boosting: cascading models of varying performance for particular conditions, specialization. 
Bagging: running several models and voting on the system output
Stacking: form a meta learner from several mini models by weighting the output in relation to the mini models
![Schema](https://pythoncursus.nl/wp-content/uploads/2020/04/ensemble-methods-boosting-bagging-stacking.png)
##### image from https://pythoncursus.nl/ensemble-methods/

Also we could use yolov8n model to improve the speed, which is a bit smaller and faster model than yolov5s, see the following image provided by yolov8 :
![yolov8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)


## Example

You will find some results on the repository and also weights in the following route : yolov5/runs/train/exp/weights/

![results](https://raw.githubusercontent.com/RemiTan/WhereIsWaldo/main/results.png)
##### From our best model, weights called best_1.pt

Video demo:
![Demo](https://github.com/RemiTan/WhereIsWaldo/blob/main/Inference%20example%20with%20the%20robot.mp4)

## Special thank to

To my partner, Vianney MELLE-VUILLOD who helped me in the project.

Thanks to our tutors, Benjamin ALLAERT and José MENNESSON, for giving us the opportunity to work on this project, which allowed us to apply what we first learned during this year and to put ourselves in a situation on this challenge. They also helped us by giving some advices and when facing some technical problems.

Thanks to Amel AISSAOUI who also provided some advices. 
