# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

##Files submitted:
- model.py : The network model and trainer
- drive.py : The driver (provided by Udacity, modified throttle and image transformation)
- model.h5 : symbolic link to model4.h5 ("transfer learning" from earlier versions of models)
- Archive (dir) : Contains previous versions of models
- This write-up

## Model Architecture
The main model architecture is derived from the NVIDIA model with some variations. The Comma.ai model
was also evaluated but eventually the former model was chosen as it performed better in some initial tests.
Nonlinearities were introduced using Exponential ReLU (ELU) activations on various layers.

The architecture is shown below:
```text
LAYER: Lambda Normalization           (None, 200, 200, 3)
LAYER: Conv2D-24-3x3-s2               (None, 100, 100, 24)
LAYER: Maxpool2D                      (None, 50, 50, 24)
LAYER: Conv2D-36-3x3-s2               (None, 25, 25, 36)
LAYER: Maxpool2D                      (None, 12, 12, 36)
LAYER: Conv2D-48-3x3-s1               (None, 12, 12, 48)
LAYER: Maxpool2D                      (None, 12, 12, 48)
LAYER: Conv2D-64-3x3-s1               (None, 10, 10, 64)
LAYER: Flatten + Dropout(0.5)         (None, 6400)
LAYER: FullyConnected + Dropout(0.5)  (None, 500)
LAYER: FullyConnected + Dropout(0.5)  (None, 200)
LAYER: FullyConnected                 (None, 30)
LAYER: OUTPUT                         (None, 1)
```
The actual code (after removing print statements/comments)
```python
def nvidia():
    """A variant of the nvidia model"""

    model = Sequential()

    # Takes image after crop and reshape
    img_shape = (200, 200, 3)

    model.add(Lambda(lambda x: x/127.5 - 1.0,
                     input_shape=img_shape,
                     output_shape=img_shape))

    model.add(Convolution2D(24,3,3,border_mode='same',
              activation='elu',subsample=(2,2)))
    model.add(MaxPooling2D())

    model.add(Convolution2D(36,3,3,border_mode='same',
              activation='elu',subsample=(2,2)))
    model.add(MaxPooling2D())

    model.add(Convolution2D(48,3,3,border_mode='same',
              activation='elu',subsample=(1,1)))

    model.add(Convolution2D(64,3,3,activation='elu',
              subsample=(1,1)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(500,activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(200,activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(30,activation='elu'))
    model.add(Dense(1, activation='linear'))

    adamopt = Adam(lr=0.0001)
    model.compile(loss='mse',optimizer=adamopt)

    return model
```


### Overcoming Overfitting
In the final model, dropout of 0.5 was added to the three large fully connected layers. Lower values of dropout (0.2)
was also tried, but in the end, the agressive dropout seemed to perform better.

I also tried to use L2 regularizer as follows:
```python
model.add(Dense(500, activation='elu',
          W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
```
However, the regularization parameter of 0.01 was arbitrarily chosen and led to poor results (steering values
were constant near 0). I did not have time to play with the regularization hyperparameters to make it work. In the end,
only dropout was used.

## Image Data for training
I used only the dataset provided by Udacity and did not generate my own training data. I did attempt to drive
around in "training mode", however my keyboard was either too sensitive or I was too unskilled of a gamer to
keep the car on track for even a short distance. Perhaps I should have invested in a joystick.
In any case, I ditched these attempts to generate data and used the provided dataset.

[image1]: ./images/figure_1.png "Sample images from camera"
[image2]: ./images/figure_2.png "After cropping/resizing"

Here's a random selection of images from the provided dataset from the center camera:
![Sample images][image1]

Before sending the image to the model, it was cropped (70px on top and 20px on bottom) to remove the horizon, sky
 and other irrelevant features, and it was resized to square images (200px x 200px). This transformation was performed
 in a function (that is also called in drive.py).

```python
def image_crop_reshape(img):
    """Returns a cropped and reshaped image"""
    return cv2.resize(img[70:140, :, :],(200, 200))
```

Here are the same images as before after the above crop/reshape operation:

![Sample images after resize/crop][image2]

## Training

### Dataset evaluation

[image3]: ./images/figure_3.png "Original dataset"
[image4]: ./images/figure_4.png "Balanced histogram"
[image5]: ./images/figure_5.png "Balanced histogram for turns"

Here's a histogram of the steering angles from the original dataset"

<center><img src="./images/figure_3.png" alt="Original Dataset" style="width: 400px;"/></center>

It must be noted that it is heavily dominated by straight images. I originally trained with this
complete dataset "as-is" and the car wouldn't properly take turns. So I pruned the dataset to subselect
a balanced dataset with all bins (steering angle range 0.1) represented proportionally. Following is the resulting
histogram.

<center><img src="./images/figure_4.png" alt="Proportional Dataset" style="width: 400px;"/></center>

When I trained on the above dataset (transfer learning from previous learned models), the performance was better
but it was still failing at sharp turns. So I did a further pruning step and retrained with the following distribution
 of steering angles represented:

<center><img src="./images/figure_5.png" alt="Sharp turns Dataset" style="width: 400px;"/></center>

After these iterations, the car was able to complete track 1 fully. It was weaving off-center a bit, which was reduced
after retraining by further training on the second set (proportional).

### Image-data generator
The following generator was used to supply training data in batches to the model.

```python
def data_generator(A,BATCH_SIZE):
    """ An image data generator"""
    i = 0
    flip = True
    while True:
        x, y = [], []
        count = 0
        while count < BATCH_SIZE:

            # Pick center (prob = 3/5), left (1/5) or right (1/5) image
            mode = np.random.choice([1, 1, 1, 2, 3], 1)[0]
            #flip = np.random.random() < 0.5

            # Random shift in width and height
            wshift, hshift = 0.2*np.random.random(2)-0.1
            xi,yi = get_image_data(A,i,mode,flip,wshift,0.0*hshift)
            x.append(xi)
            y.append(yi)

            # Increment counter for batch
            count += 1

            # Reset to beginning once we reach end
            i += 1
            if i == len(A):
                A = A.sample(frac=1).reset_index(drop=True)
                flip = False
                i = 0

        yield np.array(x), np.array(y)
```

### Image transformations for augmenting data
To augment the dataset, certain random transformations were applied to the provided images. These include:
- Using left/right cameras instead of center with steering offsets of 0.25/-0.25
- Flipping vertically (this was originally done randomly, then done epoch-by-epoch)
- Random Horizontal shift (with steering offset factor of 0.5)

## Performance on Track 1
Here's a link to the video I recorded showing Track 1 performance:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=GEJsouYLAR0"
target="_blank"><img src="http://img.youtube.com/vi/GEJsouYLAR0/0.jpg"
alt="Track 1 Performance" width="480" height="360" border="10" /></a>

## Thoughts and future directions
Getting this to work took a lot of time and this is still far from over! I found that the car is stable
 only with a throttle of 0.1 and I can't get it to work fully with a higher throttle as I have seen others
 do. I need to check more thoroughly and perhaps improve my training dataset. Here's a to-do list for myself.

  - Improve training data
  - try to get it to work with increased throttle to 0.2
  - Check on track 2
  - Look at L2 regularization and associated hyperparameters
  - Add speed as an input quantity and throttle as output quantity on the model
  - Image augmentations with shadow/darkening that may help for track 2 (Ref: Vivek Yadav's [Post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.s1ceth5ks))

