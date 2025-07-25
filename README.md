# Objective & Description
This model takes in a folder of images and produces a folder of the same images
transformed. It is meant specifically for ID photos to be cropped, thus the
program has only been tested on photos with a blank, white background,
with one centered face in frame. The model will process each image one-by-one,
recognize the face in frame, the eyes within it, and (if the `draw_lines`
argument is set to true) will draw a green rectangle around each eye and a red
line horizontally across the image, through the middle of both eyes. The model
then does a fixed crop around that horizontal line, whether or not it is
visible on the finished image. The final image is cropped with the face centered
and shoulders spanning the entire length of the bottom of the image.

# Requirements
The model can only process images that are in .jpg,
.jpeg, or .png format.

# Resources
Program uses OpenCV (via JavaCPP's JavaCV bindings) and Haar cascades for 
facial and eye detection.

# Photo Dimensions
All photos originally used to test the model have the dimensions of
3840 × 5760 that should be cropped to a final size of 900 x 1200. 
Most variables are configured proportionally, but some may
not be. 

# Folder Output
The model expects an output folder to store the images in. If you run the
model multiple times consecutively, and for any reason there is an issue
with an image's output on a later run and it does not return, it will not
So, an earlier version of the image will be in the output. All that to say,
be sure to double check the time stamps for the images and make sure they are
all consistent, or clear the output folder with each output/make a new output
folder for each run.

# Usage
Before running the model from the command line, make sure that the terminal is
properly set up. Use the following lines after making sure that java is 
installed.

```
echo 'export JAVA_HOME=$(/usr/libexec/java_home)' >> ~/.zshrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
```


To run the model, it is good to start by double-checking that the model 
compiles.

```
mvn compile
```

For each time you want to run the program, start by running the following first:

```
mvn clean package 
```

Then run the command below with the variables replaced.
The `input directory` argument is the folder with the photos to be transformed.
The `output directory` argument is the folder where the final, transformed, 
cropped images will be stored.
The `draw lines` variable represents whether or not to draw lines in specific
areas (a red line through the middle of the detected eyes, a green rectangle
around each of the detected eyes).
Use the format below:

```
java -jar target/eye-detector-app-1.0-SNAPSHOT.jar \
<input_directory> <output_directory> <draw_lines>
```

With each file that has been processed, the command line will print a statement
for the user specifying that the respective file has been transformed. Error
messages will also print here. The final images are stored in the output folder 
specified.

You may get a similar warning to the example below:

```
WARNING: A restricted method in java.lang.System has been called
WARNING: java.lang.System::load has been called by org.bytedeco.javacpp.Loader \
in an unnamed module /
(file:/Users/Loaner/Desktop/eye_detector_app/target/eye-detector-app-1.0-SNAPSHOT.jar)
WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers \
in this module
WARNING: Restricted methods will be blocked in a future release unless \
native access is enabled
```

The program should still run despite these warnings. If they are a hassle to
deal with, follow the instructions in the terminal. 

# Void Methods
Methods altering the image are void to avoid making continual copies of the
original image and risk inputting a previous copy to a new method. This way,
the image merely undergoes a pipeline of alterations with only a single copy.
While it is common to write it returning the image for Java, we are using this
technique since it follows the conventions for OpenCV more closely.

# Coordinate Disclaimers
While working with coordinate pairs, any built-in library function regarding
coordinates of the eyes are treating the eyes as their own image. For
example, if you want to find the height of the eye relative to the entire
image, you need to do face.y() + eye.y() since eye.y() on its own will not
be in the right scale. (0,0) is the top left corner of the image, with the 
y-coordinate getting larger as you go further down the image.

# Padding Disclaimers
Padding is set to be (2/3) as an overall safe ratio. These variables
are mutable. (2/3) worked best at the time of testing.

# Future Improvements
- There are multiple instances in the main method that print an error and
write the file to the output path. Rather than doing this, it is
better to have exceptions thrown to the user. This method also avoids
making the code clunky.
- The program and class should be renamed to something more appropriate, 
since it now does more than merely detecting the eyes.

# Contact
- Author: Manoush Pajouh 
- GitLab: @mpajouh_suffield
- Contact: manoushpajouh@gmail.com
- Last Updated: July 25, 2025