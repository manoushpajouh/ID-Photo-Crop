import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

/**
 * EyeDetector is a Java application that processes facial ID images by 
 * detecting faces and eyes, aligning the face so that the eyes are positioned 
 * at 1/3 the height of the final image, and outputting a standardized, cropped,
 * and optionally annotated version of each input image.
 * 
 *
 * The class uses OpenCV (via JavaCPP's JavaCV bindings) and Haar cascades for 
 * facial and eye detection.
 * It performs the following tasks:
 *   - Loads and validates input images from a directory
 *   - Detects faces and eyes using Haar cascades
 *   - Filters out false-positive eye detections based on position heuristics
 *   - Computes a crop rectangle that centers the face and aligns the eyes to
 *      1/3 of the final image height
 *   - Applies optional visual debugging by drawing eye lines and rectangles
 *   - Writes the final cropped and resized images to an output directory
 *
 * Usage:
 *   java -jar target/eye-detector-app-1.0-SNAPSHOT.jar \
 *   EyeDetector <input_directory> <output_directory> <show_lines>

 * where:
 *   <input_directory> – path to the folder containing input images
 *   <output_directory> – path where processed images will be saved
 *   <show_lines> – "true" or "false", whether to draw debug lines and eye boxes

 * Example:
 *   java -jar target/eye-detector-app-1.0-SNAPSHOT.jar \
 *   EyeDetector /Desktop/images/input Desktop/images/output true

 *
 * Note:
 *  The image cropping includes padding (2/3 of the target size)
 *  to allow flexible vertical alignment of the eyes. 
 *  The final output size is configurable  via the static variables 
 *  {@code finalHeight} and {@code finalWidth}. This version has been tested
 *  only with 900x1200 measurements.
 *
 * @author Manoush Pajouh
 * @version July 25, 2025
 */

public class EyeDetector {
    /** Goal height of the final image */
    public static int finalHeight = 1200;
    /** Goal width of the final image */
    public static int finalWidth = 900;
    /** Target final image size */
    public static Size targetSize = new Size(finalWidth, finalHeight);
    /** Where the eyes will be in the final image,
    set to be a third of the way down the entire image */
    public static int finalEyeY = Math.round(targetSize.height() * 0.33f);

    /**
     * Method to make sure that the path is correct and can be read
     * 
     * @param resourcePath the path to the file/folder of images to edit
     * 
     * @return the same inputted resourcePath
     * 
     * @throws IllegalArgumentException if the file cannot be found
     * @throws IOException if there are any input or output issues
     * for example if the file can't be opened for writing because
     * permission is denied, or the path is invalid or inaccessible.
     * 
     */
    private static String extractResourceToTempFile(String resourcePath) throws IOException {
        try (InputStream in = EyeDetector.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IllegalArgumentException("Resource not found: " + resourcePath);
            }
            File tempFile = File.createTempFile("cascade-", ".xml");
            tempFile.deleteOnExit();
            try (FileOutputStream out = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = in.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                }
            }
            return tempFile.getAbsolutePath();
        }
        catch (IOException e) {
        System.err.println("Error extracting resource: " + e.getMessage());
        // rethrow after logging
        throw e;
        }
    }

    /**
     * Method that extracts all valid images from the file at the input 
     * directory that are in the proper format (.jpg, .jpeg, .png).
     * 
     * @param inputDir input directory that contains the images to process
     * @throws IllegalArgumentException if there are no images in the
     * directory inputted
     * @return a filtered version of the input that contains only .jpg, 
     * .jpeg, .png
     * 
     */
    static File[] extractValidImages(String inputDir) {
        File inputFolder = new File(inputDir);

        // Filter out everything that is not .jpg, .jpeg, .png
        File[] imageFiles = inputFolder.listFiles((dir, name) ->
            // Filter out images that do not pass format requirements
            // Images can only be jpg, jpeg, or png for the model
            name.toLowerCase().endsWith(".jpg") ||
            name.toLowerCase().endsWith(".jpeg") ||
            name.toLowerCase().endsWith(".png"));

        // If the folder is empty or the files are invalid
        if (imageFiles == null || imageFiles.length == 0) {
            // If there are no files, throw error
            throw new IllegalArgumentException ("No images in directory" + inputDir);
        }

        return imageFiles;
    }


    /**
     * Method to only keep the eyes that are above the red line. Takes in
     * all of the eyes detected in the image and filters them based off of
     * specific thresholds so that the 'eyes' detected are between 20%
     * and 50% down the top of the face, and in the 20%-80% of the face
     * horizontally. Alters the original eyes passed in rather than
     * sending back another vector.
     * 
     * @param face the face detected in the image 
     * @param eyes the eyes detected by the model in the image
     */
    static void filterEyes(Rect face, RectVector eyes) {
        RectVector filtered = new RectVector();
        for (int i = 0; i < (int) eyes.size(); i++) {
            // Isolate the eye
            Rect eye = eyes.get(i);

            // Need to recognize only the eyes in a specific region of the face
            // Set threshold as 20% from top of face
            int topThreshold = face.y() + (int)(face.height() * 0.2);
            // Set threshold as 50% from top of face
            int bottomThreshold = face.y() + (int)(face.height() * 0.5);
            // Set threshold as 20% from left edge of the face
            int leftThreshold = face.x() + (int)(face.width() * 0.2);
            // Set threshold as 80% from left edge of the face
            int rightThreshold = face.x() + (int)(face.width() * 0.8);

            // Find the middle coordinate for the eye
            int eyeCenterY = face.y() + eye.y() + (eye.height() / 2);
            int eyeCenterX = face.x() + eye.x() + (eye.width() / 2);

            // Checking conditions to make sure eye is in the right place
            boolean verticallyValid = eyeCenterY >= topThreshold && eyeCenterY <= bottomThreshold;
            boolean horizontallyValid = eyeCenterX >= leftThreshold && eyeCenterX <= rightThreshold;

            // If the eye passes the conditions to be valid
            if (verticallyValid && horizontallyValid) {
                // Add it to the filtered vector
                filtered.push_back(eye);
            }
        }
        // Clear the original eyes vector
        eyes.resize(0);
        // Iterate through the filtered eyes
        for (int i = 0; i < filtered.size(); i++) {
            // Fill in eyes with what is in filtered
            eyes.push_back(filtered.get(i));
        }
    }

    /**
     * Method crops an image so the face is centered. Adds padding around
     * the edges so that the crop is not too tight around the face. Set
     * for the right crop for an ID image, but can be adjusted by changing
     * the targetSize global variable or the padding sizes.
     * 
     * @param image the image file to be cropped
     * @param face the face detected in the image
     * @param eyes the vector of the eyes detected in the image
     * @param targetSize the desired size of the cropped, final image
     */
    static void cropImage(Mat image, Rect face, RectVector eyes, Size targetSize) {
        // Compute average Y of eye centers
        int eyeMiddleY = 0;
        // Iterate through all eyes found
        for (int i = 0; i < (int) eyes.size(); i++) {
            Rect eye = eyes.get(i);
            int eyeY = face.y() + eye.y();
            // Add vertical centers of eyes
            eyeMiddleY += eyeY + (eye.height() / 2);
        }
        // Average vertical position of the eye centers in total image coordinates
        int avgEyeY = (int) (eyeMiddleY / eyes.size());

        // Add padding on all sides of the image so that the crop is not too tight
        int verticalPadding = (int)(targetSize.height() * (2.0 / 3.0));
        int horizontalPadding = (int)(targetSize.width() * (2.0 / 3.0));

        // Make copies of what size you want to crop so you do not alter target size
        int cropHeight = targetSize.height();
        int cropWidth = targetSize.width();

        // Add the padding
        cropHeight += 2 * verticalPadding;
        cropWidth += 2 * horizontalPadding;

        // Compute new coordinates using scale and proportion
        // Face needs to stay centered horizontally
        int cropX = face.x() + face.width() / 2 - cropWidth / 2;
        // Face needs to stay centered vertically
        // Top edge (Y) of the crop box such that when it is resized
        // the eyes align to 1/3rd down in the final output
        int cropY = avgEyeY - Math.round((float) finalEyeY / targetSize.height() * cropHeight);

        // Clamp crop position within image bounds
        // Prevents it from being cut off
        cropX = Math.max(0, Math.min(cropX, image.cols() - cropWidth));
        cropY = Math.max(0, Math.min(cropY, image.rows() - cropHeight));

        // Extract the cropped region from the image
        Rect cropRect = new Rect(cropX, cropY, cropWidth, cropHeight);
        Mat cropped = new Mat(image, cropRect);

        // Resize it to the desired target dimensions (centered and padded as planned)
        resize(cropped, image, targetSize);
    }

    /**
     * Method that adds a line through the center of height of the eye
     * in the image. Alters the image itself rather than returning a copy
     * of the image with the line drawn on it. The line is set
     * to be red.
     * 
     * @param image image to have the line drawn on it
     * @param face face recognized in the image 
     * @param eyes vector of all the eyes recognized in the image
     * 
     */

    static void drawEyeLine(Mat image, Rect face, RectVector eyes) {
        // Variable to save and update the middle of the eye
        int eyeMiddleSum = 0;
        // Iterate through and find the average of the heights of eyes
        for (int i = 0; i < (int) eyes.size(); i++) {
            // Get each eye in the array inputted
            Rect eye = eyes.get(i);
            // Top left corner of the eye rect
            // Add half the eye height to get to the middle
            int eyeMiddle = face.y() + eye.y() + (eye.height() / 2);
            // Add that center to the overall sum of eye middles
            eyeMiddleSum += eyeMiddle;
        }
        // Find the average of the eye heights
        int eyeHeightAverage = (int) (eyeMiddleSum / eyes.size());
        // Draw the line at that height across the entire image
        line(image,
            new Point(0, eyeHeightAverage),
            new Point(image.cols(), eyeHeightAverage),
            // Red
            new Scalar(0.0, 0.0, 255.0, 0.0),
            2, LINE_AA, 0);
    }

    /**
     * Method that draws a rectangle over every eye detected in the image
     * inputted. The rectangles are green. Method alters the image directly
     * rather than returning a copy.
     * 
     * @param image the image of the face that will be drawn on
     * @param face the face detected in the image
     * @param eyes the vector of eyes that will have rectangles drawn over them
     * 
     */

    static void drawEyeRect(Mat image, Rect face, RectVector eyes) {
        // Number of eyes detected in the image
        int eyeCount = (int) eyes.size();
        // Iterate through each eye detected
        for (int i = 0; i < eyeCount; i++) {
            Rect eye = eyes.get(i);
            // Isolate the y coordinate of the eye for the whole image
            int eyeY = face.y() + eye.y();
            // Isolate the x coordinate of the eye for the whole image
            // Coordinate of eye.x/y is based off of the face not the entire image
            int eyeX = face.x() + eye.x();
            // Isolate the height and width of the rectangle around the eye
            int eyeW = eye.width();
            int eyeH = eye.height();

            // Draw a rectangle around each eye
            rectangle(image,
                // Top left corner of the eye
                new Point(eyeX, eyeY),
                // Bottom right corner of the eye
                new Point(eyeX + eyeW, eyeY + eyeH),
                // Green
                new Scalar(0, 255, 0, 0),
                2, LINE_8, 0);
        }
    }

    /**
     * Method that calls other functions that draw on the original image.
     * Calls them after checking that the user inputs the argument to be true.
     * Alters the image directly rather than copying the image and returning
     * the copy.
     * 
     * @param image image from the input directory file to be drawn on
     * @param face face detected in the image
     * @param eyes vector of eyes detected in the face
     * @param enableLines the input argument from the user of whether or not
     * to have the lines drawn on the image
     */
    static void trackChanges(Mat image, Rect face, RectVector eyes, boolean enableLines) {
        // only perform changes and draw lines if input calls for it
        if (enableLines) {
            // draw rectangles around each eye
            drawEyeRect(image, face, eyes);
            // use the eyes in there to draw the eye line
            drawEyeLine(image, face, eyes);
        }
    }

    public static void main(String[] args) {
        // make sure there are only three arguments given from the user
        if (args.length != 3) {
            System.out.println("Usage: java IDPhotoCrop <input_directory> <output_directory> <show_lines>");
            return;
        }

        try {
            // First input is the path for the input folder
            String inputDir = args[0];
            // Second input is the path for the folder for the images to go to
            String outputDir = args[1];
            // Third argument should be true or false for if lines should show up
            String enableLinesInput = args[2];
            // Convert the string argument to boolean
            // Convert string so that it is in all lowercase
            boolean enableLines = Boolean.parseBoolean(enableLinesInput.toLowerCase());
            // Define and initialize the pre-trained models
            String faceCascadePath = extractResourceToTempFile("/haarcascades/haarcascade_frontalface_default.xml");
            String eyeCascadePath = extractResourceToTempFile("/haarcascades/haarcascade_eye.xml");
            CascadeClassifier faceCascade = new CascadeClassifier(faceCascadePath);
            CascadeClassifier eyeCascade = new CascadeClassifier(eyeCascadePath);
            // Array of all of the valid files in the folder
            File[] imageFiles =  extractValidImages(inputDir);

            // Create directory for the output
            new File(outputDir).mkdirs();
            // Go through every image in the input
            for (File imageFile : imageFiles) {
                // Where the final image will be saved
                String outputPath = outputDir + File.separator + imageFile.getName();
                // read and make the image at the path Mat object (to process image)
                Mat image = imread(imageFile.getAbsolutePath());
                // if the image is empty
                if (image.empty()) {
                    // give the user error message
                    System.out.println("Failed to load image: " + imageFile.getName());
                    // continue processing the other images
                    imwrite(outputPath, image);
                    continue;
                }
                // create copy of image, turn it gray so it's easier to distinguish features
                Mat gray = new Mat();
                cvtColor(image, gray, COLOR_BGR2GRAY);
                // define collection for the faces recognized in each of the images
                // creates and fills a collection for every image
                RectVector faces = new RectVector();
                faceCascade.detectMultiScale(gray, faces);
                // if no face has been detected
                if (faces.size() == 0) {
                    // give user error message and continue processing images
                    System.out.println("No face detected in " + imageFile.getName() + ". The original will be in the output folder.");
                    imwrite(outputPath, image);
                    // skip the photo and continue to the next
                    continue;
                }
                // isolate first (and likely only) face for each image
                Rect face = faces.get(0);
                // face is the region of interest
                Mat faceROI = new Mat(gray, face);
                // save collection for any eyes 'recognized' for each face
                RectVector eyes = new RectVector();
                eyeCascade.detectMultiScale(faceROI, eyes);
                // filter eyes to make sure random spots arent detected as 'eyes'
                filterEyes(face, eyes);
                // make sure there are no more than 2
                // // filtering the actual eyes
                // int eyeCount = Math.min(2, (int) eyes.size());
                // storing as a variable because otherwise get lossy conversion error
                int eyeCount = (int) eyes.size();
                // if only one eye was detected in one of the faces
                if (eyeCount == 1){
                    // give user error
                    System.out.println("Only one eye detected in " + imageFile.getName() + ". Calculations may be skewed.");
                }
                // if there are no eyes found in one of the faces
                if (eyeCount == 0) {
                    // give user error if there are no eyes found
                    System.out.println("No eyes detected in " + imageFile.getName() + ". The original photo is in the output folder.");
                    // leave the original image if no eyes are found and skip to the next
                    imwrite(outputPath, image);
                    continue;
                }

                // check if you need to add the lines or not
                trackChanges(image, face, eyes, enableLines);

                // Crop and resize
                cropImage(image, face, eyes, targetSize);

                // send the image to the output directory
                imwrite(outputPath, image);

                // with each processed image, send the file name to the user
                System.out.println("Processed: " + imageFile.getName());
            }

            // signal to the user once all images have been processed
            System.out.println("Done!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
