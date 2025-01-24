# Face Recognition Based Attendance System

This project is a face recognition-based attendance system built using Flask, OpenCV, and machine learning. It captures faces using a webcam, identifies them using a trained model, and records attendance.

## Features

- Capture and register new users
- Train a face recognition model
- Identify users and mark attendance
- View attendance records
- Delete user data

## Requirements

- Python 3.x
- Flask
- OpenCV
- NumPy
- scikit-learn
- pandas
- joblib

## Setup

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd face-recognition-based-attendance-system-master
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the following directories exist:
    - `Attendance`
    - `static`
    - `static/faces`

4. Download the `haarcascade_frontalface_default.xml` file and place it in the project directory.

## Usage

1. Run the Flask app:
    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Use the web interface to:
    - Add new users
    - Start the attendance process
    - View the list of registered users
    - Delete users

## Project Structure

- `app.py`: Main application file containing Flask routes and functions.
- `static/`: Directory containing static files and the trained model.
- `Attendance/`: Directory where attendance records are saved.
- `templates/`: Directory containing HTML templates for the web interface.

## Notes

- Ensure your webcam is connected and working.
- The model is retrained every time a new user is added or deleted.

## System Specification

### Hardware Requirements

1. Webcam
2. Display Monitor
3. Processor: Core i3 and above
4. RAM: 4GB or above
5. Storage: Minimum 40GB or above

### Software Requirements

1. Operating System: Windows 10 and above
2. Programming Language: Python 3.5 version
3. Web Server: Flask
4. Integrated Development Environment (IDE): Python IDE
5. Libraries:
    - OpenCV (cv2)
    - Flask
    - Scikit-Learn (sklearn)
6. Tools:
    - Haar Cascade Classifier (used for face detection)
    - CSV File Viewer (for viewing attendance records)

## Implementation

### Face Detection with OpenCV

The implementation of the Face Recognition Attendance System begins with the use of OpenCV, a popular computer vision library. OpenCV provides robust functionalities for image processing, including face detection. We will leverage OpenCV's pre-trained Haar cascades models to detect faces in images or live video streams.

#### Haar Cascade Algorithm

It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images (where positive images are those where the object to be detected is present, negative are those where it is not). It is then used to detect objects in other images. Luckily, OpenCV offers pre-trained Haar cascade algorithms, organized into categories (faces, eyes and so forth), depending on the images they have been trained on.

### Face Recognition Using KNN Algorithm

Once faces are detected, we will employ the K-Nearest Neighbors (KNN) algorithm for face recognition. KNN is a simple yet effective classification algorithm that can be trained on facial features extracted from images. We will use a dataset of known faces to train the KNN model, allowing it to recognize and classify faces based on similarity metrics.

#### Facial Recognition Algorithm Overview

The biometric facial recognition algorithm follows several steps of image processing:
1. Capture: Gathering physical or communication tests in predefined situations and over a specific time period.
2. Extraction: Extracting data from the captured samples to create templates using facial recognition techniques.
3. Comparison: Comparing the extracted data with existing templates for recognition.
4. Matching: Matching the face features of gathered samples with those from a facial database, usually taking just a second. The Haar Cascade method is utilized for this purpose.

### Web Interface with Flask

For the user interface, we will utilize Flask, a lightweight and versatile web framework in Python. Flask will serve as the backend for our web application, handling requests, routing, and data management. The web interface will allow users to interact with the system, such as capturing images, registering new faces, and viewing attendance records.

### Attendance Data Management Using Pandas

To manage attendance data efficiently, we will integrate Pandas, a powerful data manipulation and analysis library. Pandas will enable us to store, organize, and analyze attendance records in a structured format, such as CSV or Excel files. We can perform tasks like marking attendance, calculating attendance percentages, and generating reports.

### Integration and Testing

During the implementation phase, we will integrate these components seamlessly to ensure smooth functionality. Integration testing will be conducted to validate system behavior, accuracy in face detection and recognition, user interface responsiveness, and data management capabilities. Rigorous testing and validation will be performed to address any issues or inconsistencies.

### Working

1. Registration of Users:
    - Captures multiple images (nimgs) of each user's face using a webcam and saves them in the static/faces folder.
    - Trains a KNN classifier (train_model) using the captured face images to recognize registered users.
2. Taking Attendance:
    - Detects faces in real-time using the webcam (start route).
    - Recognizes the detected faces using the trained KNN model (identify_face).
    - Updates the attendance by adding the recognized user's information to the CSV file (add_attendance).
3. Web Interface:
    - Provides a web interface (home.html) for displaying today's attendance and adding new users.
    - Displays attendance data in a tabular format (<table>).
4. Additional Features:
    - Checks for existing trained models ('static/face_recognition_model.pkl') before taking attendance.
    - Automatically creates required directories and files (Attendance, static, static/faces, Attendance-<date>.csv).

## License

This project is licensed under the MIT License.

