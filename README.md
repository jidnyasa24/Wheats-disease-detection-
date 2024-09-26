# **Wheat Crop Disease Detection Using Deep Learning**

## **Project Overview**


Wheat crops are susceptible to various diseases that can significantly reduce yield if not detected early. Traditionally, disease detection relies on manual observation by experts, which can be time-consuming and prone to error. Given the vast size of wheat fields, there is a growing need for automated disease detection systems to assist farmers in quickly diagnosing and responding to potential threats.


The aim of this project was to develop an automated system to detect wheat crop diseases using deep learning models. The system needed to be accurate, easy to use, and capable of providing results in real-time. Additionally, it was required to integrate environmental data (through a weather API) to provide context on disease susceptibility based on the current weather conditions.


To achieve this, I built a convolutional neural network (CNN) using the **VGG19** architecture, which is well-suited for image classification tasks. The model was trained on a dataset of images of healthy and diseased wheat crops. For deployment, I used **Flask**, a lightweight web framework, to create a web interface where users can upload images of wheat leaves for real-time disease detection.

I also integrated a weather API to retrieve real-time weather data such as temperature and humidity, which can affect disease development. This integration allows the system to provide additional context and warnings when weather conditions are favorable for the spread of certain diseases.


The final model achieved an accuracy of **XX%** on the test set. The system provides accurate predictions of wheat crop diseases and is accessible through a simple web interface. By combining disease detection with real-time weather data, the system offers farmers a powerful tool for managing their crops and responding quickly to potential disease outbreaks.

---

## **Project Methodology**

### **1. Data Collection and Preprocessing**
The dataset for this project consisted of images of wheat crop leaves, both healthy and infected with various diseases. The images were collected from publicly available datasets related to plant pathology and agriculture.

Key preprocessing steps included:
- **Resizing Images**: All images were resized to 224x224 pixels, which is the input size required for the VGG19 model.
- **Normalization**: Pixel values were normalized to a range between 0 and 1.
- **Data Augmentation**: Techniques such as rotation, flipping, and zooming were applied to artificially increase the size of the dataset and improve the model’s generalization.

### **2. Model Architecture: VGG19**
For the model, I used **VGG19**, a pre-trained convolutional neural network that has been proven to perform well on image classification tasks. VGG19 is known for its deep architecture, which can capture fine details in images, making it ideal for detecting subtle differences between healthy and diseased crops.

The pre-trained VGG19 model was used as the base, and I added a fully connected dense layer for the classification task. The model's final layer uses **softmax** activation to output probabilities for each disease class, allowing the system to classify images into multiple categories (such as healthy or diseased).

**Key features of the model include:**
- **Transfer Learning**: By using the pre-trained VGG19 model, the training process was faster and required less data compared to training a model from scratch.
- **Fine-tuning**: I unfroze some of the deeper layers of the VGG19 model to fine-tune it for the specific task of wheat disease detection.

### **3. Model Training**
The model was trained using the following configurations:
- **Optimizer**: Adam optimizer, known for its efficient performance in deep learning tasks.
- **Loss Function**: Categorical cross-entropy, which is appropriate for multi-class classification.
- **Metrics**: Accuracy was used to track the model's performance, and additional metrics such as precision and recall were used to evaluate the model’s ability to handle imbalanced data.

Training was conducted for **X epochs** with a batch size of **Y**, and early stopping was implemented to prevent overfitting.

### **4. Flask Deployment**
Once the model was trained, I used **Flask** to deploy it as a web application. Flask provides a simple and lightweight framework that allowed me to create a web interface where users can upload images of wheat crops for real-time disease detection.

- **Image Upload**: Users can upload images through the web interface, which are then passed to the trained VGG19 model for inference.
- **Real-time Predictions**: The model processes the image and returns a prediction about the health of the crop, classifying it as either healthy or diseased.

### **5. Weather API Integration**
To enhance the system's usefulness, I integrated a weather API (e.g., **OpenWeatherMap**) to provide real-time weather data alongside the disease prediction. This integration helps farmers understand how environmental conditions such as temperature, humidity, and wind speed might impact disease progression.

- **Weather Data**: The API fetches real-time data for the user’s location, which is displayed alongside the disease prediction.
- **Contextual Insights**: Based on the weather data, the system can offer additional advice, such as alerting users when weather conditions are favorable for disease outbreaks.

### **6. Final Output**
The final output of the system includes:
- **Disease Prediction**: A classification of the uploaded image into one of the predefined categories (e.g., healthy, rust, powdery mildew).
- **Weather Information**: Current weather conditions are displayed alongside the prediction, providing users with a more complete understanding of their crop's health.

---

## **Technologies Used**
- **Deep Learning Framework**: TensorFlow and Keras
- **Pre-trained Model**: VGG19
- **Web Framework**: Flask
- **Weather API**: OpenWeatherMap (or a similar weather API)
- **Frontend**: HTML, CSS, and JavaScript for the web interface
