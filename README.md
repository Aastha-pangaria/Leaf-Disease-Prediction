# Leaf-Disease-Prediction

This project aims to classify plant leaf diseases using deep learning, leveraging MobileNetV3Small for accurate and efficient image classification. What began as a baseline model has evolved into a fine-tuned, high-performing model — showcasing a structured and iterative approach to deep learning model development.

Plant leaf diseases can drastically impact agricultural productivity. By using deep learning, we can automate early disease detection and enable farmers to take quick action. This project classifies leaf images into 71 categories using transfer learning with MobileNetV3Small.

To get started, clone the repository, install the dependencies using pip install -r requirements.txt, and open the notebook to run it cell by cell. The project uses TensorFlow, Keras, and standard data science libraries such as NumPy, Matplotlib, and Pandas.

We started by developing a basic model using MobileNetV3Small as a frozen feature extractor, pre-trained on ImageNet. We added a lightweight classifier head on top consisting of a GlobalAveragePooling2D layer, Dropout for regularization, and a Dense softmax layer for classification. This model served as our benchmark, reaching a baseline accuracy of around 68.51%.

## Better Model [Model - 1]

The second version (Model 1) maintained the same architecture but incorporated improved data augmentation techniques such as random flipping, zooming, and shearing. We also tuned the training process using callbacks like ReduceLROnPlateau and EarlyStopping. Although the base model was still frozen and the accuracy coming as 67.79% which didn't improve or decrease drastically, these optimizations led to better validation performance.

## Fine-Tuned Model [Model - 2]

The final and most effective version, Model 2, introduced partial fine-tuning. We unfroze the last 20 layers of the MobileNetV3Small model, allowing it to adapt more specifically to our dataset. With this fine-tuning and advanced training strategies, the model achieved an accuracy of approximately 89%, significantly outperforming the earlier versions.

## The Journey Just Begins...

This project marks the beginning of my deep learning journey. From understanding transfer learning to building and fine-tuning CNN models, each step taught me the power of iteration and experimentation in model development. While the accuracy achieved here is a milestone, I believe it's just the surface of what's possible. I am excited to dive deeper into model optimization, interpretability, and deployment to solve real-world problems that matter. Stay tuned — this is just the start of something impactful.
