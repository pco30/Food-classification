# Food-classification

The code performs transfer learning for food classification using the InceptionResNetV2 model. It starts by importing necessary libraries and defining the base directory for the dataset. It loads and processes images from training, validation, and evaluation directories, and prints the number of images in each category. Data augmentation is applied to the training and validation datasets using `ImageDataGenerator`.

The InceptionResNetV2 model is loaded with pre-trained weights and a custom classification head is added. The model is compiled and trained with initial frozen layers, followed by fine-tuning with a lower learning rate. Early stopping and model checkpointing are used during training to save the best model. The model is then evaluated on the test set, and predictions are made on the evaluation images. Finally, the results are visualized, and a classification report and confusion matrix are generated to assess performance.
