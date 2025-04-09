from siamese_model import SiameseModel
import matplotlib.pyplot as plt

# Initialize siamese model
siamese_model = SiameseModel()

# Train/test model
siamese_model.train_model()
siamese_model.eval_model()
print(
    f"Testing dataset has an accuracy of {siamese_model.test_accuracy} and loss of {siamese_model.test_loss}"
)
print("Confusion matrix of predictions:")
print(siamese_model.test_confusion_matrix)

# Plot training/validation learning curves
train_loss = siamese_model.training_history.history["loss"]
val_loss = siamese_model.training_history.history["val_loss"]

plt.figure(figsize=(12, 8))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation/Testing Loss")
plt.title("Learning Curves vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Categorical Cross Entropy")
plt.legend()
plt.show()
