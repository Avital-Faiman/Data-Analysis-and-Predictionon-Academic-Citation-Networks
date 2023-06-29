import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

loss_train = np.load('loss_train.npy')
loss_train = loss_train.tolist()
loss_val = np.load('loss_val.npy')
loss_val = loss_val.tolist()
acc_train = np.load('acc_train.npy')
acc_train = acc_train.tolist()
acc_val = np.load('acc_val.npy')
acc_val = acc_val.tolist()
y_train = np.load('y_train.npy')
y_train = y_train.tolist()
y_val = np.load('y_val.npy')
y_val = y_val.tolist()
pred_val = np.load('pred_val.npy')
pred_val = pred_val.tolist()
pred_train = np.load('pred_train.npy')
pred_train = pred_train.tolist()

plt.plot(loss_train, label="Train", color='hotpink', alpha=0.5)
plt.plot(loss_val, label="Val", color="seagreen")
plt.title("Train and Val loss by epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(acc_train, label="Train", color='hotpink', alpha=0.5)
plt.plot(acc_val, label="Val", color="seagreen")
plt.title("Train and Val accuracy by epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('acc.png')
plt.show()



actual_val = y_val
predicted_val = pred_val

confusion_matrix = metrics.confusion_matrix(actual_val, predicted_val)
plt.imshow(confusion_matrix, cmap='Purples')
plt.title('Val Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Ground Truth Labels')
plt.xticks(range(1, 41, 2), rotation=45, fontsize=7)
plt.yticks(range(1, 41, 2), rotation=45, fontsize=7)
plt.colorbar()
plt.show()

actual_train = y_train
predicted_train = pred_train

confusion_matrix = metrics.confusion_matrix(actual_train, predicted_train)
plt.imshow(confusion_matrix, cmap='Purples')
plt.title('Train Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Ground Truth Labels')
plt.xticks(range(1, 41, 2), rotation=45, fontsize=7)
plt.yticks(range(1, 41, 2), rotation=45, fontsize=7)
plt.colorbar()
plt.show()

