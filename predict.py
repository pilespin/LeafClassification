
import numpy as np
from tfHelper import tfHelper
import data

tfHelper.log_level_decrease()
# tfHelper.numpy_show_entire_array(28)
np.set_printoptions(linewidth=200)


print ("Load data ...")
X_pred, X_id, label = data.load_data_predict()

# X_pred = X_pred.reshape(X_pred.shape[0], 28, 28, 1)
# X_pred = X_pred.astype('float32')
# X_pred /= 255


model = tfHelper.load_model("model")

######################### Predict #########################
predictions = model.predict(X_pred)

print(predictions)

# for i in predictions:
# 	str1 = ""
# 	for elem in i:
# 		str1 += ',' + str(elem) 
# 	print (str1)
# 	# indexMax = np.argmax(i)
# 	# print (str('----- ') + str(indexMax) + str(' -----'))
# exit(0)

# print("OUTPUT")
AllPrediction = []
for i in range(predictions.shape[0]):
	indexMax = np.argmax(predictions[i])
	AllPrediction.append(indexMax)

with open("output", "w+") as file:
	for line in label[:-1]:
		file.write(line + ",")
	else:
		file.write(str(label[-1]))
		file.write("\n")

	# for line, id in zip(AllPrediction, X_id):
		# file.write(str(id) + "," + str(line) + "\n")

	for line, id in zip(predictions, X_id):
		str1 = ""
		for elem in line:
			str1 += ',' + str(round(elem, 1)) 
		file.write(str(id) + str1 + "\n")
		# print (str1)
