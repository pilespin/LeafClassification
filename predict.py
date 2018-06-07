
import numpy as np
from tfHelper import tfHelper
import data

tfHelper.log_level_decrease()
# tfHelper.numpy_show_entire_array(28)
# np.set_printoptions(linewidth=200)


print ("Load data ...")
_, X_id, label = data.load_data_predict()

X_pred = tfHelper.get_dataset_with_one_folder('classed/.None', 'L')
X_pred = data.normalize(X_pred)

model = tfHelper.load_model("model_img")
# model = tfHelper.load_model("model")

######################### Predict #########################
predictions = model.predict(X_pred)

# print(predictions)
# exit (0)


# All features
with open("output_img_detailed", "w+") as file:
	# Head
	for line in label[:-1]:
		file.write(line + ",")
	else:
		file.write(str(label[-1]))
		file.write("\n")

	for line, id in zip(predictions, X_id):
		str1 = ""
		for elem in line:
			str1 += ',' + str(round(elem, 3)) 
		file.write(str(id) + str1 + "\n")

# One features

AllPrediction = []
for i in predictions:
	indexMax = np.argmax(i)
	# print(indexMax)
	AllPrediction.append(indexMax)

with open("output_img", "w+") as file:
	# Head
	for line, id in zip(AllPrediction, X_id):
		file.write(str(id) + "," + str(line) + "\n")
