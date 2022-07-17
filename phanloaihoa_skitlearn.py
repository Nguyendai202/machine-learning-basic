from socketserver import DatagramRequestHandler
from sklearn import datasets, neighbors
import numpy as np
from sklearn.model_selection import train_test_split# thu vien tach du lieu 
from sklearn.metrics import accuracy_score # metrics >> acuuracy(hàm trong thư viện metrics) là chỉ số ddooj chính xác 
iris= datasets.load_iris()
iris_X= iris.data# data
iris_y= iris.target# lable( nhan)

randIndex = np.arange(iris_X.shape[0])# tạo ra các điểm dữ liệu vị trí của list từ 1 đến hết có trong data
np.random.shuffle(randIndex)# xáo trộn ngẫu nhiên nhưng đảm bảo sự cân bằng 
# gán lại giá trị sau khi xáo trộn ví trị cho dữ liệu
iris_X = iris_X[randIndex]
iris_y = iris_y[randIndex]
#tách dữ liệu train và tập test
X_train, X_test, y_train, y_test = train_test_split(iris_X,iris_y, test_size=50)# các điểm dữ liệu phải ở numpy array, iris_y là một list chứa nhãn
# test_size=x , x là số lượng dữ liệu  test muốn lấy 
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)# huấn luyện duex liệu và nhãn>> tạo nên model
y_predict= knn.predict(X_test)# dự đoán nhiều điểm mới ( hoặc 1 điểm tùy b )

accuracy=accuracy_score(y_predict,y_test)
print(accuracy)