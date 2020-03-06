all:
	mpic++ svm_train.cpp svm.cpp -o svm_train -std=c++11
	mpic++ svm_predict.cpp svm.cpp -o svm_predict -std=c++11
clean:
	rm svm_train
