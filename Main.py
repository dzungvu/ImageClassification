import SVM
import os
import Help

Root = 'db'
SVM_score = 0

for data_direct in os.listdir(Root):

    folder_direct = os.path.join(Root, data_direct)
    print ('Get data train')
    X_train, Y_train = Help.getFeatureFromFolder(folder_direct,'train', 'vgg16_dogs')
    print('Get data test')
    X_test, Y_test = Help.getFeatureFromFolder(folder_direct,'dev', 'vgg16_dogs')

    print('Load model')
    SVM_model = SVM.my_svm(X_train, Y_train)
    print('Predict')
    SVM_pred_labels = SVM_model.predict(X_test)
    print('Calculate score')
    SVM_batch_score = SVM_model.score(X_test, Y_test)

    print('write label')
    with open(os.path.join(folder_direct, "lbval.txt"), 'w+') as predict_result:
        predict_result.write('SVM\n')
        for label in SVM_pred_labels:
            predict_result.write(str(label) + "\n")
            print (str(label))


    print('write score')
    with open(os.path.join(folder_direct, "val.txt"), 'w+') as score_file:
        score_file.write('SVM\n')
        score_file.write(str(SVM_batch_score))
        print (str(SVM_batch_score))



    SVM_score += SVM_batch_score
    print("SVM: ",SVM_batch_score)

print("Mean SVM score: ", SVM_score/len(os.listdir(Root)))