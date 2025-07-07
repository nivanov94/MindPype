import mindpype as mp

class ClassifierUnitTest:
    def __init__(self):
        self.__session = mp.Session.create()

    def TestClassifierPrint(self):
        svm = mp.classifier.Classifier.create_SVM(self.__session)
        print(svm)

def test_execute():
    t = ClassifierUnitTest()
    t.TestClassifierPrint()
    