import mindpype as mp
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

class GraphUnitTest():
    def __init__(self):
        self.__session = mp.Session.create()
        self.__graph = mp.Graph.create(self.__session)
        
    def TestCrossValidationFunction(self, raw_data, init_data, init_labels_data, num_classes, num_folds, stat):
        mp_clsf = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
        inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
        predictions = mp.Tensor.create(self.__session, (50,))
        init_tensor = mp.Tensor.create_from_data(self.__session, init_data)
        init_labels = mp.Tensor.create_from_data(self.__session, init_labels_data)
        node = mp.kernels.ClassifierKernel.add_to_graph(self.__graph,inTensor,mp_clsf,predictions,
                                                        num_classes=num_classes,initialization_data=init_tensor,labels=init_labels)
        mean_stat = self.__graph.cross_validate(predictions, folds=num_folds, statistic=stat)
        # self.__graph.verify()
        # self.__graph.initialize()
        # self.__graph.execute()
        return mean_stat
        
        
def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(50,50)
    init_data = np.random.randn(50,50)
    init_labels_data = np.random.randint(0,2, (50,))
    num_folds = 5
    KernelExecutionUnitTest_Object = GraphUnitTest()
    classifier = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    stats = ['accuracy', 'f1', 'precision', 'recall', 'cross_entropy']
    
    for s in stats:
        res = KernelExecutionUnitTest_Object.TestCrossValidationFunction(raw_data, init_data, init_labels_data, num_classes=2, num_folds=num_folds, stat=s)
        skf = StratifiedKFold(n_splits=num_folds)
        mean_stat = 0
        for train_index, test_index in skf.split(init_data, init_labels_data):
            classifier.fit(init_data[train_index], init_labels_data[train_index])
            expected_predicitions = classifier.predict(init_data[test_index])
            if s == 'accuracy':
                stat = accuracy_score(init_labels_data[test_index], expected_predicitions)
            elif s == 'f1':
                stat = f1_score(init_labels_data[test_index], expected_predicitions)
            elif s == 'precision':
                stat = precision_score(init_labels_data[test_index], expected_predicitions)
            elif s == 'recall':
                stat = recall_score(init_labels_data[test_index], expected_predicitions)
            else:
                stat = log_loss(init_labels_data[test_index], expected_predicitions)
            mean_stat += stat
        mean_stat /= num_folds
        assert res == mean_stat  
    
test_execute()