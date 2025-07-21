import mindpype as mp
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

class GraphUnitTest():        
    # def TestCrossValidationFunction(self, raw_data, init_data, init_labels_data, num_classes, num_folds, stat):
    #     mp_clsf = mp.Classifier.create_LDA(self.__session, shrinkage='auto', solver='lsqr')
    #     inTensor = mp.Tensor.create_from_data(self.__session, raw_data)
    #     predictions = mp.Tensor.create(self.__session, (50,))
    #     init_tensor = mp.Tensor.create_from_data(self.__session, init_data)
    #     init_labels = mp.Tensor.create_from_data(self.__session, init_labels_data)
    #     virtual_tensor = mp.Tensor.create_virtual(self.__session)
    #     node1  = mp.kernels.TransposeKernel.add_to_graph(self.__graph, inTensor, virtual_tensor, init_input=init_tensor, init_labels=init_labels)
    #     node2 = mp.kernels.ClassifierKernel.add_to_graph(self.__graph,inTensor,mp_clsf,predictions,
    #                                                     num_classes=num_classes)
    #     mean_stat = self.__graph.cross_validate(predictions, folds=num_folds, statistic=stat)
    #     self.__graph.verify()
    #     self.__graph.initialize()
    #     self.__graph.execute()
    #     return mean_stat

    def TestCV(self):
        session = mp.Session.create()
        graph = mp.Graph.create(session)

        clf = mp.Classifier.create_LDA(session)

        raw_data = np.random.randn(50,12,200)
        init_data = np.random.randn(50,12,200)
        init_labels = np.concatenate(
            (np.zeros((25,)), np.ones((25,))), axis=0
        )

        raw_data = mp.Tensor.create_from_data(session, raw_data)
        init_data = mp.Tensor.create_from_data(session, init_data)   ## must be 3 dmiensional !!!
        init_labels = mp.Tensor.create_from_data(session, init_labels)

        v1 = mp.Tensor.create_virtual(session)
        v2 = mp.Tensor.create_virtual(session)
        v3 = mp.Tensor.create_virtual(session)
        out_preds = mp.Tensor.create(session, (50,))
        bad = mp.Scalar.create(session, int)

        # csp = mp.kernels.csp.CommonSpatialPatternKernel.add_to_graph(graph, raw_data, v1, initialization_data=init_data, labels=init_labels)
        var = mp.kernels.VarKernel.add_to_graph(graph, raw_data, v2, axis=-1, init_input=init_data, init_labels=init_labels)
        log = mp.kernels.LogKernel.add_to_graph(graph, v2, v3)
        lda = mp.kernels.ClassifierKernel.add_to_graph(graph, v3, clf, out_preds)

        # Target validation must be produced by node in graph
        try:
            cv = graph.cross_validate(bad)
        except KeyError:
            pass

        cv = graph.cross_validate(out_preds, statistic='accuracy')
        cv = graph.cross_validate(out_preds, statistic='f1')
        cv = graph.cross_validate(out_preds, statistic='precision')
        cv = graph.cross_validate(out_preds, statistic='recall')
        cv = graph.cross_validate(out_preds, statistic='cross_entropy')

        # graph.verify()
        # graph.initialize()
        # graph.execute()
        
    def TestGraph(self):
        session = mp.Session.create()
        graph = mp.Graph.create(session)

        src = mp.source.InputLSLStream.create_marker_uncoupled_data_stream(session, active=False)

        inA = mp.Scalar.create_from_value(session, 5)
        inB = mp.Scalar.create_from_value(session, 5)
        in_src = mp.Scalar.create_from_source(session, int, src)

        out1 = mp.Scalar.create(session, int)
        out2 = mp.Scalar.create(session, int)
        out_and = mp.Scalar.create(session, bool)

        node1 = mp.kernels.AdditionKernel.add_to_graph(graph, inA, inB, out1)

        node2 = mp.kernels.AndKernel.add_to_graph(graph, out1, out2, out_and)

        node_src = mp.kernels.DivisionKernel.add_to_graph(graph, in_src, out1, out2)

        # Can't call inactive stream for source data
        try:
            graph.execute()
        except RuntimeError:
            pass
        
    def TestGraphInvalid(self):
        session = mp.Session.create()
        graph = mp.Graph.create(session)
        graph1 = mp.Graph.create(session)
        graph2 = mp.Graph.create(session)

        inA = mp.Scalar.create_from_value(session, 10)
        inB = mp.Scalar.create_from_value(session, 20)
        out = mp.Scalar.create(session, int)

        out1 = mp.Scalar.create(session, int)
        out2 = mp.Scalar.create(session, int)

        node1 = mp.kernels.AdditionKernel.add_to_graph(graph, inA, inB, out)
        node2 = mp.kernels.AdditionKernel.add_to_graph(graph, inA, inB, out)

        node3 = mp.kernels.AdditionKernel.add_to_graph(graph1, out1, inB, out2)
        node4 = mp.kernels.AdditionKernel.add_to_graph(graph1, out2, inB, out1)
        
        # Can't have multiple nodes write to single data object
        try:
            graph.verify()
        except ValueError:
            pass

        # Invalid graph
        try:
            graph1.verify()
        except ValueError:
            pass

        in_clf = mp.Tensor.create_from_data(session, [1,2,3,4])
        preds = mp.Tensor.create_from_data(session, [1,2,1,2])
        clf = mp.Classifier.create_LDA(session)
        node_clf = mp.kernels.ClassifierKernel(graph2, in_clf, clf, preds, output_probs=50, num_classes=2)

        graph2.verify()
        graph2.initialize()
        graph2.execute()
        
    def TestUpdateGraph(self, raw_data, init_data, init_labels):
        session = mp.Session.create()
        graph = mp.Graph.create(session)

        out = mp.Tensor.create(session, (50,))
        clf = mp.Classifier.create_LDA(session)

        raw_data = mp.Tensor.create_from_data(session, raw_data)
        init_data = mp.Tensor.create_from_data(session, init_data)
        init_labels = mp.Tensor.create_from_data(session, init_labels)

        node = mp.kernels.classifier.ClassifierKernel.add_to_graph(graph, raw_data, clf, out, initialization_data=init_data, labels=init_labels)

        graph.initialize()

        init_data = np.zeros((50,50))
        init_labels = np.zeros((50,))  ## is this how this works?

        graph.update()
        # node.execute()
        # graph.execute()
    
def test_execute():
    np.random.seed(44)
    raw_data = np.random.randn(50,50)
    init_data = np.random.randn(50,50)
    init_labels_data = np.random.randint(0,2, (50,))

    KernelExecutionUnitTest_Object = GraphUnitTest()
    # classifier = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    # stats = ['accuracy', 'f1', 'precision', 'recall', 'cross_entropy']
    
    # init_after_transpose = np.transpose(init_data)

    # for s in stats:
    #     res = KernelExecutionUnitTest_Object.TestCrossValidationFunction(raw_data, init_after_transpose, init_labels_data, num_classes=2, num_folds=num_folds, stat=s)
    #     skf = StratifiedKFold(n_splits=num_folds)
    #     mean_stat = 0
    #     for train_index, test_index in skf.split(init_after_transpose, init_labels_data):
    #         classifier.fit(init_after_transpose[train_index], init_labels_data[train_index])
    #         expected_predicitions = classifier.predict(init_after_transpose[test_index])
    #         if s == 'accuracy':
    #             stat = accuracy_score(init_labels_data[test_index], expected_predicitions)
    #         elif s == 'f1':
    #             stat = f1_score(init_labels_data[test_index], expected_predicitions)
    #         elif s == 'precision':
    #             stat = precision_score(init_labels_data[test_index], expected_predicitions)
    #         elif s == 'recall':
    #             stat = recall_score(init_labels_data[test_index], expected_predicitions)
    #         else:
    #             stat = log_loss(init_labels_data[test_index], expected_predicitions)
    #         mean_stat += stat
    #     mean_stat /= num_folds
    #     assert res == mean_stat  

    KernelExecutionUnitTest_Object.TestCV()
    # KernelExecutionUnitTest_Object.TestCVInvalid()
    KernelExecutionUnitTest_Object.TestGraph()
    KernelExecutionUnitTest_Object.TestGraphInvalid()
    KernelExecutionUnitTest_Object.TestUpdateGraph(raw_data, init_data, init_labels_data)
