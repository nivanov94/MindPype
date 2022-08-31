"""Important note for those editing the kernels within the GUI. There are 5 places where
a kernel name needs to be edited, and they need to be the EXACT same in each place (capitals, etc)

This also applies to users wanting to add a kernel. 

#TODO: These could AND DEFINITELY SHOULD be within a single dictionary

Firstly, self.data defines which kernels are available for user selection.
    It is a dict with the following strucutre:
        {'Type_of_node_1': ['kernel_of_type_1_1', 'kernel_of_type_1_2'], 'Type_of_node_2': [.....]}

Secondly, self.kernel_parameters defines the parameters the user will be able to select.
    Structure is a dict with the following structure
    {'Type_of_node_1': {'kernel_of_type_1_1': {Name of Parameter to appear to user: user input type}}}
            int for integer, str for string, float for float, bool for boolean, bytearray for file selection (BYTEARRAY DOESN"T WORK YET, USE str FOR MANUAL PATH ENTRY)

Thirdly, self.library_class_methods defines the code that will be written to the output file
    {'Type_of_node_1': {'kernel_of_type_1_1': string command}}

Fourth, self.input_num defines the number of inputs for a particular node
    structure is a dict with the following structure
        {num_of_inputs_0: ['kernel_of_type_1_1'], num_of_inputs_1: ['kernel_of_type_1_2', 'kernel_of_type_1_3']}

Finally, self.output_num defines the number of outputs for a particular node
    it is structured the same as self.input_num"""

import warnings
from math import atan2, cos, pi, sin

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
import sys

from numpy import isin

class mwindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #Setting main window title on launch

        self.setWindowTitle("BCIPY Graph Builder")
        self.setGeometry(300,200,640,520)
        
        # Initial Object options available for users to create
        self.data = {
            'Filter' : ['Butterworth', 'Chebyshev Type-I', 'Chebyshev Type-II' 'Elliptic', 'Bessel'],
            'Classifier': ['LDA', 'SVM', 'Logistic Regression', 'Custom'],
            'Simple Operations': ['Add', 'Subtract', 'Multiply', 'Divide', 'Absolute', 'Mean', 'Log', 'Min', 'Max', 'Standard Deviation', 'Var', 'Threshold'],
            'Riemann Operations' : ['Riemann Distance', 'Riemann Mean', 'Riemann MDM Classifier'],
            'Other Operations': ['CDF', 'CSP', 'Feature_normalization', 'Covariance', 'Reduced Sum', 'Tangent Space', 'Running Average', 'XDawn Covariances', 'Z Score'],
            'Logical' : ['And', 'Or', 'Xor' 'Not', 'Equal', 'Greater Than', 'Less Than'],
            'Manipulate Data': ['Extract', 'Concatenate', 'Set Data', 'Stack', 'Transpose'],
            'Tensor': ['Tensor From Data', 'Tensor From Handle', 'Virtual Tensor', 'Non-Virtual Tensor'],
            'Input Data': ['Lab Streaming Layer Input', 'Epoched MAT File', 'Continuous MAT File', 'Class Separated MAT File'],
            'Scalar':['Scalar From Value', 'Scalar From Handle', 'Virtual Scalar', 'Non-Virtual Scalar'],
            'Array': ['Array From Data', 'Array From Handle', 'Virtual Array', 'Non-Virtual Array']            
            }

        # Required Object Parameters, structure is as follows
            # Node Type: {Kernel Type: {Name of Parameter to appear to user: user input type}}
            # int for integer, str for string, float for float, bool for boolean, bytearray for file selection (BYTEARRAY DOESN"T WORK YET, USE str FOR MANUAL PATH ENTRY)
        self.kernel_parameters = {
            'Filter': 
                {
                    'Butterworth': 
                        {
                            'Order': int, 
                            'Critical Frequencies': str, 
                            'Bandpass Type': str,
                            'output': str,
                            'Sampling Frequency': float, 
                            'Implementation': str},

                    'Chebyshev Type-I': 
                        {
                            'Order': int,
                            'Maximum Ripple': float, 
                            'Bandpass Type': str, 
                            'output': str, 
                            'Sampling Frequency': float
                        },
                    
                    'Chebyshev Type-II': 
                        {
                            'Order': int,
                            'Minimum Attenuation': float, 
                            'Bandpass Type': str, 
                            'output': str, 
                            'Sampling Frequency': float
                        },
                    'Elliptic':
                        {
                            'Order': int,
                            'Maximum Ripple': float, 
                            'Minimum Attenuation': float,
                            'Bandpass Type': str, 
                            'output': str, 
                            'Sampling Frequency': float
                        },
                    'Bessel':
                        {
                            'Order': int,
                            'Critical Frequencies': str,
                            'Bandpass Type': str, 
                            'output': str, 
                            'Critical Frequency Normalization': str,
                            'Sampling Frequency': float
                        }
                },
            
            'Classifier':
                {
                    'LDA': 
                        {
                            'Solver': str,
                            'Shrinkage': str,
                            'Priors': str,
                            'Number of components': int,
                            'Store Covariance': bool,
                            'Tolerance': float,
                            'Training Data': str
                        }, 
                    
                    'SVM': 
                        {
                            'Regularization Parameter': float, 
                            'Kernel': str,
                            'Degree': int,
                            'Gamma' : str,
                            'Independent Term in Kernel': float,
                            'Shrinking': bool,
                            'Probability Estimates': bool,
                            'Tolerance': float,
                            'Cache Size': float,
                            'Class Weight': str,
                            'Verbose': bool,
                            'Hard iterations limit': int,
                            'OVR/OVO': str,
                            'break_ties': bool,
                            'Random State': str
                        },     
                    'Logistic Regression':
                        {   
                            'Penalty': str,
                            'Dual' : bool,
                            'tol': float,
                            'C':float,
                            'Fit Intercept':bool,
                            'Intercept Scaling': float,
                            'Class Weight': str,
                            'Random State': int,
                            'Solver': str,
                            'Max Iterations': int,
                            'Multi_class': str,
                            'Verbose': int,
                            'Warm Start':bool,
                            'N_Jobs': int,
                            'Elastic-Net Mixing Parameter':float
                        },
                    'Custom':
                        {

                        }
                },
            
            'Simple Operations':
                {
                    'Add': {}, 
                    'Subtract': {}, 
                    'Multiply': {}, 
                    'Divide': {},
                    'Absolute': {},
                    'Mean': 
                        {
                            'Axis': str
                        },
                    'Log': {},
                    'Min': {},
                    'Max': {},
                    'Standard Deviation': {},
                    'Var': 
                        {
                            'Axis': str,
                            'Delta Degrees of Freedom': int
                        }, 
                    'Threshold':
                        {
                            "Thresh": float
                        }
                },
            'Riemann Operations':
                {
                    'Riemann Distance': {},
                    'Riemann Mean':{'Weights': str},
                    'Riemann MDM Classifier':
                        {
                            'Training Data File Path/Object name': str, 
                            'Training Labels File Path/Object name': str,
                            'Data Tensor name in file (MAT FILE ONLY)': str,
                            'Label Tensor name in file (MAT FILE ONLY)': str,
                        }
                },

            'Other Operations':
                {
                    'CDF':
                        {
                            'Dist': str,
                            'Location Parameter': str,
                            'Scale Parameter': str,
                            'Distribution Shape Parameter': str
                        }, 
                    'CSP':
                        {
                            'Number of Filters': int,
                            'Training Data File Path/Object name': str, 
                            'Training Labels File Path/Object name': str,
                            'Data Tensor name in file (MAT FILE ONLY)': str,
                            'Label Tensor name in file (MAT FILE ONLY)': str, 
                        }, 
                    'Feature_normalization':
                        {
                            'Method': str,
                            'Training Data File Path/Object': str, 
                            'Training Labels File Path/Object name': str,
                            'Data Tensor name in file (MAT FILE ONLY)': str,
                            'Label Tensor name in file (MAT FILE ONLY)': str,
                        }, 
                    'Covariance':
                        {
                            'Regularization': float
                        }, 
                    'Reduced Sum': 
                        {
                            'Axis': str,
                            'Keep Dimensions': bool
                        }, 
                    'Tangent Space':
                        {

                        }, 
                    'Running Average':
                        {
                            'Running Average Capacity': int,
                            'Axis': str
                        }, 
                    'XDawn Covariances':
                        {
                            'Number of Filters': int,
                            'Apply Filters': bool,
                            'Classes': str,
                            'Estimator': str,
                            'XDawn Estimator': str,
                            'Baseline Covariance': str,
                            'Training Data File Path/Object name': str, 
                            'Training Labels File Path/Object name': str,
                            'Data Tensor name in file (MAT FILE ONLY)': str,
                            'Label Tensor name in file (MAT FILE ONLY)': str,
                        }, 
                    'Z Score':
                        {
                            'Training Data File Path/Object Name': str, 
                            'Data Tensor name in file (MAT FILE ONLY)': str,
                        }
                },

            'Logical' : 
                {
                    'And':{},
                    'Or':{}, 
                    'Xor':{},
                    'Not':{}, 
                    'Equal':{}, 
                    'Greater Than':{}, 
                    'Less Than':{},
                },
            
            'Manipulate Data': 
                {
                    'Extract':
                        {
                            'Indices': str,
                            'Reduce Dimensions': bool
                        },
                    'Concatenate':{'Axis': str},
                    'Set Data':{'Axis': str, 'Index': str}, 
                    'Stack':{'Axis': str}, 
                    'Transpose':{'Axes': str},
                },
            
            'Tensor':
                {
                    'Tensor From Data': 
                        {
                            'Shape': str, 'Data': str
                        }, 
                    'Tensor From Handle': 
                        {
                            'Handle': str
                        }, 
                    'Virtual Tensor': 
                        {
                            'Shape': str
                        }, 
                    'Generic Tensor': 
                        {
                            'Shape': str
                        }
                },
            
            'Input Data': 
                {
                    'Lab Streaming Layer Input': 
                        {
                            'Stream Source': str,
                            'Prop': str,
                            'Prop Value': str,
                            'Ns': float,
                            'Labels File': str,
                            'Channels':str,
                            'Marker': bool,
                            'Marker_FMT': str
                        }, 
                    'Epoched MAT File':
                        {
                            'File Name': str, 
                            'Path': str, 
                            'Label Varname Map': str, 
                            'Dimensions': str,
                            
                        },
                    'Continuous MAT File':
                        {
                            'Data file': str, 
                            'Labels File': str, 
                            'Data name in MAT File': str, 
                            'Labels name in MAT File': str,
                            'Event Duration (sample number)': int,
                            'Start Index': int,
                            'End Index': int,
                            'Relative Start': int,
                        },
                    'Class Separated MAT File':
                        {
                            
                            'Data file': str, 
                            'Labels File': str, 
                            'Data name in MAT File': str, 
                            'Labels name in MAT File': str,
                            'Number of Classes': int,
                            'Event Duration (sample number)': int,
                            'Start Index': int,
                            'End Index': int,
                            'Relative Start': int,
                        }
                },
            
            'Scalar':{'Scalar From Value': {'Value': int}, 'Scalar From Handle': {'Data Type': str, 'Handle': str}, 'Virtual Scalar': {'Data Type': str}, 'Generic Scalar': {'Data Type': str}},
            
            'Array': {'Generic Array': {'Capacity': int, 'Element Template': str}, 'Circle Buffer': {'Capacity': int, 'Element Template': str}}
        }

        filler = ''
        # formatted strings for compilation
        self.library_class_methods = {
            'Filter': 
            {
                'Butterworth': f'{filler} = Filter.create_butter({filler}, {filler})\nfilter.create_node(',
                'Cheby': 'g = Filter.create_gaussian('
                
                
            },
            
            'Classifier':
                {
                    'LDA': 'LDA.command(',
                    'CSP': 'CSP.command(', 'New_classifier': 'f =  New_classifier(sklearn.new_classifier({filler}), '
                }
        }


        self.edges = {}

        self.user_params = []
        
        self.create_ui()


        # Configuring button clicks available on main page
        self.add_Button.clicked.connect(lambda: UIFunctions.collect_params(self, True))
        self.add_Button.clicked.connect(lambda: Node.create_node(self.scene, self.Node_add.currentText(), self.Kernel_Selector.currentText(), self.user_params))
        self.Node_add.currentIndexChanged.connect(lambda: UIFunctions.UpdateKernelCombo(self))
        self.set_Props_Button.clicked.connect(lambda: UIFunctions.dialog_layout(self))
        self.cancel_dialog.clicked.connect(lambda: UIFunctions.reset_dialog(self))
        self.ok_dialog.clicked.connect(lambda: UIFunctions.confirm_dialog(self))
        self.compile_button.clicked.connect(lambda: UIFunctions.compile_graph(self))
        
        # Removed trial/class setting window since blocks are no longer used
        """# self.compiler_window_button_box.button(QDialogButtonBox.Save).clicked.connect(lambda: UIFunctions.set_trials_layout(self))
        #self.setNumberOfTrials.clicked.connect(lambda: UIFunctions.set_trials_layout(self))
        #self.compile.clicked.connect(lambda:UIFunctions.compile_graph(self))
        #self.confirmTrialNumbers.clicked.connect(lambda: UIFunctions.collect_trial_params(self))
        #Node.mouseDoubleClickEvent(Node, self.scene.removeItem(Node))
        #Node.mouseDoubleClickEvent(print(self.scene.selectedItems()))
        #self.ui.Btn_Toggle.clicked.connect(lambda: UIFunctions.toggleMenu(self,250, True))"""
        
        self.show()

    def create_ui(self):
        #Create Main Horizontal Layout
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        #Create Graph Creator Window (left side of screen)
        self.scene = scene()
        self.view = QGraphicsView(self.scene, self)
        self.view.setGeometry(0, 0, 640, 440)
        

        #Create Node Creator Window (right side of screen)

        self.right_widget = QWidget()
        self.right_side = QVBoxLayout()
        self.right_widget.setLayout(self.right_side)
        
        #Create Menu to select what type of node to create

        self.model = QStandardItemModel()

        self.Node_add = QComboBox()         
        self.Node_add.setModel(self.model)

        self.Kernel_Selector = QComboBox()
        self.Kernel_Selector.setModel(self.model)

        for k, v in self.data.items():
            node_type = QStandardItem(k)
            self.model.appendRow(node_type)
            for value in v:
                kernel = QStandardItem(value)
                node_type.appendRow(kernel)

        
        UIFunctions.UpdateKernelCombo(self)
        UIFunctions.keyPressEvent(self, QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))

        # Add dropdowns to window and 
        self.right_side.addWidget(self.Node_add)
        self.right_side.addWidget(self.Kernel_Selector)
        
        self.buttons = QWidget(self.right_widget)
        self.button_layout = QHBoxLayout()
        self.buttons.setLayout(self.button_layout)

        self.add_Button = QPushButton("Add Node")
        self.set_Props_Button = QPushButton("Set Properties")
        self.add_Button.setEnabled(False)
        
        self.right_side.addWidget(self.buttons)
        self.button_layout.addWidget(self.set_Props_Button)
        self.button_layout.addWidget(self.add_Button)

        self.compile_button = QPushButton('Compile')
        

        self.main_layout.addWidget(self.view)
        self.main_layout.addWidget(self.right_widget)
        self.right_side.addWidget(self.compile_button)

        #Property Setter Dialog Layout
        self.property_setter = QDialog(self.main_widget)
        #self.button_box = QDialogButtonBox(self.property_setter)
        #self.button_box.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.cancel_dialog = QPushButton('Cancel')
        self.ok_dialog = QPushButton('Confirm')
        self.diag_layout = QVBoxLayout()


        # Graph Compiling Dialog Layout (removed once blocks were removed)
        """#self.compiler_window = QDialog(self.main_widget)
        #self.setTrialsWindow = QDialog(self.main_widget)
        #self.BlockParam1Box = QHBoxLayout()
        #self.TrialNumPrompt = QLabel("Number of Classes")
        #self.TrialNumUserInput = QSpinBox()
        #self.compiler_window_button_box = QDialogButtonBox(self.compiler_window)
        #self.VertCompileLayout = QVBoxLayout()
        #self.compiler_window.setMinimumSize(300,150)
        #self.compiler_window.setLayout(self.VertCompileLayout)
        #self.BlockParam1Box.addWidget(self.TrialNumPrompt)
        #self.BlockParam1Box.addWidget(self.TrialNumUserInput)
        #self.VertCompileLayout.addLayout(self.BlockParam1Box)
        #self.VertCompileLayout.addWidget(self.compiler_window_button_box)
        #self.compiler_window_button_box.addButton(QDialogButtonBox.Save)
        #self.compiler_window_button_box.button(QDialogButtonBox.Save).setText("Set Trials")
        #self.compiler_window_button_box.addButton(QDialogButtonBox.Cancel)
        #self.compiler_window_button_box.button(QDialogButtonBox.Cancel).setText("Compile Graph")
        #self.compiler_window_button_box.button(QDialogButtonBox.Save).setMinimumWidth(150)
        #self.compiler_window_button_box.button(QDialogButtonBox.Cancel).setMinimumWidth(150)
        

       
        #Trial configuration dialog layout
        #self.VertTrialsLayout = QVBoxLayout(self.setTrialsWindow)
        #self.setTrialsWindow.setLayout(self.VertTrialsLayout)
        #self.confirmTrialNumbers = QPushButton("Confirm")
#
        #self.cancelNumberOfTrials = QPushButton("Cancel")
        
        #variables to store
        self.num_classes = 0
        self.trials_per_class = []"""

class scene(QGraphicsScene):
    startItem = newConnection = None
    def __init__(self):
        
        self.input_num = {
            0: 
                [
                    'Lab Streaming Layer Input', 
                    'Epoched MAT File', 
                    'Continuous MAT File', 
                    'Epoched MAT File', 
                    'Tensor From Data', 
                    'Tensor From Handle', 
                    'Scalar From Value', 
                    'Scalar From Handle', 
                    'Array From Data', 
                    'Array From Handle'],
            
            1: 
                [
                    'Absolute',
                    'Bessel', 
                    'Chebyshev Type-I', 
                    'Chebyshev Type-II', 
                    'Elliptic', 
                    'Butterworth',
                    'Feature Normalization', 
                    'Extract', 
                    'Covariance', 
                    'CSP' 
                    'CDF', 
                    'Butterworth', 
                    'Gaussian', 
                    'Virtual Tensor', 
                    'Non-Virtual Tensor', 
                    'Virtual Scalar', 
                    'Non-Virtual Scalar', 
                    'LDA', 
                    'SVM', 
                    'Logistic Regression', 
                    'Log',
                    'Max',
                    'Min',
                    'Reduced Sum',
                    'Riemann MDM Classifier',
                    'Riemann Mean',
                    'Riemann Potato',
                    'Running Average',
                    'Set Data',
                    'Stack',
                    'Standard Deviation',
                    'Tangent Space',
                    'Threshold',
                    'Transpose',
                    'Var',
                    'XDawn Covariance',
                    'Z Score'
                    ],

            2:
                [
                    'Equal', 
                    'Concatenate', 
                    'Add', 
                    'Subtract', 
                    'Multiply', 
                    'Divide', 
                    'Greater Than', 
                    'Less Than', 
                    'And',
                    'Or',
                    'Xor',
                    'Not',
                    'Riemann Distance'
                    ]
        }

        self.output_num = {
            0: 
                [],
            
            1: 
                [
                    'Lab Streaming Layer Input', 
                    'Epoched MAT File', 
                    'Continuous MAT File', 
                    'Epoched MAT File', 
                    'Tensor From Data', 
                    'Tensor From Handle', 
                    'Scalar From Value', 
                    'Scalar From Handle', 
                    'Array From Data', 
                    'Array From Handle',
                    'Absolute',
                    'Bessel', 
                    'Chebyshev Type-I', 
                    'Chebyshev Type-II', 
                    'Elliptic', 
                    'Butterworth',
                    'Feature Normalization', 
                    'Extract', 
                    'Covariance', 
                    'CSP', 
                    'CDF', 
                    'Butterworth', 
                    'Gaussian', 
                    'Virtual Tensor', 
                    'Non-Virtual Tensor', 
                    'Virtual Scalar', 
                    'Non-Virtual Scalar', 
                    'LDA', 
                    'SVM', 
                    'Logistic Regression', 
                    'Log',
                    'Max',
                    'Min',
                    'Reduced Sum',
                    'Riemann MDM Classifier',
                    'Riemann Mean',
                    'Riemann Potato',
                    'Running Average',
                    'Set Data',
                    'Stack',
                    'Standard Deviation',
                    'Tangent Space',
                    'Threshold',
                    'Transpose',
                    'Var',
                    'XDawn Covariance',
                    'Z Score',
                    'Equal', 
                    'Concatenate', 
                    'Add', 
                    'Subtract', 
                    'Multiply', 
                    'Divide', 
                    'Greater Than', 
                    'Less Than', 
                    'And',
                    'Or',
                    'Xor',
                    'Not',
                    'Riemann Distance'

                    ],
    

            2:
                []
        }

        self.nodes = []
        return super().__init__()

    def controlPointAt(self, pos):
        mask = QPainterPath()
        mask.setFillRule(Qt.WindingFill)
        for item in self.items(pos):
            if mask.contains(pos):
                # ignore objects hidden by others
                return
            if isinstance(item, ControlPoint):
                return item


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self.controlPointAt(event.scenePos())
            if item:
                self.startItem = item
                self.newConnection = Edge(item, event.scenePos())
                self.addItem(self.newConnection)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.newConnection:
            item = self.controlPointAt(event.scenePos())
            if (item and item != self.startItem):
                    p2 = item.scenePos()
            else:
                p2 = event.scenePos()
            self.newConnection.setP2(p2)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.newConnection:
            item = self.controlPointAt(event.scenePos())
            if item and item != self.startItem:
                self.newConnection.setEnd(item)
                if self.startItem.addLine(self.newConnection):
                    item.addLine(self.newConnection)
                else:
                    # delete the connection if it exists; remove the following
                    # line if this feature is not required
                    #self.startItem.removeLine(self.newConnection)
                    self.removeItem(self.newConnection)
                    pass
            else:
                self.removeItem(self.newConnection)
        self.startItem = self.newConnection = None
        super().mouseReleaseEvent(event)


class UIFunctions(mwindow):
    def __init__(self):
        super().__init__()

    def collect_params(self, enable):
        self.user_params = {}
        #f = open("sample2.txt", "a")
    
        #f.write(self.library_class_methods[self.Node_add.currentText()][self.Kernel_Selector.currentText()])
        
        for i in reversed(range(self.diag_layout.count())):
            item = self.diag_layout.itemAt(i)
            if isinstance(item, QHBoxLayout):
                for j in range(item.count()):
                    if isinstance(item.itemAt(j).widget(), QSpinBox):
                            #f.write(str(item.itemAt(j).widget().value()).rstrip('\n'))
                        #self.user_params.append(item.itemAt(j).widget().value())
                        self.user_params[item.itemAt(0).widget().text()] = item.itemAt(j).widget().value()
                            #f.write(','.rstrip('\n'))

                    elif isinstance(item.itemAt(j).widget(), QLineEdit):
                            #f.write(str(item.itemAt(j).widget().currentText()).rstrip('\n'))
                        #self.user_params.append(item.itemAt(j).widget().text())
                        self.user_params[item.itemAt(0).widget().text()] = item.itemAt(j).widget().text()
                            #f.write(','.rstrip('\n'))
        #print(self.user_params)
        self.add_Button.setEnabled(False)
        
    def UpdateKernelCombo(self):
        ind = self.model.index(self.Node_add.currentIndex(), 0, self.Node_add.rootModelIndex())
        self.Kernel_Selector.setRootModelIndex(ind)
        self.Kernel_Selector.setCurrentIndex(0)

    def zoom_out(self):
        self.view.transform()

    def reset_dialog(self):
        self.property_setter.reject()

    def confirm_dialog(self):
        self.add_Button.setEnabled(True)
        self.property_setter.done(1)

    def dialog_layout(self):
        dialog_buttons = QHBoxLayout()
        for i in reversed(range(self.diag_layout.count())):
            item = self.diag_layout.itemAt(i)
            if isinstance(item, QHBoxLayout):
                for j in range(item.count()):
                    if isinstance(item.itemAt(j).widget(), QPushButton):
                        if item.itemAt(j).widget().text() == 'Select File':
                            item.itemAt(j).widget().deleteLater()
                    else:
                        item.itemAt(j).widget().deleteLater()

        self.property_setter.setMinimumSize(300, 150)
        
        params = self.kernel_parameters[self.Node_add.currentText()][self.Kernel_Selector.currentText()]
        added_params_list = []
        for k, v in params.items():
            entry = QHBoxLayout()
            param_name = QLabel(k)
            if v == int:
                num_param = QSpinBox()
                num_param.setMaximum(100000)

            elif v == str:
                num_param = QLineEdit()

            elif v == float:
                num_param = QDoubleSpinBox()

            elif v == bool:
                num_param = QComboBox()
                num_param.addItems(('True', 'False'))

            #TODO: Figure out how to use a dropdown for preset text-based parameters instead of strings
            elif v == set:
                num_param = QComboBox()

            #TODO: Figure out how to store file paths on push button so dialog return can be saved
            elif v == bytearray:
                num_param = QPushButton('Select File')
                num_param.clicked.connect(lambda: UIFunctions.get_data_file_name(added_params_list))

            added_params_list.append(num_param)
            entry.addWidget(param_name)
            entry.addWidget(num_param)
            self.diag_layout.addLayout(entry)

        self.diag_layout.addLayout(dialog_buttons)

        dialog_buttons.addWidget(self.cancel_dialog)
        dialog_buttons.addWidget(self.ok_dialog)

        self.property_setter.setLayout(self.diag_layout)
        
        self.property_setter.open()


        # Removed when blocks were removed
    """#def set_trials_layout(self):
    #    for i in reversed(range(self.VertTrialsLayout.count())):
    #        item = self.VertTrialsLayout.itemAt(i)
    #        if isinstance(item, QHBoxLayout):
    #            for j in reversed(range(item.count())):
    #                item.itemAt(j).widget().deleteLater()
    #                #print("First for loop")
    #            item.setParent(None)
#
#
    #    self.num_classes = self.TrialNumUserInput.value()
    #    for i in range(self.num_classes):
    #        entryBox = QHBoxLayout()
    #        Label = QLabel(f"Class {i+1} Trials:")
    #        Entry = QSpinBox()
    #        entryBox.addWidget(Label)
    #        #print("3rd for loop")
    #        entryBox.addWidget(Entry)
    #        self.VertTrialsLayout.addLayout(entryBox)
    #    
    #    self.setTrialsWindow.show()
    #    self.VertTrialsLayout.addWidget(self.confirmTrialNumbers)
    #    
#
    #def collect_trial_params(self):
    #    self.trials_per_class = []
    #    for i in reversed(range(self.VertTrialsLayout.count())):
    #        item = self.VertTrialsLayout.itemAt(i)
    #        if isinstance(item, QHBoxLayout):
    #            for j in reversed(range(item.count())):
    #                if isinstance(item.itemAt(j).widget(), QSpinBox):
    #                    self.trials_per_class.append(item.itemAt(j).widget().value())
    #                    #print("4th for loop")
    #                item.itemAt(j).widget().deleteLater()
    #                 
    #    self.trials_per_class.reverse()
#
    #    self.setTrialsWindow.close()"""


    def compile_graph(self):
        session = 'session'

        scheduled_nodes = []
        name = QFileDialog.getSaveFileName(self, 'Save File')[0]
        print(name)

        
        f = open(f"{name}.py", "w")
        f.write(f"import BCIPY\nimport numpy as np\ndef main():\n\t{session} = Session.create()\n")
        
        
        #Steps to iterate through and compile nodes
            # Iterate through all nodes, if it is an edge, create an entry in an edges dictionary {self.edge_num: edge_in_library}
            # Need to check whether the producing node has multiple output edges
                # Since some nodes will have multiple control points, need to check whether the multiple control multiple have output edges
                
        
        for item in self.scene.items():
            edge_count = 0
            if isinstance(item, Node):
                if item.node_type in ['Tensor', 'Array', 'Scalar', 'Input Data']:
                    item.edge_num = edge_count
                    self.edges[edge_count] == f't[{edge_count}]'
                    edge_count += 1
                else:
                    sample_starting_node = item
                    break
                
        scheduled_nodes = self.recurse_nodes(sample_starting_node, scheduled_nodes)

        print('\n')

    def recurse_nodes(self, node, scheduled_nodes):
        # start at top node --> append to list of scheduled nodes
        # recurse down nodes
        if node.top_point.lines == []:
            scheduled_nodes.append(node)
            return scheduled_nodes
        
        elif node.top_point.lines[0].start in scheduled_nodes or node.top_point.lines[0].end in scheduled_nodes:
            scheduled_nodes.append(node)
            consumer = node.bot_point.lines[0].start

        else:
            producer = node.top_point.lines[0].start
            if producer == node:
                producer = node.top_point.lines[0].end
            scheduled_nodes = self.recurse_nodes(producer, scheduled_nodes)
            

        pass
        


class Node(QGraphicsPolygonItem):
    def __init__(self, scene, node_type, kernel_type, params):
        QGraphicsPolygonItem.__init__(self)
        
        self.node_type = node_type
        self.kernel_type = kernel_type
        self._scene = scene
        self.draw_status = False
        self.params = params
        self.edge_num = None
        self.top_points = []
        self.bot_points = []


        self.create_layout()
        self.show()

    def create_layout(self):
        
        #self.setDefaultTextColor(QColor(Qt.black))
        #self.rect = self.boundingRect()
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsFocusable)

        self.set_shape()
        self.set_controls()
    
    def set_controls(self):

        for sublist in self._scene.input_num.values():
            if self.kernel_type in sublist:
                in_index = list(self._scene.input_num.values()).index(sublist)
        
        for sublist in self._scene.output_num.values():
            if self.kernel_type in sublist:
                output_index = list(self._scene.output_num.values()).index(sublist)
        
        if in_index == 0:
            pass

        elif in_index == 1:

            top_point = ControlPoint(self)
            top_point.setX(self.boundingRect().center().x())
            top_point.setY(self.boundingRect().top())
            top_point.setBrush(Qt.red)
            self.top_points.append(top_point)
            
        elif in_index == 2:

            top_point = ControlPoint(self)
            top_point_2 = ControlPoint(self)
            self.top_points.append(top_point)
            self.top_points.append(top_point_2)
            top_point.setX(self.boundingRect().left())
            top_point.setY(self.boundingRect().top())
            top_point_2.setX(self.boundingRect().right())
            top_point_2.setY(self.boundingRect().top())  
            top_point.setBrush(Qt.red)
            top_point_2.setBrush(Qt.red)

        
        

        if output_index == 0:
            pass

        elif output_index == 1:

            bot_point = ControlPoint(self)
            bot_point.setX(self.boundingRect().center().x())
            bot_point.setY(self.boundingRect().bottom())
            bot_point.setBrush(Qt.green)
            self.bot_points.append(bot_point)
            
        elif output_index == 2:

            bot_point = ControlPoint(self)
            bot_point_2 = ControlPoint(self)
            self.bot_points.append(bot_point)
            self.bot_points.append(bot_point_2)
            bot_point.setX(self.boundingRect().left())
            bot_point.setY(self.boundingRect().bottom())
            bot_point_2.setX(self.boundingRect().right())
            bot_point_2.setY(self.boundingRect().bottom())  
            bot_point.setBrush(Qt.green)
            bot_point_2.setBrush(Qt.green)

    
    def mouseDoubleClickEvent(self, event=QGraphicsSceneMouseEvent):
        super().mouseDoubleClickEvent(event)
        if event.button() == Qt.LeftButton:
            self.param_editor()
    
    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)
        if event.key() in {Qt.Key_Delete, Qt.Key_Backspace}:
            self.deleteAllEdges()
            self._scene.removeItem(self)
        
    def deleteAllEdges(self):
        for top_point in self.top_points:
            for existing in top_point.lines:  
                top_point.removeLine(existing)
                if existing in existing.start.lines:
                    existing.start.lines.remove(existing)
            
                #since a line can be drawn in two directions
                if existing in existing.end.lines:
                    existing.end.lines.remove(existing)
            
        for bot_point in self.bot_points:
            for existing in bot_point.lines:
                bot_point.removeLine(existing)
                if existing in existing.start.lines:
                    existing.start.lines.remove(existing)
            
                #since a line can be drawn in two directions
                if existing in existing.end.lines:
                    existing.end.lines.remove(existing)
            

    def param_editor(self):
        self.param_editor_window = QDialog()
        self.param_editor_window.setMinimumSize(300, 150)
        self.param_editor_vert_layout = QVBoxLayout()
        self.param_editor_window.setLayout(self.param_editor_vert_layout)
        for k, v in self.params.items():
            entry = QHBoxLayout()
            label = QLabel(k)
            
            entry.addWidget(label)

            if type(v) == str:
                edit = QLineEdit()
                edit.setText(v)
            elif type(v) == int:
                edit = QSpinBox()
                edit.setValue(v)
            elif type(v) == float:
                edit = QDoubleSpinBox()
                edit.setValue(v)
            elif type(v) == bytearray:
                edit = QPushButton('Select File')
                edit.clicked.connect(edit.setText(QFileDialog.getOpenFileName()))
                
            
            entry.addWidget(edit)

            self.param_editor_vert_layout.addLayout(entry)
        
        button_box = QDialogButtonBox()
        self.param_editor_vert_layout.addWidget(button_box)
        button_box.addButton(QDialogButtonBox.Cancel)
        button_box.addButton(QDialogButtonBox.Save)
        button_box.button(QDialogButtonBox.Save).setText("Apply Changes")
        button_box.button(QDialogButtonBox.Save).setMinimumWidth(150)
        button_box.button(QDialogButtonBox.Cancel).setMinimumWidth(150)

        button_box.button(QDialogButtonBox.Cancel).clicked.connect(lambda: self.param_editor_window.reject())
        button_box.button(QDialogButtonBox.Save).clicked.connect(lambda: self.collect_modified_params())
        self.param_editor_window.show()

    def collect_modified_params(self):
        for i in reversed(range(self.param_editor_vert_layout.count())):
            item = self.param_editor_vert_layout.itemAt(i)
            if isinstance(item, QHBoxLayout):
                for j in range(item.count()):
                    if isinstance(item.itemAt(j).widget(), QSpinBox):
                            #f.write(str(item.itemAt(j).widget().value()).rstrip('\n'))
                        #self.user_params.append(item.itemAt(j).widget().value())
                        self.params[item.itemAt(0).widget().text()] = item.itemAt(j).widget().value()
                            #f.write(','.rstrip('\n'))

                    elif isinstance(item.itemAt(j).widget(), QLineEdit):
                            #f.write(str(item.itemAt(j).widget().currentText()).rstrip('\n'))
                        #self.user_params.append(item.itemAt(j).widget().text())
                        self.params[item.itemAt(0).widget().text()] = item.itemAt(j).widget().text()
                            #f.write(','.rstrip('\n'))
        print(self.params)
        self.param_editor_window.accept()

    def set_shape(self):
        node_styles = {
            'Filter': 
                {
                    'Butterworth': 
                        {
                            'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                            'Colour': 'light blue'
                            },
                    'Chebyshev Type-I':
                        {
                            'Shape': QPolygon([QPoint(80,40), QPoint(80,0), QPoint(40,0), QPoint(40,40)]),
                            'Colour': 'light green'
                        },
                    'Chebyshev Type-I':
                        {
                            'Shape': QPolygon([QPoint(80,40), QPoint(80,0), QPoint(40,0), QPoint(40,40)]),
                            'Colour': 'light green'
                        },
                    'Elliptic':
                        {
                            'Shape': QPolygon([QPoint(80,40), QPoint(80,0), QPoint(40,0), QPoint(40,40)]),
                            'Colour': 'green'
                        },
                    'Bessel':
                        {
                            'Shape': QPolygon([QPoint(80,40), QPoint(80,0), QPoint(40,0), QPoint(40,40)]),
                            'Colour': 'light green'
                        },

                },
            
            'Classifier':
                {
                    'LDA':
                    {
                        'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                        'Colour': 'orange'
                    },
                    'SVM':
                    {
                        'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                        'Colour': 'orange'
                    },

                    'Logistic Regression':
                    {
                        'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                        'Colour': 'orange'
                    },

                    'Custom':
                    {
                        'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                        'Colour': 'orange'
                    },
                    
                },

            'Simple Operations':
                {
                    'Add':
                    {
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Subtract':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Divide':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Multiply':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Absolute':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Mean':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Log':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Min':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Max':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Standard Deviation':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Var':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                    'Threshold':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'yellow'
                    },
                },

            'Riemann Operations':
                {
                    'Riemann Distance':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'purple'
                    },
                    'Riemann Mean':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'purple'
                    },
                    'Riemann MDM Classifier':{
                        'Shape': QPolygon([QPoint(0,40), QPoint(0,0), QPoint(40,0), QPoint(40,40)]),
                        'Colour': 'purple'
                    },
                },

                'Other Operations':
                {
                    'CDF':
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }, 
                    'CSP':
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }, 
                    'Feature_normalization':
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }, 
                    'Covariance':
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }, 
                    'Reduced Sum': 
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }, 
                    'Tangent Space':
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }, 
                    'Running Average':
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }, 
                    'XDawn Covariances':
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }, 
                    'Z Score':
                        {
                            'Shape': QPolygon([QPoint(20,0), QPoint(0,20), QPoint(20,40), QPoint(40,20)]),
                            'Colour': 'light yellow'
                        }
                },
            
            'Logical' : 
                {
                    'And':{'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'},
                    'Or':{'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'}, 
                    'Xor':{'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'},
                    'Not':{'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'}, 
                    'Equal':{'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'}, 
                    'Greater Than':{'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'}, 
                    'Less Than':{'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'},
                },

            'Manipulate Data': 
                {
                    'Extract':
                        {
                           'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'cyan'
                        },
                    'Concatenate':{'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'cyan'},
                    'Set Data':{'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'cyan'}, 
                    'Stack':{'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'cyan'}, 
                    'Transpose':{r'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'cyan'},
                },
            'Tensor':
                {
                    'Tensor From Data':
                    {
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'yellow'
                    },
                    'Tensor From Handle':{
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'yellow'
                    },
                    'Virtual Tensor':{
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'yellow'
                    },
                    'Generic Tensor':{
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'yellow'
                    }
                },

            'Scalar':
                {
                    'Scalar From Value':
                    {
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'purple'
                    },
                    'Scalar From Handle':{
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'magenta'
                    },
                    'Virtual Scalar':{
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'cyan'
                    },
                    'Generic Scalar':{
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'gray'
                    }
                },

            'Input Data':
                {
                    'Lab Streaming Layer Input':
                    {
                        'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'pink'
                    },

                    'Epoched MAT File':
                    {
                        'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'
                    },

                    'Continuous MAT File':
                    {
                        'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'
                    },

                    'Class Separated MAT File':
                    {
                        'Shape': QPolygon([QPoint(20,40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'light red'
                    }
                },

            'Array':
                {
                    'Generic Array':
                    {
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'yellow'
                    },
                    'Circle Buffer':{
                        'Shape': QPolygon([QPoint(20,-40), QPoint(0,0), QPoint(40,0)]),
                        'Colour': 'yellow'
                    }
                },
            }
        
        self.setPolygon(node_styles[self.node_type][self.kernel_type]['Shape'])
        self.setBrush(QColor(node_styles[self.node_type][self.kernel_type]['Colour']))

        self.textItem = QGraphicsSimpleTextItem(self.kernel_type, self)
        rect = self.textItem.boundingRect()
        rect.moveCenter(self.boundingRect().center())
        self.textItem.setPos(rect.topLeft())
        self.setPen(QPen(Qt.black, 1))

    @classmethod
    def create_node(cls, scene, node_type, kernel_type, params):
        node = cls(scene, node_type, kernel_type, params)
        scene.addItem(node)
        

class Arrow(QGraphicsPolygonItem):
    def __init__(self, edge):
        
        #Arrow initial conditions
        self.edge = edge
        self.angle = atan2(-(self.edge._line.dy()), self.edge._line.dx())
        self.arrowSize = 10
        self.arrowP1 = self.edge._line.p1() - QPointF(sin(self.angle + pi / 3) * self.arrowSize, cos(self.angle + pi / 3) * self.arrowSize)
        self.arrowP2 = self.edge._line.p1() - QPointF(sin(self.angle + pi - (pi / 3)) * self.arrowSize, cos(self.angle + pi - (pi / 3)) * self.arrowSize)

        points = QPolygonF([self.arrowP1, self.edge._line.p1(), self.arrowP2])
        
        QGraphicsPolygonItem.__init__(self, points, self.edge)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setPen(QPen(Qt.black))
        self.setBrush(QColor(Qt.black))

class Edge(QGraphicsLineItem):
    
    def __init__(self, start, p2):
        super().__init__()
        self.start = start
        self.end = None
        self._line = QLineF(start.scenePos(), p2)
        
        self.setLine(self._line)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsFocusable)
        self.arrowHead = Arrow(self)
        self.arrowSize = 10

        #self.arrow = ArrowHead(self)
        #self.arrow.setX(self._line.p1().x())
        #self.arrow.setY(self._line.p1().y())


        self.angle = atan2(-(self._line.dy()), self._line.dx())

    def controlPoints(self):
        return self.start, self.end

    def setP2(self, p2):
        self._line.setP2(p2)
        self.setLine(self._line)
        #self.setArrow(self._line.p2())
        self.updateArrowHead(self._line.p2())
        
    #def setArrow(self, p2):
    #    self.arrow.setX(self._line.p2().x())
    #    self.arrow.setY(self._line.p2().y())

    def updateArrowHead(self, p2):
        self.angle = atan2(-(self._line.dy()), self._line.dx())
        self.arrowHead.arrowP1 = self._line.p2() - QPointF(sin(self.angle + pi / 3) * self.arrowSize, cos(self.angle + pi / 3) * self.arrowSize)
        self.arrowHead.arrowP2 = self._line.p2() - QPointF(sin(self.angle + pi - (pi / 3)) * self.arrowSize, cos(self.angle + pi - (pi / 3)) * self.arrowSize)
        points = QPolygonF([self.arrowHead.arrowP1, self._line.p2(), self.arrowHead.arrowP2])
        self.arrowHead.setPolygon(points)

    def setStart(self, start):
        self.start = start
        self.updateLine()

    def setEnd(self, end):
        self.end = end
        self.updateLine(end)

    def updateLine(self, source):
        if source == self.start:
            self._line.setP1(source.scenePos())
        else:
            self._line.setP2(source.scenePos())
            #self.setArrow(source.scenePos())
            self.updateArrowHead(source.scenePos())
        self.setLine(self._line)

    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)
        if event.key() in {Qt.Key_Delete, Qt.Key_Backspace}:
            try:
                self.start.removeLine(self, False)
                self.end.removeLine(self)
                del self
            except RuntimeError or AttributeError:
                self.scene().removeItem(self)
                del self

            

    @classmethod    
    def create_edge(cls, starting_node):
        edge = cls(starting_node)

class ControlPoint(QGraphicsEllipseItem):
    def __init__(self, parent):
        super().__init__(-5, -5, 10, 10, parent)
        self.parent = parent
        self.lines = []
        self.setBrush(QBrush(QColor(Qt.red)))
        # this flag **must** be set after creating self.lines!
        self.setFlags(self.ItemSendsScenePositionChanges)

    def addLine(self, lineItem):
        for existing in self.lines:
            if existing.controlPoints() == lineItem.controlPoints():
                # another line with the same control points already exists
                return False
        self.lines.append(lineItem)
        return True

    def removeLine(self, lineItem, flag = True):
        for existing in self.lines:
            if existing.controlPoints() == lineItem.controlPoints():
                if flag:
                    self.scene().removeItem(existing)
                self.lines.remove(existing)
                return True
        return False

    def itemChange(self, change, value):
        for line in self.lines:
            line.updateLine(self)
        return super().itemChange(change, value)


app = QApplication(sys.argv)
Window = mwindow()
sys.exit(app.exec_())