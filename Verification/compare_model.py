'''
A model to verify the model made for predict diabete.
Zhihao Zhang
2021/3/7
'''

import numpy as np
import matplotlib.pyplot as plt


def Check_correctness(result,data,string=True,plot=True,figsize=[8,6]):
    '''
    Return the correctness of the classification, the sensitivity and the specificity.
    
    The four types of classifications are given in the order of ['true positive','ture negative','false positive','false negative']
    The outcome 1 was set to be the true outcome
   
    Parameters
    ----------
    result : array_like
    The numpy array or list which conatin the prdiction by the model from the verification data.

    data : array_like
    The numpy array or list which conatin the outcomes of the verification data.

    string : bool, optional
    When is true, the sensitivity and specificity will be printed. Defaut is True.

    plot : bool, optional
    When is true, all figures will be shown. Defaut is True.

    figsize : list, optional 
    The size of the figures, defult is [8,6].
   
    Return
    ------
    statistics : list
    In the order of ['true positive','ture negative','false positive','false negative']

    sensitivity : float
    The sensitivity of the prediction.

    specificity : float
    The specificity of the prediction.
    '''
    # check the data size
    if len(data) is not len(result):
        raise Exception('Sizes of the input and the data are not compatible')

    # initialization
    _true_positive = 0
    _true_negative = 0
    _false_positive = 0
    _false_negative = 0

    # counting
    for i in range(len(data)):
        if data[i] == 1:
            if result[i] == 1:
                _true_positive += 1
            elif result[i] == 0:
                _false_positive += 1
            else:
                raise Exception('Except input as 0 or 1 but given',result[i])
                break
        else:
            if result[i] == 0:
                _true_negative += 1
            elif result[i] == 1:
                _false_negative += 1
            else:
                raise Exception('Except input as 0 or 1 but given',result[i])
                break
    
    # plot the result in histogram
    if plot:
        _height = [_true_positive,_true_negative,_false_positive,_false_negative]
        _ind = np.linspace(0.5,3.5,4)
        _width = 0.6
        _labels = ['true positive','ture negative','false positive','false negative']

        fig = plt.figure(figsize=figsize,tight_layout=True)
        ax = plt.axes()
        ax.bar(_ind,_height,_width)
        ax.set_xticks(_ind)
        ax.set_xticklabels(_labels)
        plt.ylabel('counts')
        plt.show()

    # calculate the sensitivity and specificity
    _sensitivity = _true_positive/(_false_negative+_true_positive)
    _specificity = _true_negative/(_true_negative+_false_positive)
    if string:
        print('The sensitivity of the model is %.3f'%(_sensitivity))
        print('The specificity of the model is %.3f'%(_specificity))

    return [_true_positive,_true_negative,_false_positive,_false_negative],_sensitivity,_specificity




