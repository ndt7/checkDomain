"""Train and test LSTM classifier"""
import dga_classifier.data as data
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.model_selection import train_test_split

# xay dung model 
def build_model(max_features, maxlen):
    """Build LSTM model"""
    model = Sequential()  # khoi tao model 
    model.add(Embedding(max_features, 128, input_length=maxlen))  # embedding 
    model.add(LSTM(128))                
    model.add(Dropout(0.5))                 
    model.add(Dense(1))                 
    model.add(Activation('sigmoid'))    

    model.compile(loss='binary_crossentropy',optimizer='rmsprop')    # ham mat mat va toi uu 
    return model

# ham run batch_size = 128 so du lieu trong 1 lan cap nhat tham so 
# interation so lan batch_size ma model phai duyet trong 1 epock
# 1 epoch la 1 lan duyet qua het cac du lieu trong tap huan luyen   
def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    indata = data.get_data()    # ham lay du lieu 

    # Extract data and labels
    X = [x[1] for x in indata]   # lay du lieu train 
    labels = [x[0] for x in indata]         # lay nhan 

    # Generate a dictionary of valid characters  /// xac thuc 
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

    max_features = len(valid_chars) + 1   # max dac trung 
    maxlen = np.max([len(x) for x in X])

    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # Convert labels to 0-1    # chuan hoa du lieu 
    y = [0 if x == 'benign' else 1 for x in labels]

    final_data = []

    #
    for fold in range(nfolds):
        print("fold %u/%u") % (fold+1, nfolds)

        # chia tap du lieu 
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels,test_size=0.2)
        
        print('Build model...')
        model = build_model(max_features, maxlen)

        print("Train...")

        # phan chia du lieu 
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        # train voi max_epcok 
        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)

            t_probs = model.predict_proba(X_holdout)
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print('Epoch %d: auc = %f (best=%f)') % (ep, t_auc, best_auc)

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict_proba(X_test)

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 2:
                    break

        final_data.append(out_data)

    return final_data

