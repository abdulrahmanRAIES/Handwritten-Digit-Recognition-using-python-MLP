from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from ExtractFeaturs import extract


def mlp(X_train, X_test, y_train, y_test):
    
    clf =MLPClassifier(activation='relu', hidden_layer_sizes=(200, 200), alpha = 0.3,max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Training Score :: {}\n".format(clf.score(X_train, y_train)))
    print("Testing Score :: {}\n".format(clf.score(X_test, y_test)))
   
   # filename = 'finalized_m2odel.sav'
   # pickle.dump(clf, open(filename, 'wb'))  # enable to save the model

def main():
    
    print("Starting to Extract Featurs from images using HOG ")
    print("----------------------------------------------")
    
    DATA,Labels = extract().extracting()
    X_train, X_test, y_train, y_test = train_test_split(DATA, Labels, test_size=0.10)
    mlp( X_train, X_test, y_train, y_test)
    
if __name__=="__main__":
    main()