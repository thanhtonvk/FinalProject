import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score
class NameClassifier:
    def __init__(self,model_path):
        self.model = pickle.load(open(model_path, 'rb'))
    def preprocessing(self,real_x):
        data_train = pd.read_csv("data_train_moredata4.csv")
        x_train = data_train["name"]
        # ngram-char level - we choose max number of words equal to 30000 except all words (100k+ words)
        tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
        tfidf_vect_ngram_char.fit(x_train)
        # assume that we don't have test set before
        real_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(real_x)
        return real_tfidf_ngram_char
    def predict_to_csv(self,real_x,real_y,file_path_export):
        real_tfidf_ngram_char = self.preprocessing(real_x)
        real_y = np.array(real_y.values, dtype = "int")
        real_predictions = self.model.predict(real_tfidf_ngram_char)
        real_predictions_proba = self.model.predict_proba(real_tfidf_ngram_char)
        results = pd.DataFrame([real_x, real_y, real_predictions, real_predictions_proba[:,0]]).transpose()
        results.columns = ["name", "label", "Prediction result", "prob non_person"]
        results.to_csv(file_path_export)
        f = open(file_path_export,'a',encoding="utf-8")
        f.write("Accuracy on real data: "+ str(accuracy_score(real_predictions, real_y))+'\n')
        tn, fp, fn, tp = confusion_matrix(real_y, real_predictions).ravel()
        tnr = tn/(tn + fp)
        f.write(("ACC non person name: %f" % tnr)+'\n')
        tpr = tp / (tp + fn)
        f.write(("Acc person name: %f" % tpr)+'\n')
        f.close()
        return file_path_export
    def predict_to_csv_non_statistic(self,real_x,file_path_export):
        real_tfidf_ngram_char = self.preprocessing(real_x)
        real_predictions = self.model.predict(real_tfidf_ngram_char)
        real_predictions_proba = self.model.predict_proba(real_tfidf_ngram_char)
        results = pd.DataFrame([real_x, real_predictions, real_predictions_proba[:,0]]).transpose()
        results.columns = ["name", "Prediction result", "prob non_person"]
        results.to_csv(file_path_export)
        return file_path_export


