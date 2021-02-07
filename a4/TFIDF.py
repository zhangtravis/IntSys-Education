import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

def TFIDF_approach(path_to_train):
    d = pd.read_csv("data/pt_data.csv")
   
    d_p = d['tokstem']

    tf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1,5),
        max_features=15000
    )
   

    tf.fit(d_p)

    return tf

if __name__ == '__main__':
    tf = TFIDF_approach('data/pt_data.csv')
    d = pd.read_csv("data/pt_data.csv")
    tr = pd.read_csv("data/pval_data.csv")
    pt = pd.read_csv("data/ptest_data.csv")
    d_p = d['tokstem']
    pt_p = pt['tokstem']
    tr_p = tr['tokstem']

    x_train = tf.transform(d_p)
    y_train = d['class']
    x_val = tf.transform(tr_p)
    x_test = tf.transform(pt_p)
    y_test = pt['class']

    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB()
    classifier.fit(x_train,d['class'])
    print('NB accuracy:',classifier.score(x_val,tr['class']))

    ###logistic regression
    from sklearn.linear_model import LogisticRegression
    
    ##first approach
    lg1 = LogisticRegression()
    lg1.fit(x_train,d['class'])
    print('logistic accuracy:',lg1.score(x_val,tr['class']))

    ##grid search
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C':range(1,10),
                'dual':[True,False]
                }
    lgGS = LogisticRegression(,max_iter = 10000)
    grid = GridSearchCV(lgGS, param_grid=param_grid,cv=3,n_jobs=-1)
    grid.fit(x_train,y_train)

    lg_final = grid.best_estimator_

    print("final-logistic accuracy:", lg_final.score(x_test,y_test))
