import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
class TextPreprocess:

    def __init__(self):
        self.vocabulary = []
        self.vocab_index = {}

    def tokenize(self, text:str):
        text = text.lower()
        return word_tokenize(text)
    
    def build_vocabulary(self, texts:list):
        self.vocabulary = sorted(set(word for text in texts for word in self.tokenize(text)))
        self.vocab_index = {word: i for i, word in enumerate(self.vocabulary)}

    def binarize_bow(self, texts):
        X = np.zeros((len(texts), len(self.vocabulary)))
        for i, text in enumerate(texts):
            for word in text:
                if word in self.vocab_index:
                    X[i][self.vocab_index[word]] = 1

        return X

class BernoulliNB:
    def __init__(self):
        self.phi = None
        self.phi_j1 = None
        self.phi_j0 = None
        self.posterior1 = None
        self.posterior0 = None
        
    def fit(self, X, y):
        y0 = np.sum(y == 0)
        y1 = np.sum(y == 1)

        self.phi = y1 / X.shape[0]

        self.phi_j0 = (np.sum(X[y == 0], axis = 0) + 1) / (y0 + 2)
        self.phi_j1 = (np.sum(X[y == 1], axis = 0) + 1) / (y1 + 2)

    def predict(self, X):
        epsilon = 1e-8
        phi_j0 = np.clip(self.phi_j0, epsilon, 1 - epsilon)
        phi_j1 = np.clip(self.phi_j1, epsilon, 1 - epsilon)
        posterior1 = np.sum(X * np.log(phi_j1) + (1 - X) * np.log(1 - phi_j1), axis = 1) + np.log(self.phi)
        posterior0 = np.sum(X * np.log(phi_j0) + (1 - X) * np.log(1 - phi_j0), axis = 1) + np.log(1 - self.phi)
        return (posterior1 > posterior0).astype(int)
    
#Main
if __name__ == '__main__':
    # from sklearn.datasets import make_classification
    # X, y = make_classification(n_samples=1000, n_classes=2, n_informative=2, n_features=4)

    # # X = (X > 0).astype(int)

    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # model = BernoulliNB()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    from sklearn.metrics import classification_report, accuracy_score
    # cr = classification_report(y_test, y_pred)
    # print("classification report: \n", cr)
    # ac = accuracy_score(y_test, y_pred)
    # print("accuracy:", ac)

    # print("\npredicted classes : ", np.unique(y_pred, return_counts=True))
    # print("training set: ", np.unique(y_train, return_counts=True))
    # print("test set", np.unique(y_test, return_counts=True))

    import pandas as pd
    data = pd.read_csv(r'C:\Users\dcode\VIBE\Machine Learning\NLP\Texts\SMSSpamCollection.txt', sep='\t', names=['labels', 'message'])
    data['labels'] = data['labels'].map({'spam': 1, 'ham': 0})
    preprocessor = TextPreprocess()
    preprocessor.build_vocabulary(data['message'])
    X = preprocessor.binarize_bow(data['message'])
    y = data['labels'].values.astype(int)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = BernoulliNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cr = classification_report(y_test, y_pred)
    print("classification report: \n", cr)
    ac = accuracy_score(y_test, y_pred)
    print("accuracy:", ac)

    test = '''Subject: Congratulations! You've Won $1,000,000!

Dear User,

We are pleased to inform you that your email address has been selected as the lucky winner of our International Online Lottery Draw.

You have won a total sum of **ONE MILLION UNITED STATES DOLLARS ($1,000,000.00)**.

To claim your prize, please reply to this email with the following information:
- Full Name
- Address
- Phone Number
- Copy of a valid ID

Please note: This offer is confidential and must not be disclosed to any third party until your claim has been processed. Failure to respond within 48 hours will result in forfeiture of your prize.

Kind regards,  
Mr. James Adams  
Claims Manager  
**Global Rewards International**  
Email: claimnow@lotterywinners.co

---

**Act fast! This is a once-in-a-lifetime opportunity!**
'''

    text = preprocessor.binarize_bow([test])
    print(model.predict(text))






        



