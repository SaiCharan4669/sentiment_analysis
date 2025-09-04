import pandas as pd 
import numpy as np
import re
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.pipelines = {}
        self.best_pipeline = None

    def preprocess_text(self, text):
        """Perform text preprocessing for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, email addresses, and special characters
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'[^a-zA-Z\s!?.:;]', '', text)
        
        # Handle contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        
        # Tokenize and remove stopwords
        tokens = nltk.word_tokenize(text)
        sentiment_stopwords = self.stopwords - {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'nor'}
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens
            if token not in sentiment_stopwords and len(token) > 1
        ]
        
        return " ".join(cleaned_tokens)

    def exploratory_data_analysis(self, df):
        """Perform EDA on the dataset"""
        print(f"Dataset shape: {df.shape}")
        print("\n=== Missing Values ===")
        print(df.isnull().sum())
        
        print("\n=== Sentiment Distribution ===")
        print(df['sentiment'].value_counts())
        
        print("\n=== Class Percentages ===")
        print(df['sentiment'].value_counts(normalize=True) * 100)
        
        # Visualization
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        df['sentiment'].value_counts().plot(kind='bar')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()
        
        # Text analysis
        df['text_length'] = df['review'].str.len()
        df['word_count'] = df['review'].str.split().str.len()
        
        print("\n=== Text Length Statistics ===")
        print(df.groupby('sentiment')[['text_length', 'word_count']].describe())
        
        # Visualizations
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        for sentiment in df['sentiment'].unique():
            data = df[df['sentiment'] == sentiment]['text_length']
            plt.hist(data, bins=30, alpha=0.6, label=sentiment, edgecolor='black')
        plt.title('Text Length Distribution')
        plt.ylabel('Frequency')
        plt.xlabel('Text Length')
        plt.legend(title='Sentiment')
        
        plt.subplot(1, 2, 2)
        for sentiment in df['sentiment'].unique():
            data = df[df['sentiment'] == sentiment]['word_count']
            plt.hist(data, bins=30, alpha=0.6, label=sentiment, edgecolor='lightgrey')
        plt.title('Word Count Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.show()

    def create_word_clouds(self, df):
        """Create word clouds for each sentiment"""
        plt.figure(figsize=(16, 7))
        for i, sentiment in enumerate(df['sentiment'].unique()):
            text = ' '.join(df[df['sentiment'] == sentiment]['review'])
            
            plt.subplot(1, 2, i+1)
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'{sentiment.capitalize()} Sentiment')
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

    def build_pipelines(self):
        """Build multiple classification pipelines"""
        vectorizers = {
            'tfidf_unigram': TfidfVectorizer(max_features=5000, ngram_range=(1, 1), stop_words='english'),
            'tfidf_bigram': TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
            'tfidf_trigram': TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
        }
        
        classifiers = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'svm': SVC(probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        for vec_name, vectorizer in vectorizers.items():
            for clf_name, classifier in classifiers.items():
                pipeline_name = f"{vec_name}_{clf_name}"
                self.pipelines[pipeline_name] = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)
                ])

    def train_and_evaluate_pipelines(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all pipelines"""
        results = {}
        
        for name, pipeline in self.pipelines.items():
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'pipeline': pipeline,
                'predictions': y_pred
            }
            print(f"Accuracy: {accuracy:.4f}")
        
        return results

    def find_best_pipeline(self, results):
        """Identify the best performing pipeline"""
        best_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_pipeline = results[best_name]['pipeline']
        print(f"\nBest pipeline: {best_name}")
        print(f"Best accuracy: {results[best_name]['accuracy']:.4f}")
        return best_name, results[best_name]

    def detailed_evaluation(self, X_test, y_test, pipeline_name, best_result):
        """Perform detailed evaluation of the best model"""
        print(f"\n=== Detailed Evaluation for {pipeline_name} ===")
        y_pred = best_result['predictions']
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['negative', 'positive'], 
                   yticklabels=['negative', 'positive'])
        plt.title(f'Confusion Matrix - {pipeline_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def hyperparameter_tuning(self, X_train, y_train):
        """Optimize hyperparameters for the best pipeline"""
        print("\n=== Hyperparameter Tuning ===")
        param_grid = {}
        
        # Define parameter grid based on classifier type
        classifier_name = type(self.best_pipeline.named_steps['classifier']).__name__.lower()
        
        if 'logisticregression' in classifier_name:
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2']
            }
        elif 'multinomialnb' in classifier_name:
            param_grid = {
                'classifier__alpha': [0.1, 0.5, 1.0]
            }
        elif 'svc' in classifier_name:
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        elif 'randomforest' in classifier_name:
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20]
            }
        
        if not param_grid:
            print("No parameter grid defined for this classifier")
            return None
        
        grid_search = GridSearchCV(
            self.best_pipeline,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        
        # Update best pipeline
        self.best_pipeline = grid_search.best_estimator_
        return grid_search

    def predict_sentiment(self, texts):
        """Predict sentiment for new texts"""
        if self.best_pipeline is None:
            raise ValueError("No trained pipeline available")
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        predictions = self.best_pipeline.predict(processed_texts)
        return predictions

# Main execution
def main():
    # Load dataset
    df = pd.read_csv("IMDB Dataset.csv")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Perform EDA
    analyzer.exploratory_data_analysis(df)
    
    # Create word clouds
    analyzer.create_word_clouds(df)
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['processed_text'] = df['review'].apply(analyzer.preprocess_text)
    
    # Split data
    X = df['processed_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build pipelines
    analyzer.build_pipelines()
    
    # Train and evaluate
    results = analyzer.train_and_evaluate_pipelines(X_train, X_test, y_train, y_test)
    
    # Find best pipeline
    best_name, best_result = analyzer.find_best_pipeline(results)
    
    # Detailed evaluation
    analyzer.detailed_evaluation(X_test, y_test, best_name, best_result)
    
    # Hyperparameter tuning
    analyzer.hyperparameter_tuning(X_train, y_train)
    
    # Final evaluation
    y_pred = analyzer.best_pipeline.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Model Accuracy: {final_accuracy:.4f}")
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()