# Credit-Card-Fraud-Detection

NAME: VASUDEVA REDDY

NO.OF WEEKS: 4

PROJECT NAME :Email-Spam-Classifier

PROJECT SCOPE:

Credit Card Fraud Detection is a proactive security mechanism used by financial institutions to identify, flag, and block unauthorized transactions. By analyzing user patterns in real time, these systems prevent criminals from spending money that is not theirs, protecting both merchants and consumers from severe financial losses.


source code:


# 1. Load your dataset (assuming a standard Kaggle CSV format)
df = pd.read_csv('creditcard.csv')

# 2. Separate features (X) and target label (y)
# 'Class' is 0 for legitimate transactions and 1 for fraudulent ones
X = df.drop(columns='Class', axis=1)
y = df['Class']

# 3. Split data into training (80%) and testing (20%) sets
# stratify=y is critical here to preserve the tiny percentage of fraud in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate the model on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
print(f"Testing Accuracy: {accuracy:.4f}")



