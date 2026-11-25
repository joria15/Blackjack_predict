# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "blkjckhands.csv"

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "mojocolors/900000-hands-of-blackjack-results",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

from sklearn.tree import DecisionTreeClassifier #import sklearn
from sklearn.metrics import RocCurveDisplay #visualize data
import joblib #import storage base

#Player prediction:
def player_train_test_data():
  X_train, y_train, X_test, y_test = [], [], [], [] #storing training and testing data
  player_clf = DecisionTreeClassifier(min_samples_leaf=100, random_state=42) #create a decision tree classifier specifically for the player data so it can learn patterns from the cards drawed by the player
  #and make sure that each leaf has atleast 100 examples to go off of

  player_cards = df[['card1', 'card2', 'card3', 'card4', 'card5']] #the value of each card for each row
  for row in player_cards.itertuples(index=True, name=None): #get each row of 5 cards received by the player, also remove index and name when iterating so it is only card numbers

    hand = [card for card in row[1:] if card != 0] #get all actual card numbers, exclude empty slots which are represented by zero and start at index 1 to skip the index value

    if row[0] < 720000: #80% training data
      for i in range(1, len(hand) - 1): #get all rounds for each hand to train the model how to predict the next card as it reveals the next one each round (start with two cards)
        X_train.append(hand[:(i+1)] + ([0] * (4 - i))) #seen: pad the rest of the list with 0 so the data is all the same length, so first element add 4 zeros, and then for 2 add 3 zeros etc...
        y_train.append(hand[i+1]) #unseen (integer)
    else: #20% testing data
      for i in range(1, len(hand) - 1): #get all rounds for each hand to train the model how to predict the next card as it reveals the next one each round (start with two cards)
        X_test.append(hand[:(i+1)] + ([0] * (4 - i))) #seen: pad the rest of the list with 0 so the data is all the same length, so first element add 4 zeros, and then for 2 add 3 zeros etc...
        y_test.append(hand[i+1]) #unseen

  player_clf.fit(X_train, y_train) #fit the data that we collected from the player onto the decision tree so that it can learn from the data and create patterns

  return player_clf, X_test, y_test #return the decision tree so that it can be used for testing

#player testing
def player_test_result(player_clf, X_test, y_test):
  player_clf.predict(X_test)
  print(player_clf.score(X_test, y_test))


#save the model and data (comment out after the first use)
# player_clf, X_test, y_test = player_train_test_data()
# joblib.dump(player_clf, "blackjack_tree.joblib") #save model
# joblib.dump((X_test, y_test), "blackjack_test_data.joblib") #save test_data
# print("save was successful!")

#test the model and data (uncomment out after the first use)
player_clf = joblib.load("blackjack_tree.joblib") #load model
X_test, y_test = joblib.load("blackjack_test_data.joblib") #load test data
player_test_result(player_clf, X_test, y_test)

print(player_clf.predict([[9, 5, 0, 0, 0]]))


