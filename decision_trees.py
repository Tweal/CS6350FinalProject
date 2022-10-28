import pandas as pd
import sklearn.ensemble as ensemble
from sklearn import tree
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pathlib import Path

train_raw = pd.read_csv('Data/train_final.csv')
final_raw = pd.read_csv('Data/test_final.csv')

output_location = 'Data/Attempts/Tree/'
Path(output_location).mkdir(parents=True, exist_ok=True)

cat_features = ['workclass', 'education', 'marital.status', 'occupation',
                'relationship', 'race', 'sex', 'native.country']


def OHE(x_train, y_train, x_test, y_test, final):
    # Make copies so shit don't get fucky when I start poking at 'em later
    x_train = x_train.copy()
    y_train = y_train.copy()
    x_test = x_test.copy()
    y_test = y_test.copy()
    final = final.copy()

    # Concat them all to get unique values for all of them
    all = pd.concat([x_train, x_test, final])[cat_features]

    enc = OneHotEncoder(sparse=False).fit(all)
    feature_names = enc.get_feature_names_out(cat_features)

    train_encoded = enc.transform(x_train[cat_features])
    train_encoded = pd.DataFrame(train_encoded, columns=feature_names)
    x_train_new = pd.concat([x_train.drop(cat_features, axis=1),
                             train_encoded], axis=1)

    test_encoded = enc.transform(x_test[cat_features])
    test_encoded = pd.DataFrame(test_encoded, columns=feature_names)
    x_test_new = pd.concat([x_test.drop(cat_features, axis=1), test_encoded],
                           axis=1)

    final_encoded = enc.transform(final[cat_features])
    final_encoded = pd.DataFrame(final_encoded, columns=feature_names)
    final_new = pd.concat([final.drop(cat_features, axis=1), final_encoded],
                          axis=1)

    final_new['Prediction'] = 0

    return x_train_new, y_train, x_test_new, y_test, final_new


def ordinal_encoding(x_train, y_train, x_test, y_test, final):
    '''Returns ordinally encoded categorical variables
    '''
    # Make copies so shit don't get fucky when I start poking at 'em later
    x_train = x_train.copy()
    y_train = y_train.copy()
    x_test = x_test.copy()
    y_test = y_test.copy()
    final = final.copy()
    train_cat = x_train[cat_features]
    test_cat = x_test[cat_features]
    final_cat = final[cat_features]

    all = pd.concat([x_train, x_test, final])[cat_features]
    # Sklearn ordinal encoding
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=41)
    enc.fit(all)
    train_cat = pd.DataFrame(enc.transform(train_cat), columns=cat_features)
    test_cat = pd.DataFrame(enc.transform(test_cat), columns=cat_features)
    final_cat = pd.DataFrame(enc.transform(final_cat), columns=cat_features)

    # update categoricals
    x_train[train_cat.columns] = train_cat
    x_test[test_cat.columns] = test_cat
    final[final_cat.columns] = final_cat

    return x_train, y_train, x_test, y_test, final


def clean_data(data, final):
    '''Cleans data and splits into training and test
    '''
    x_train, x_test, y_train, y_test = train_test_split(data.drop('income>50K',
                                                        axis=1),
                                                        data['income>50K'])

    # KNNImputer fills missing with 'nearest neighbor's values
    imp = SimpleImputer(missing_values='?', strategy='most_frequent')
    imp.fit(x_train)
    # replace '?' with most common of matching label
    x_train = pd.DataFrame(imp.transform(x_train), columns=x_train.columns)

    x_test = pd.DataFrame(imp.transform(x_test), columns=x_test.columns)
    # clean final test data
    final_clean = pd.DataFrame(imp.transform(final.drop('ID', axis=1)),
                               columns=final.columns[1:])

    # reappend ID column
    final_clean['ID'] = final['ID']
    # Appending this so I can drop it later without error
    final_clean['Prediction'] = 0

    return x_train, y_train, x_test, y_test, final_clean


def decision_tree(x_train, y_train, x_test, y_test, final, plot=False,
                  prefix=''):

    print('Decision Tree')

    print('  No restrictions on Depth')
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    clf = clf.fit(x_train, y_train)
    print(f'    Training accuracy: {clf.score(x_train, y_train)}')
    print(f'    Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                                 axis=1))
    name = output_location + prefix + 'treeNoRestrictions.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)

    print('  Max Depth 10')
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
    clf = clf.fit(x_train, y_train)
    print(f'    Training accuracy: {clf.score(x_train, y_train)}')
    print(f'    Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                                 axis=1))
    name = output_location + prefix + 'treeMaxDepth10.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)

    print('  Max Depth 5')
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf = clf.fit(x_train, y_train)
    print(f'    Training accuracy: {clf.score(x_train, y_train)}')
    print(f'    Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                                 axis=1))
    name = output_location + prefix + 'treeMaxDepth5.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)

    if(plot):
        # This makes full screen with tkagg backend
        plt.get_current_fig_manager().window.state('zoomed')
        tree.plot_tree(clf, fontsize=6, feature_names=x_train.columns,
                       class_names=['0', '1'])
        plt.show()
    breakpoint


def random_forest(x_train, y_train, x_test, y_test, final, prefix=''):
    print(f"Random Forest: {prefix}")

    print("  Max Depth: 10")
    clf = ensemble.RandomForestClassifier(criterion='entropy', max_depth=10)

    clf.fit(x_train, y_train)
    print(f'    Training Accuracy: {clf.score(x_train, y_train)}')
    print(f'    Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                      axis=1))
    name = output_location + prefix + 'randomForestDepth10.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)

    print("  Max Depth: 5")
    clf = ensemble.RandomForestClassifier(criterion='entropy', max_depth=5)

    clf.fit(x_train, y_train)
    print(f'    Training Accuracy: {clf.score(x_train, y_train)}')
    print(f'    Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                      axis=1))
    name = output_location + prefix + 'randomForestDepth5.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)

    print("  Max Depth: 1")
    clf = ensemble.RandomForestClassifier(criterion='entropy', max_depth=1)

    clf.fit(x_train, y_train)
    print(f'    Training Accuracy: {clf.score(x_train, y_train)}')
    print(f'    Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                      axis=1))
    name = output_location + prefix + 'randomForestStumps.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)


def ada_boost(x_train, y_train, x_test, y_test, final, prefix=''):
    print(f'AdaBoost: ' + prefix)

    print(f'  Default parameters, 50 estimators, LR: 1')
    clf = ensemble.AdaBoostClassifier()
    clf.fit(x_train, y_train)
    print(f'    Training Accuracy: {clf.score(x_train, y_train)}')
    print(f'    Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                      axis=1))
    name = output_location + prefix + 'adaboostdefault.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)

    print(f'  500 estimators, LR: .01')
    clf = ensemble.AdaBoostClassifier(n_estimators=500, learning_rate=0.01)
    clf.fit(x_train, y_train)
    print(f'    Training Accuracy: {clf.score(x_train, y_train)}')
    print(f'    Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                      axis=1))
    name = output_location + prefix + 'adaboost500.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)


def bagged(x_train, y_train, x_test, y_test, final, prefix=''):
    print(f'Bagged: ' + prefix)

    clf = ensemble.BaggingClassifier()
    clf.fit(x_train, y_train)
    print(f'  Training Accuracy: {clf.score(x_train, y_train)}')
    print(f'  Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                      axis=1))
    name = output_location + prefix + 'baggeddefault.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)

    clf = ensemble.BaggingClassifier(n_estimators=500, max_features=.5)
    clf.fit(x_train, y_train)
    print(f'  Training Accuracy: {clf.score(x_train, y_train)}')
    print(f'  Test Accuracy: {clf.score(x_test, y_test)}')
    final['Prediction'] = clf.predict(final.drop(['ID', 'Prediction'],
                                      axis=1))
    name = output_location + prefix + 'bagged8feat.csv'
    final[['ID', 'Prediction']].to_csv(name, index=False)


cleaned = clean_data(train_raw, final_raw)
# decision_tree(*ordinal_encoding(*cleaned), prefix='ordinal_')

# random_forest(*ordinal_encoding(*cleaned), prefix="ordinal_")

# random_forest(*OHE(*cleaned), prefix="OHE_")

# ada_boost(*OHE(*cleaned), prefix="OHE_")
# ada_boost(*ordinal_encoding(*cleaned), prefix="ordinal_")

bagged(*OHE(*cleaned), prefix="OHE_")
bagged(*ordinal_encoding(*cleaned), prefix="ordinal_")

breakpoint
