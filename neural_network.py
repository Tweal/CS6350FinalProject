import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance

train_raw = pd.read_csv('Data/train_final.csv')
final_raw = pd.read_csv('Data/test_final.csv')

output_location = 'Data/Attempts/NN/'
Path(output_location).mkdir(parents=True, exist_ok=True)

cat_features = ['workclass', 'education', 'marital.status', 'occupation',
                'relationship', 'race', 'sex', 'native.country']


def ordinal_encoding(x_train, x_test, y_train, y_test, final):
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
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=404)
    enc.fit(all)
    train_cat = pd.DataFrame(enc.transform(train_cat), columns=cat_features)
    test_cat = pd.DataFrame(enc.transform(test_cat), columns=cat_features)
    final_cat = pd.DataFrame(enc.transform(final_cat), columns=cat_features)

    # update categoricals
    x_train[train_cat.columns] = train_cat
    x_test[test_cat.columns] = test_cat
    final[final_cat.columns] = final_cat

    return x_train, x_test, y_train, y_test, final


def clean_data(data, final):
    '''Cleans data and splits into training and test
    '''
    split = train_test_split(data.drop('income>50K', axis=1),
                             data['income>50K'])

    x_train, x_test, y_train, y_test, final = ordinal_encoding(*split, final)

    # KNNImputer fills missing with 'nearest neighbor's values
    imp = KNNImputer(n_neighbors=20)
    imp.fit(x_train)
    # replace NAN with value of 20 closest neighbors
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


def importance(train_x, train_y, test_x, test_y, _):
    model = KNeighborsRegressor()
    model.fit(train_x, train_y)

    def run(X, y):
        results = permutation_importance(model, X, y, n_repeats=50,
                                         scoring='neg_mean_squared_error')
        importance = results.importances_mean
        # summarize feature importance
        for i, v in enumerate(importance):
            print(f'Feature: {X.columns[i]}, Score: {v:.5f}')

    print('Train Importance')
    run(train_x, train_y)
    print('Test Importance')
    run(test_x, test_y)


def nn(train_x, train_y, test_x, test_y, final):

    low_value_features = ['workclass', 'education', 'marital.status',
                          'occupation', 'relationship', 'race', 'sex',
                          'native.country']
    new_train = train_x.drop(low_value_features, axis=1)
    new_test = test_x.drop(low_value_features, axis=1)

    # feature_check(train_x, train_y, test_x, test_y, new_train, new_test)

    new_train, new_test = standardize_check(new_train, train_y,
                                            new_test, test_y)

    # layer_check(new_train, train_y, new_test, test_y)

    # Final model and results
    print("Final model results:")
    cols = ["ID", "Prediction"]
    pruned_final = final.drop(low_value_features + cols, axis=1)
    normed_final = StandardScaler().fit(pruned_final).transform(pruned_final)
    model = MLPClassifier(hidden_layer_sizes=(500,))
    model.fit(new_train, train_y)
    print(f'    Training Accuracy: {model.score(new_train, train_y):.3f}')
    print(f'    Test Accuracy: {model.score(new_test, test_y):.3f}')
    final['Prediction'] = model.predict(normed_final)
    name = output_location + 'nnfinal.csv'
    final[cols].to_csv(name, index=False)


def feature_check(train_x, train_y, test_x, test_y, pruned_train, pruned_test):
    print('Default Parameters')
    model = MLPClassifier()
    model.fit(train_x, train_y)
    print(f'    Training Accuracy: {model.score(train_x, train_y):.3f}')
    print(f'    Test Accuracy: {model.score(test_x, test_y):.3f}')

    print('Pruned Features')
    model.fit(pruned_train, train_y)
    print(f'    Training Accuracy: {model.score(pruned_train, train_y):.3f}')
    print(f'    Test Accuracy: {model.score(pruned_test, test_y):.3f}')


def standardize_check(train_x, train_y, test_x, test_y):
    print('Normalized Data')
    scaler = StandardScaler().fit(train_x)
    normed_train = scaler.transform(train_x)
    scaler = StandardScaler().fit(test_x)
    normed_test = scaler.transform(test_x)

    model = MLPClassifier(max_iter=1000)
    model.fit(normed_train, train_y)
    print(f'    Training Accuracy: {model.score(normed_train, train_y):.3f}')
    print(f'    Test Accuracy: {model.score(normed_test, test_y):.3f}')

    print('[0,1] Scaled Data')
    scaler = MinMaxScaler().fit(train_x)
    scaled_train = scaler.transform(train_x)
    scaler = MinMaxScaler().fit(test_x)
    scaled_test = scaler.transform(test_x)

    model.fit(normed_train, train_y)
    print(f'    Training Accuracy: {model.score(scaled_train, train_y):.3f}')
    print(f'    Test Accuracy: {model.score(scaled_test, test_y):.3f}')

    return normed_train, normed_test


def layer_check(train_x, train_y, test_x, test_y):
    for n in [3, 10, 50, 100, 500, 1000]:
        print(f'{n} layers')
        model = MLPClassifier(hidden_layer_sizes=(n,))
        model.fit(train_x, train_y)
        print(f'    Training Accuracy: {model.score(train_x, train_y):.3f}')
        print(f'    Test Accuracy: {model.score(test_x, test_y):.3f}')

    print('4 layer 120 neuron layers')
    model = MLPClassifier(hidden_layer_sizes=(4, 120))
    model.fit(train_x, train_y)
    print(f'    Training Accuracy: {model.score(train_x, train_y):.3f}')
    print(f'    Test Accuracy: {model.score(test_x, test_y):.3f}')


cleaned = clean_data(train_raw, final_raw)

# importance(*cleaned)

nn(*cleaned)
