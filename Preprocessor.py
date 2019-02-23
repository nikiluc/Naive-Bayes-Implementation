import numpy as np
from sklearn.impute import SimpleImputer
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import TenCrossValidation as tcv


def preprocess(file):


    data = arff.loadarff(file)

    df = pd.DataFrame(data[0]) # loading file as a dataframe

    convert = df.select_dtypes([np.object])
    convert = convert.stack().str.decode('utf-8').unstack() # remove b

    for col in convert:
        df[col] = convert[col]


    df.replace('?', np.NaN, inplace = True) # replace all question marks with NAN

    tf = df.values # numpy array

    # kbins discretization for all features
    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    numeric_transformer = Pipeline(steps=[
        ('kbins', KBinsDiscretizer(n_bins = 7, encode='ordinal', strategy='uniform'))])


    # impute missing vales (all categorical)
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country', 'class']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.NaN, strategy='most_frequent'))])


    ct = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])


    test = ct.fit_transform(df)


    i = 1

    # replace values from ordinal encoding with bin edges

    while i < len(test) + 1:



        if (test[i - 1:i][:,0] == 0.0):

            test[i - 1:i][:, 0] = '17-27'


        elif (test[i - 1:i][:,0] == 1.0):

            test[i - 1:i][:, 0] = '28-37'

        elif (test[i - 1:i][:,0] == 2.0):

            test[i - 1:i][:, 0] = '38-48'

        elif (test[i - 1:i][:,0] == 3.0):

            test[i - 1:i][:, 0] = '49-58'

        elif (test[i - 1:i][:,0] == 4.0):

            test[i - 1:i][:, 0] = '59-69'

        elif (test[i - 1:i][:,0] == 5.0):

            test[i - 1:i][:, 0] = '70-79'

        elif (test[i - 1:i][:,0] == 6.0):

            test[i - 1:i][:, 0] = '80-90'

        i = i + 1


    i = 1

    while i < len(test) + 1:

        if (test[i - 1:i][:,1] == 0.0):


            test[i - 1:i][:,1] = '12285-223444'

        elif (test[i - 1:i][:,1] == 1.0):

            test[i - 1:i][:,1] = '223444-434603'

        elif (test[i - 1:i][:,1] == 2.0):

            test[i - 1:i][:, 1] = '434604-645762'

        elif (test[i - 1:i][:,1] == 3.0):

            test[i - 1:i][:, 1] = '645763-856922'

        elif (test[i - 1:i][:,1] == 4.0):

            test[i - 1:i][:, 1] = '856923-1068081'

        elif (test[i - 1:i][:,1] == 5.0):

            test[i - 1:i][:, 1] = '1068082-1279240'

        elif (test[i - 1:i][:,1] == 6.0):

            test[i - 1:i][:, 1] = '1279241-1490400'

        i = i + 1

    i = 1

    while i < len(test) + 1:

        if (test[i - 1:i][:, 2] == 0.0):

            test[i - 1:i][:, 2] = '1-3'

        elif (test[i - 1:i][:, 2] == 1.0):

            test[i - 1:i][:, 2] = '4-5'

        elif (test[i - 1:i][:, 2] == 2.0):

            test[i - 1:i][:, 2] = '6-7'

        elif (test[i - 1:i][:, 2] == 3.0):

            test[i - 1:i][:, 2] = '8-9'

        elif (test[i - 1:i][:, 2] == 4.0):

            test[i - 1:i][:, 2] = '9-10'

        elif (test[i - 1:i][:, 2] == 5.0):

            test[i - 1:i][:, 2] = '11-13'

        elif (test[i - 1:i][:, 2] == 6.0):

            test[i - 1:i][:, 2] = '14-16'

        i = i + 1


    i = 1

    while i < len(test) + 1:

        if (test[i - 1:i][:, 3] == 0.0):

            test[i - 1:i][:, 3] = '0-14285'

        elif (test[i - 1:i][:, 3] == 1.0):

            test[i - 1:i][:, 3] = '14286-28571'

        elif (test[i - 1:i][:, 3] == 2.0):

            test[i - 1:i][:, 3] = '28572-42856'

        elif (test[i - 1:i][:, 3] == 3.0):

            test[i - 1:i][:, 3] = '42857-57142'

        elif (test[i - 1:i][:, 3] == 4.0):

            test[i - 1:i][:, 3] = '57143-71427'

        elif (test[i - 1:i][:, 3] == 5.0):

            test[i - 1:i][:, 3] = '71427-85713'

        elif (test[i - 1:i][:, 3] == 6.0):

            test[i - 1:i][:, 3] = '85714-99999'

        i = i + 1

    i = 1

    while i < len(test) + 1:

        if (test[i - 1:i][:, 4] == 0.0):

            test[i - 1:i][:, 4] = '0-622'

        elif (test[i - 1:i][:, 4] == 1.0):

            test[i - 1:i][:, 4] = '623-1244'

        elif (test[i - 1:i][:, 4] == 2.0):

            test[i - 1:i][:, 4] = '1245-1866'

        elif (test[i - 1:i][:, 4] == 3.0):

            test[i - 1:i][:, 4] = '1867-2489'

        elif (test[i - 1:i][:, 4] == 4.0):

            test[i - 1:i][:, 4] = '2490-3111'

        elif (test[i - 1:i][:, 4] == 5.0):

            test[i - 1:i][:, 4] = '3112-3733'

        elif (test[i - 1:i][:, 4] == 6.0):

            test[i - 1:i][:, 4] = '3734-4356'

        i = i + 1

    i = 1

    while i < len(test) + 1:

        if (test[i - 1:i][:, 5] == 0.0):

            test[i - 1:i][:, 5] = '1-15'

        elif (test[i - 1:i][:, 5] == 1.0):

            test[i - 1:i][:, 5] = '16-29'

        elif (test[i - 1:i][:, 5] == 2.0):

            test[i - 1:i][:, 5] = '30-43'

        elif (test[i - 1:i][:, 5] == 3.0):

            test[i - 1:i][:, 5] = '44-57'

        elif (test[i - 1:i][:, 5] == 4.0):

            test[i - 1:i][:, 5] = '58-71'

        elif (test[i - 1:i][:, 5] == 5.0):

            test[i - 1:i][:, 5] = '72-85'

        elif (test[i - 1:i][:, 5] == 6.0):

            test[i - 1:i][:, 5] = '86-99'

        i = i + 1


    return test





















