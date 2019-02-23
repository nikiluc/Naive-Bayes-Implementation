import NaiveBayes as nb
from random import randrange
from random import seed
import sys


sys.stdout = open('adults.out', 'w')

seed(1)

def precision(tp, fp):  # calculate precision

    return (tp) / (tp + fp)


def calcRecall(tp, fn):  # calculate recall

    return tp / (tp + fn)


def calcf1(r, p):  # calculate f1

    return (2 * r * p) / (r + p)


def microRecall(tpList, fn): # calculate micro recall

    return sum(tpList) / (sum(tpList) + sum(fn))


def microPrecision(tpList, fpList):  # calculate micro precision

    return (sum(tpList)) / ((sum(tpList)) + sum(fpList))


def calcMacro(items): # for both macro precision and recall

    return (sum(items)) / (len(items))


def calcAccuracy(tpList, tnList, fpList, fnList): # calculate accuracy

    return (sum(tpList) + sum(tnList)) / (sum(tpList) + sum(tnList) + sum(fpList) + sum(fnList))

def calcAverage(list):

    return (sum(list)) /(len(list))




def validation(data, folds = 10):

    copy = list(data)
    split = list()

    # stores results from each model

    mipavg = []

    miravg = []

    mapavg = []

    maravg = []

    mif1 = []

    maf1 = []

    accavg = []


    sizeofFold = int(len(data)/folds)

    # split the data into folds

    for i in range(folds):

        fold = list()

        while len(fold) < sizeofFold:

            index = randrange(len(copy))

            fold.append(copy.pop(index))

        split.append(fold)


    classes = ['<=50K', '>50K']

    i = 0

    train = None

    while i < folds:

        print('Model #' + str(i + 1))


        # re-append previously popped fold for new model
        if train != None:
            split.append(train)

        countdata = {} # dictionary for confusion matrix values for each label


        # List per label
        tpList = []
        fpList = []
        tnList = []
        fnList = []
        recalls = []
        precis = []
        f1List = []


        # confusion matrix values for each label

        countdata[">50KtrueP"] = 0
        countdata["<=50KtrueP"] = 0
        countdata[">50KfalseP"] = 0
        countdata["<=50KfalseP"] = 0
        countdata["<=50KtrueN"] = 0
        countdata[">50KtrueN"] = 0
        countdata[">50KfalseN"] = 0
        countdata["<=50KfalseN"] = 0


        testdata = split[i]  # 1/10 data used for testing

        train = split.pop(0)

        classifier = nb.NaiveBayes(split)


        for item in testdata:

            correct = item[14]

            prediction = classifier.choose(item)

            if correct == prediction:  # correct prediction

                countdata[correct + "trueP"] += 1

                for label in classes:

                    if label != correct:

                        countdata[label + "trueN"] += 1  # right answer but for different class label

            else:  # incorrect prediction
                countdata[correct + "falseN"] += 1

                countdata[prediction + "falseP"] += 1


        # append data to appropriate list
        for label in classes:

            tpList.append(countdata[label + "trueP"])

            tnList.append(countdata[label + "trueN"])

            fpList.append(countdata[label + "falseP"])

            fnList.append(countdata[label + "falseN"])

            p = precision(countdata[label + "trueP"], countdata[label + "falseP"])

            r = calcRecall(countdata[label + "trueP"], countdata[label + "falseN"])

            precis.append(p)

            recalls.append(r)

            f1List.append(calcf1(r, p))


        # calculations of micro/macro precision, recall, f1 and accuracy

        microPrec = microPrecision(tpList, fpList)


        microRec = microRecall(tpList, fnList)

        macroPrec = calcMacro(precis)

        macroRec = calcMacro(recalls)

        microF1 = calcf1(microRec, microPrec)

        macroF1 = calcMacro(f1List)

        accuracy = calcAccuracy(tpList, tnList, fpList, fnList)


        # store values from each iteration

        mipavg.append(microPrec)

        miravg.append(microRec)

        mapavg.append(macroPrec)

        maravg.append(macroRec)

        mif1.append(microF1)

        maf1.append(macroF1)

        accavg.append(accuracy)


        print("Micro Precision: " + str(microPrec))

        print("Micro Recall: " + str(microRec))

        print("Micro F1: " + str(microF1))

        print("Macro Precision: " + str(macroPrec))

        print("Macro Recall: " + str(macroRec))

        print("Macro F1: " + str(macroF1))

        print("Accuracy: " + str(accuracy))

        print('\n')

        i = i + 1

    # calculate and print out averages

    finalmicroP = calcAverage(mipavg)
    finalmicroR = calcAverage(miravg)
    finalmacroP = calcAverage(mapavg)
    finalmacroR = calcAverage(maravg)
    finalmacroF1 = calcAverage(maf1)
    finalmicroF1 = calcAverage(mif1)
    finalaccuracy = calcAverage(accavg)


    print("Results: ")

    print("Average Micro Precision: " + str(finalmicroP))

    print("Average Micro Recall: " + str(finalmicroR))

    print("Average Micro F1: " + str(finalmicroF1))

    print("Average Macro Precision: " + str(finalmacroP))

    print("Average Macro Recall: " + str(finalmacroR))

    print("Average Macro F1: " + str(finalmacroF1))

    print("Average Accuracy: " + str(finalaccuracy))
