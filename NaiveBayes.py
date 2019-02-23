

class NaiveBayes:

    def __init__(self, data):

        counts = {}  # Holds counts of each category in a feature
        self.classes = {}  # >50K <=50k
        self.prob = {}

        for fold in data:  # element in list

            for item in fold:  # data rows in first element
                i = 0
                while i < len(item):  # per val in each feature
                    if item[i] in counts:

                        counts[item[i]] += 1

                    else:
                        # item doesn't exist yet
                        counts[item[i]] = 1

                    if item[i] == item[14]:  # label (add row to appropriate item in classes)
                        if item[i] in self.classes:

                            self.classes[item[i]].append(item)

                        else:

                            self.classes[item[i]] = [item]
                    i = i + 1

        for item in counts.keys():
            # calculate probabilities for each item in each feature
            self.prob[item] = self.calcProb(counts[item], len(data))

        features = counts.keys()

        for item in self.classes.keys():  # >50K, <= 50K
            for row in self.classes[item]:

                i = 0
                # creation of key to hold count for each category in each class label
                while i < len(row):
                    item2 = row[i] + " " + item

                    if item2 in counts:
                        counts[item2] += 1

                    else:
                        counts[item2] = 1

                    i = i + 1

            # creating keys to store probabilities
            for feature in features:

                count = feature + " " + item

                if count in counts:

                    self.prob[count] = self.calcProb(counts[count], len(self.classes[item]))

                else:
                    self.prob[count] = self.calcProb(0, len(self.classes[item]))


    # calculate probability (.5 and 1 for smoothing)

    def calcProb(self, num, total):

        return (num + .5) / (total + 1)


    def choose(self, data):

        storedp = {}

        for classLabel in self.classes:
            # Calculate total probability by multiplying them together
            i = 0
            total = None

            while i < len(data):
                # everything but class label
                if data[i] != data[14]:

                    key = str(data[i]) + " " + classLabel

                    if total == None:   # probability to be stored

                        total = self.prob[key]

                    else:

                        # special case

                        if key == 'Holand-Netherlands <=50K':
                            self.prob[key] = 1

                        elif key == 'Holand-Netherlands >50K':
                            self.prob[key] = 0

                        total *= self.prob[key]

                i = i + 1
            storedp[classLabel] = total * self.prob[classLabel]

        choice = [0, None]

        # class label returned

        for classLabel in storedp:

            if storedp[classLabel] > choice[0]:

                choice = [storedp[classLabel], classLabel]

        return choice[1]  # return label or none
