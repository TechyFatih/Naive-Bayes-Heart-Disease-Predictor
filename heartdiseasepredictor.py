from csv import reader
from naiveBayes import naive_bayes

TRAIN_AMOUNT = 152 #Amount of training data
TEST_AMOUNT = 151 #Amount of testing data


###---INITIALIZATION---###


train = []
test = []

with open('heartdisease.csv') as csvfile:
    scanner = reader(csvfile)

    count = 0
    for row in scanner:
        #Features:   1: Sex | 5: Blood sugar | 6: EC Result | 8: Angina | 13: Heart disease
        new_row = [row[1],row[5],row[6],row[8],row[13]]
        for index, item in enumerate(new_row):
            item = int(bool(float(item)))
            new_row[index] = item
        if count <= TRAIN_AMOUNT:
            train.append(new_row)
        else:
            test.append(new_row)
        count += 1

heart_disease = [] #Probabilities of heart disease (0: Yes | 1: No)
evidence = [ [[],[]], [[],[]], [[],[]], [[],[]] ] #Probabilities of evidence given heart disease
#1st dimension: Features (0: Sex | 1: Blood sugar | 2: EC Result | 3: Angina | 4: Heart disease)
#2nd dimension: Binary value (0: Yes | 1: No)
#3rd dimension: Heart disease binary (0: Yes | 1: No)


###---TRAINING---###


hd_count = 0 #Heart disease count
features_count = [0,0,0,0] #"True" value counts of each feature
features_countHD = [0,0,0,0] #Same as above, but only if patient has heart disease

for patient in train:
    hd_count += patient[4]

    for index in range(0,4):
        features_count[index] += patient[index]
        if patient[4]:
            features_countHD[index] += patient[index]

heart_disease.append(hd_count/TRAIN_AMOUNT) #Has heart disease
heart_disease.append(1-hd_count/TRAIN_AMOUNT) #Does not have heart disease

index = 0
for feature in features_count:
    fc = features_count[index]
    fcHD = features_countHD[index]
    evidence[index][0].append(fc/fcHD)                                  #Value: True | HD: True
    evidence[index][0].append(TRAIN_AMOUNT-fc/fcHD)                     #Value: True | HD: False
    evidence[index][1].append(fc/TRAIN_AMOUNT-fcHD)                     #Value: False | HD: True
    evidence[index][1].append((TRAIN_AMOUNT-fc)/(TRAIN_AMOUNT-fcHD))    #Value: False | HD: False
    index += 1
    

###---TESTING---###


results = []
for patient in test:
    evidenceProbs = []
    for index in range(0,4):
        #Complicated 1-liner which decides which conditinoal probabilities to use
        evidenceProbs.append(evidence[index][patient[index]][patient[4]])
    results.append(naive_bayes(heart_disease, evidenceProbs))


###---EVALUATION---###


accuracy = 0
for index, result in enumerate(results):
    if result == test[index][4]:
        accuracy += 1

print("Predicting heart disease with Naive Bayes...")
print("Accuracy:", accuracy/TEST_AMOUNT)
