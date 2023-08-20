import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import copy
import sklearn

# Function to split data into 5 parts and return
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    l = len(dataset)//n_folds
    for i in range(n_folds):
        dataset_split.append(dataset[i*l:(i+1)*l])
    return dataset_split

###################################################################################

# Create a function that calculates p(x|y):
def gaussian_function(x, y_mean, y_var):
    p = 1/(np.sqrt(2*np.pi*y_var)) * np.exp((-(x-y_mean)**2)/(2*y_var))
    return p

###################################################################################

# Function to train data
def train(train_data):
    # Number of Cammeo
    n_Cammeo = train_data['Class'][train_data['Class'] == 'Cammeo'].count()

    # Number of Osmancik
    n_Osmancik = train_data['Class'][train_data['Class'] == 'Osmancik'].count()

    # Total rows for checking
    total_rows=train_data['Class'].count()

    # Number of Cammeo divided by the total rows: P(Cammeo)
    P_Cammeo = n_Cammeo/total_rows

    # Number of females divided by the total rows :P(Osmancik)
    P_Osmancik = n_Osmancik/total_rows

    train_data_means = train_data.groupby('Class').mean()
    train_data_variance = train_data.groupby('Class').var()

    # Means for Cammeo
    Cammeo_Area_mean = train_data_means['Area']["Cammeo"]
    Cammeo_Perimeter_mean = train_data_means['Perimeter']["Cammeo"]
    Cammeo_Major_Axis_Length_mean = train_data_means['Major_Axis_Length']["Cammeo"]
    Cammeo_Minor_Axis_Length_mean = train_data_means['Minor_Axis_Length']["Cammeo"]
    Cammeo_Eccentricity_mean = train_data_means['Eccentricity']["Cammeo"]
    Cammeo_Convex_Area_mean = train_data_means['Convex_Area']["Cammeo"]
    Cammeo_Extent_mean = train_data_means['Extent']["Cammeo"]

    # Means for Osmancik
    Osmancik_Area_mean = train_data_means['Area']["Osmancik"]
    Osmancik_Perimeter_mean = train_data_means['Perimeter']["Osmancik"]
    Osmancik_Major_Axis_Length_mean = train_data_means['Major_Axis_Length']["Osmancik"]
    Osmancik_Minor_Axis_Length_mean = train_data_means['Minor_Axis_Length']["Osmancik"]
    Osmancik_Eccentricity_mean = train_data_means['Eccentricity']["Osmancik"]
    Osmancik_Convex_Area_mean = train_data_means['Convex_Area']["Osmancik"]
    Osmancik_Extent_mean = train_data_means['Extent']["Osmancik"]

    # Var for Cammeo
    Cammeo_Area_var = train_data_variance['Area']["Cammeo"]
    Cammeo_Perimeter_var = train_data_variance['Perimeter']["Cammeo"]
    Cammeo_Major_Axis_Length_var = train_data_variance['Major_Axis_Length']["Cammeo"]
    Cammeo_Minor_Axis_Length_var = train_data_variance['Minor_Axis_Length']["Cammeo"]
    Cammeo_Eccentricity_var = train_data_variance['Eccentricity']["Cammeo"]
    Cammeo_Convex_Area_var = train_data_variance['Convex_Area']["Cammeo"]
    Cammeo_Extent_var = train_data_variance['Extent']["Cammeo"]

    # Var for Osmancik
    Osmancik_Area_var = train_data_variance['Area']["Osmancik"]
    Osmancik_Perimeter_var = train_data_variance['Perimeter']["Osmancik"]
    Osmancik_Major_Axis_Length_var = train_data_variance['Major_Axis_Length']["Osmancik"]
    Osmancik_Minor_Axis_Length_var = train_data_variance['Minor_Axis_Length']["Osmancik"]
    Osmancik_Eccentricity_var = train_data_variance['Eccentricity']["Osmancik"]
    Osmancik_Convex_Area_var = train_data_variance['Convex_Area']["Osmancik"]
    Osmancik_Extent_var = train_data_variance['Extent']["Osmancik"]

    Osmancik_dict = {
        'Area': [Osmancik_Area_mean,Osmancik_Area_var],
        'Perimeter': [Osmancik_Perimeter_mean,Osmancik_Perimeter_var],
        'Major_Axis_Length' : [Osmancik_Major_Axis_Length_mean,Osmancik_Major_Axis_Length_var],
        'Minor_Axis_Length' : [Osmancik_Minor_Axis_Length_mean,Osmancik_Minor_Axis_Length_var],
        'Eccentricity' : [Osmancik_Eccentricity_mean,Osmancik_Eccentricity_var],
        'Convex_Area' : [Osmancik_Convex_Area_mean,Osmancik_Convex_Area_var],
        'Extent' : [Osmancik_Extent_mean,Osmancik_Extent_var]
    }

    Cammeo_dict = {
        'Area': [Cammeo_Area_mean,Cammeo_Area_var],
        'Perimeter': [Cammeo_Perimeter_mean,Cammeo_Perimeter_var],
        'Major_Axis_Length' : [Cammeo_Major_Axis_Length_mean,Cammeo_Major_Axis_Length_var],
        'Minor_Axis_Length' : [Cammeo_Minor_Axis_Length_mean,Cammeo_Minor_Axis_Length_var],
        'Eccentricity' : [Cammeo_Eccentricity_mean,Cammeo_Eccentricity_var],
        'Convex_Area' : [Cammeo_Convex_Area_mean,Cammeo_Convex_Area_var],
        'Extent' : [Cammeo_Extent_mean,Cammeo_Extent_var]
    }

    return P_Osmancik, P_Cammeo, Cammeo_dict, Osmancik_dict

###################################################################################

# Function to predict, test and return accuracy scores
def predict_and_test(Cammeo_dict, Osmancik_dict, P_Osmancik, P_Cammeo, validation_data, feature_list):

    t_p = 0
    t_n = 0
    f_p = 0
    f_n = 0

    for i in range(len(validation_data)):
        post_Cammeo = 1
        post_Osmancik = 1
        for feature in feature_list:
            post_Cammeo *= gaussian_function(validation_data.iloc[i][feature], Cammeo_dict[feature][0], Cammeo_dict[feature][1])
            post_Osmancik *= gaussian_function(validation_data.iloc[i][feature], Osmancik_dict[feature][0], Osmancik_dict[feature][1])
        
        post_Osmancik*=P_Osmancik
        post_Cammeo*=P_Cammeo
        
        predicted_res = ""
        if post_Cammeo > post_Osmancik :
            #print("Cammeo")
            predicted_res="Cammeo"
        else:
            #print("Osmancik")
            predicted_res="Osmancik"
        
        # Cam - positive    Osm - negative
        if predicted_res == "Cammeo" and validation_data.iloc[i]['Class'] == "Cammeo":
            t_p+=1
        if predicted_res == "Osmancik" and validation_data.iloc[i]['Class'] == "Osmancik":
            t_n+=1
        if predicted_res == "Cammeo" and validation_data.iloc[i]['Class'] == "Osmancik":
            f_p+=1
        if predicted_res == "Osmancik" and validation_data.iloc[i]['Class'] == "Cammeo":
            f_n+=1

    accuracy = (t_p+t_n)/(t_p+t_n+f_p+f_n)

    precision_Cam = t_p/(f_p+t_p)
    precision_Osm = t_n/(f_n+t_n)

    recall_Cam = t_p/(f_n+t_p)
    recall_Osm = t_n/(f_p+t_n)

    f1_score_Cam = (2*precision_Cam*recall_Cam)/(precision_Cam+recall_Cam)
    f1_score_Osm = (2*precision_Osm*recall_Osm)/(precision_Osm+recall_Osm)

    n_test_Cammeo = validation_data['Class'][validation_data['Class'] == 'Cammeo'].count()
    n_test_Osmancik = validation_data['Class'][validation_data['Class'] == 'Osmancik'].count()

    return [accuracy, precision_Cam, precision_Osm, recall_Cam, recall_Osm, f1_score_Cam, f1_score_Osm, n_test_Cammeo, n_test_Osmancik]

# Print Result in tabular form
def print_res(accuracy_params):
    accuracy, precision_Cam, precision_Osm, recall_Cam, recall_Osm, f1_score_Cam, f1_score_Osm, n_test_Cammeo, n_test_Osmancik = accuracy_params
    print("\t\tprecision\trecall\t\tf1-score\tsupport")
    print()
    print("Osmancik (0)\t{osm_pre:.2f}\t\t{osm_recall:.2f}\t\t{osm_f1_score:.2f}\t\t{osm_sup}".format(osm_pre=precision_Osm,osm_recall= recall_Osm,osm_f1_score=f1_score_Osm,osm_sup=n_test_Osmancik))
    print("Cammeo (1)\t{cam_pre:.2f}\t\t{cam_recall:.2f}\t\t{cam_f1_score:.2f}\t\t{cam_sup}".format(cam_pre=precision_Cam,cam_recall= recall_Cam,cam_f1_score=f1_score_Cam,cam_sup=n_test_Cammeo))
    print()
    print("accuracy\t\t\t\t\t{acc:.2f}\t\t{acc_sup}".format(acc=accuracy,acc_sup=n_test_Cammeo+n_test_Osmancik))
    print()

###################################################################################
    
def predict_and_test_sklearn(dataset):
    #Using Sklearn

    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn import metrics

    # dataset = dataset.drop(["Area"], axis = 1)
    # dataset = dataset.drop(["Convex_Area"], axis = 1)
    dataset.Class = [1 if i== "Cammeo" else 0 for i in dataset.Class]
    X = dataset.drop(["Class"], axis = 1)
    y = dataset.Class.values

    X = (X - np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X,axis=0))

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = np.around(nb.predict(x_test))

    #print("Naive Bayes score: ",nb.score(x_test, y_test))
    print("Classification Report of Model trained using Sklearn-")
    print(metrics.classification_report(y_test, y_pred))

###################################################################################

if __name__ == '__main__':

    # Read Dataset
    dataset = pd.read_csv("./Rice_Cammeo_Osmancik.csv")

    shuffled_data= dataset.sample(frac=1)
    train_data_num = int(shuffled_data.shape[0]*0.7)
    train_data = shuffled_data.iloc[:train_data_num]
    test_data = shuffled_data.iloc[train_data_num:]

    folds = cross_validation_split(train_data, 5)

    # Prediction list complete
    feature_list1=['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']

    models = []
    i=0
    for i in range(5):
        train_set = pd.concat(folds[j] for j in range(5) if j!=i)
        P_Osmancik, P_Cammeo, Cammeo_dict, Osmancik_dict = train(train_set)
        accuracy_params = predict_and_test(Cammeo_dict, Osmancik_dict, P_Osmancik, P_Cammeo, folds[i],feature_list1)
        models.append([P_Osmancik, P_Cammeo, Cammeo_dict, Osmancik_dict, accuracy_params])
    
    models.sort(key=lambda x: x[4][0], reverse=True)
    
    # Final testing on 30% data
    final_accuracy_params = predict_and_test(models[0][2], models[0][3], models[0][0], models[0][1], test_data, feature_list1)
    print("Classification Report of Model trained from Scratch-")
    print_res(final_accuracy_params)

    # Hyperparameter tuning
    # Prediction list complete
    feature_list2=['Area', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Extent']

    models_tuned = []
    i=0
    for i in range(5):
        train_set = pd.concat(folds[j] for j in range(5) if j!=i)
        P_Osmancik, P_Cammeo, Cammeo_dict, Osmancik_dict = train(train_set)
        accuracy_params = predict_and_test(Cammeo_dict, Osmancik_dict, P_Osmancik, P_Cammeo, folds[i],feature_list2)
        models_tuned.append([P_Osmancik, P_Cammeo, Cammeo_dict, Osmancik_dict, accuracy_params])
    
    models_tuned.sort(key=lambda x: x[4][0], reverse=True)
    
    # Final testing on 30% data
    final_accuracy_params = predict_and_test(models_tuned[0][2], models_tuned[0][3], models_tuned[0][0], models_tuned[0][1], test_data, feature_list2)
    print("Classification Report of Model trained from Scratch (after altering features) -")
    print_res(final_accuracy_params)

    # Predict using sklearn
    predict_and_test_sklearn(dataset)
        
    # Code to find the redundant parameter

    accu=[[],[],[],[],[],[],[]]
    feature_list2=['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']
    for k in range(len(feature_list2)):
        new_list = copy.deepcopy(feature_list2)
        new_list.pop(k)
        print(k)
        for j in range(5):
            models_tuned = []
            i=0
            for i in range(5):
                train_set = pd.concat(folds[j] for j in range(5) if j!=i)
                P_Osmancik, P_Cammeo, Cammeo_dict, Osmancik_dict = train(train_set)
                accuracy_params = predict_and_test(Cammeo_dict, Osmancik_dict, P_Osmancik, P_Cammeo, folds[i],new_list)
                models_tuned.append([P_Osmancik, P_Cammeo, Cammeo_dict, Osmancik_dict, accuracy_params])
            
            models_tuned.sort(key=lambda x: x[4][0], reverse=True)

            # Final testing on 30% data
            final_accuracy_params = predict_and_test(models_tuned[0][2], models_tuned[0][3], models_tuned[0][0], models_tuned[0][1], test_data, new_list)
            accu[k].append(final_accuracy_params[0])

    print(accu)
    
    
    print("Classification Report of Model trained from Scratch (after altering features) -")
    print_res(final_accuracy_params)

        
