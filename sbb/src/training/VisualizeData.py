import os
import shutil
import sys
import tensorflow as tf

def PrintDataToTxt(model,test_image_count,x_test,y_test,x_test_images):

    index = 0
    i = 0

    print("Creating folders...")
    index = CreateDir(index)

    print("Printing data to txt file and saving images...")
    with open("test.txt", "w") as f:#sys.stdout
        for i in range(test_image_count): 

            predictions = model.predict(x_test[i:i+1],verbose = 0)
            predictionIsBridge = predictions[0][1]
            predictionIsNotBridge = predictions[0][0]
            f.write("ID{}\tIsBridge={}\tpredictionIsBridge{:.7f}\tpredictionIsNotBridge= {:.7f}\n".format(i,y_test[i],predictionIsBridge,predictionIsNotBridge))
            #print("ID= "+str(i), "IsBridge= "+str(y_test[i]),  "predictionIsBridge= %.7f" % predictionIsBridge, "predictionIsNotBridge= %.7f"  % predictionIsNotBridge, sep="\t")

            if (predictionIsBridge >= 0.5 and y_test[i][1] == 1):
                #is true bridge
                SaveImage(x_test_images[i],"\TrueBridge\\"+str(i)+".raw",index)
            if (predictionIsNotBridge >= 0.5 and y_test[i][1] == 1):
                #is false bridge
                SaveImage(x_test_images[i],"\FalseBridge\\"+str(i)+".raw",index)
            if (predictionIsBridge < 0.5 and y_test[i][1] == 0):
                #is true non-bridge
                SaveImage(x_test_images[i],"\TrueNonBridge\\"+str(i)+".raw",index)
            if(predictionIsNotBridge < 0.5 and y_test[i][1] == 0):
                #is false non-bridge
                SaveImage(x_test_images[i],"\FalseNonBridge\\"+str(i)+".raw",index)


def CreateDir(index):

    pathCreated = False
    #check if path exists if it does skip and check next one
    while(not pathCreated):
        if (not os.path.isdir(os.path.abspath(os.getcwd()) + "\\ImageClasses\\" + str(index))):
            os.mkdir(os.path.abspath(os.getcwd()) + "\\ImageClasses\\" + str(index))
            os.mkdir(os.path.abspath(os.getcwd()) + "\\ImageClasses\\" + str(index) + "\TrueBridge")
            os.mkdir(os.path.abspath(os.getcwd()) + "\\ImageClasses\\" + str(index) + "\FalseBridge")
            os.mkdir(os.path.abspath(os.getcwd()) + "\\ImageClasses\\" + str(index) + "\TrueNonBridge")
            os.mkdir(os.path.abspath(os.getcwd()) + "\\ImageClasses\\" + str(index) + "\FalseNonBridge")
            pathCreated = True

        else:
            index += 1
    return index
            

def SaveImage(image, path, index):

    f = open(os.path.abspath(os.getcwd()) + "\\ImageClasses\\"+ str(index) +path,"wb")
    f.write(image)
    f.close()

def PrintDataToTxt2(model, val_data):
    

    with open("D:\\test2.txt", 'w') as f:
        for x, y_true in val_data.unbatch():  
            x = tf.expand_dims(x, axis=0)  # add batch dimension for model prediction
            y_pred = model.predict(x)
            y_pred = tf.squeeze(y_pred, axis=0)  # Remove batch dimension from prediction
          
            y_true_label = tf.argmax(y_true, axis=-1).numpy()
            y_pred_label = tf.argmax(y_pred, axis=-1).numpy()
        
            f.write("IsBridge={}\tpredictionIsBridge{:.7f}\tpredictionIsNotBridge= {:.7f}\n".format(y_true_label,y_pred[1],y_pred[0]))

