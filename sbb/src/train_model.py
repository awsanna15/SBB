import os

import tensorflow as tf

from training.model import SBBModeling
from training.data import SBBDataLoader
from training.VisualizeData import PrintDataToTxt
from training.VisualizeData import PrintDataToTxt2
 
import tf2onnx
import onnx

def main():
    im_size = 56
    # train_data, val_data = SBBDataLoader.create_generators(src_path=r'D:\AIML-Image Data\SBB\Data\Version9\Test2', im_size=im_size)
    train_data, val_data = SBBDataLoader.create_generators_softmax(src_path=r'C:\Data\Version9\Test3', im_size=im_size)
    val_data_ub = val_data.unbatch()
    mpdata= val_data_ub.map(lambda x, y: x)
    len_data = tf.reduce_sum(mpdata.reduce(0,lambda x,_: x+1)).numpy()
    images = list(val_data_ub.map(lambda x, y: x))
    labels = list(val_data_ub.map(lambda x, y: y))
    model = SBBModeling.build_model(im_size=im_size)
    #model = SBBModeling.build_resnet(im_size=im_size)
    #model = SBBModeling.build_resnet_reg(im_size=im_size)
    #model = SBBModeling.build_custom_resnet(im_size=im_size)
    #model = SBBModeling.build_origin_model(im_size=im_size)

    if False:
        cce = tf.keras.losses.CategoricalCrossentropy()
        for tmp_x, tmp_y in train_data:
            tmp_y_hat = model(tmp_x)
            loss = cce(tmp_y, tmp_y_hat).numpy()
            break

    exp_dir = r'C:\Users\annag\SBB_model_02\SBB_model_02\sbb\Data\experiments'
    n = len(os.listdir(exp_dir))
    training_dir = os.path.join(exp_dir, '{:04d}'.format(n))
    os.makedirs(training_dir)
    
    log_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(training_dir, 'logs'), histogram_freq=1)
    model_path = os.path.join(training_dir, 'model')
    check_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path, 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True,
        save_freq='epoch',
    )
    callbacks = [log_cb, check_cb]
            
    model.fit(train_data, validation_data=val_data, epochs=100, callbacks=callbacks)

    # save the model to disk

    model.save(os.path.abspath(os.getcwd())+"\savedmodel",overwrite=True)


    # Use from_function for tf functions
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    onnx.save(onnx_model, os.getcwd()+"\savedmodel\SBBmodel.onnx")
    images=tf.stack(images)
    labels=tf.stack(labels)
    score, acc = model.evaluate(images ,labels)
    print('Test accuracy:', acc)
    data_size= val_data.cardinality().numpy()
    PrintDataToTxt2(model, val_data)
    #PrintDataToTxt(model,val_data.size(),val_data.x , val_data.y, val_data.x)
    for x_batch, y_batch in val_data:
        x_test = x_batch.numpy()
        y_test = y_batch.numpy()
        PrintDataToTxt(model,data_size,x_test , y_test, x_test)
   


if __name__ == '__main__':
    main()

