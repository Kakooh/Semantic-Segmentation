from __future__ import print_function
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import matplotlib.pyplot as plt

from glob import glob
import scipy.misc
import numpy as np    

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'    
    
    # Load the model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # extract the layers of the vgg
    vgg_graph = tf.get_default_graph()
    image_input = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
#tests.test_load_vgg(load_vgg, tf)


#def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
#    """
#    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
#    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
#    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
#    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
#    :param num_classes: Number of classes to classify
#    :return: The Tensor for the last layer of output
#    """
#    # TODO: Implement function
#    
#    # FCN Layer-8: The last fully connected layer of VGG16 is replaced by a 1x1 convolution.
#    FCN8 = tf.layers.conv2d(inputs=vgg_layer7_out, 
#                            filters=num_classes, 
#                            kernel_size=[1, 1], name="fcn8")
#    
#    # FCN Layer-9: FCN Layer-8 is upsampled 2 times to match dimensions with Layer 4 of VGG 16
#    
#    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so 
#    # that we can add skip connection with 4th layer
#    FCN9 = tf.layers.conv2d_transpose(inputs=FCN8, 
#                                      filters=vgg_layer4_out.get_shape().as_list()[-1],
#                                      kernel_size=4, strides=(2, 2), 
#                                      padding='SAME', name="fcn9")
#   
#    # At each stage, the upsampling process is further refined by adding features 
#    # from coarser but higher resolution feature maps from lower layers in VGG16.
#    # skip connection (element-wise addition)
#    FCN9_skip = tf.add(FCN9, vgg_layer4_out, name="fcn9_plus_vgg_layer4")
#
#    # Upsample    
#    FCN10 = tf.layers.conv2d_transpose(inputs=FCN9_skip, 
#                                       filters=vgg_layer3_out.get_shape().as_list()[-1],
#                                       kernel_size=4, strides=(2, 2), 
#                                       padding='SAME', name="fcn10")   
#    
#    FCN10_skip = tf.add(FCN10, vgg_layer3_out, name="fcn10_plus_vgg_layer3")
#
#    # Upsample
#    # FCN Layer-11: FCN Layer-10 is upsampled 4 times to match dimensions with input 
#    # image size so we get the actual image back and depth is equal to number of classe
#    FCN11 = tf.layers.conv2d_transpose(inputs=FCN10_skip, filters=num_classes,
#                                      kernel_size=16, strides=(8, 8), 
#                                      padding='SAME', name="fcn11")  
#    
#    
#    return FCN11
#tests.test_layers(layers)
  
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # FCN Layer-8: The last fully connected layer of VGG16 is replaced by a 1x1 convolution.
    FCN8 = tf.layers.conv2d(inputs=vgg_layer7_out, 
                            filters=num_classes, 
                            kernel_size=[1, 1], name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so 
    # that we can add skip connection with 4th layer
    FCN9 = tf.layers.conv2d_transpose(inputs=FCN8, 
                                      filters=num_classes,
                                      kernel_size=4, strides=(2, 2), 
                                      padding='SAME', name="fcn9")
   
    # Convert layer4_out's depth to num_classes
    layer4_out2 = tf.layers.conv2d(inputs=vgg_layer4_out, 
                                   filters=num_classes, 
                                   kernel_size=[1, 1], name="layer4_out2")    
    # At each stage, the upsampling process is further refined by adding features 
    # from coarser but higher resolution feature maps from lower layers in VGG16.
    # skip connection (element-wise addition)
    FCN9_skip = tf.add(FCN9, layer4_out2, name="fcn9_plus_vgg_layer4")

    # Upsample    
    FCN10 = tf.layers.conv2d_transpose(inputs=FCN9_skip, 
                                       filters=num_classes,
                                       kernel_size=4, strides=(2, 2), 
                                       padding='SAME', name="fcn10")   
    
    # Convert layer4_out's depth to num_classes
    layer3_out2 = tf.layers.conv2d(inputs=vgg_layer3_out, 
                                   filters=num_classes, 
                                   kernel_size=[1, 1], name="layer3_out2")
    # skip connection
    FCN10_skip = tf.add(FCN10, layer3_out2, name="fcn10_plus_vgg_layer3")

    # Upsample
    # FCN Layer-11: FCN Layer-10 is upsampled 4 times to match dimensions with input 
    # image size so we get the actual image back and depth is equal to number of classe
    FCN11 = tf.layers.conv2d_transpose(inputs=FCN10_skip, 
                                       filters=num_classes,
                                       kernel_size=16, strides=(8, 8), 
                                       padding='SAME', name="fcn11")    
    
    return FCN11  
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits = tf.reshape(nn_last_layer, [-1, num_classes], name="fcn_logits")
    corr_labels = tf.reshape(correct_label, [-1, num_classes], name="fcn_labels")
    
    total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=corr_labels, logits=logits)
    cross_entropy_loss = tf.reduce_mean(total_loss, name="fcn_loss")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    train_op = optimizer.minimize(cross_entropy_loss, name="fcn_train_op")    
    
    return logits, train_op, cross_entropy_loss
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, 
             input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    lr = 0.0001
    dropout = 0.2
#    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    losses = []
    for i in range(epochs):
        # image, gt_label = next(get_batches_fn(batch_size))
        total_loss = 0.0
        batch = 0
        print('epoch: {}\n'.format(i))
        for image, gt_label in get_batches_fn(batch_size):
            batch += 1
            feed_dict = {input_image: image, correct_label: gt_label, learning_rate: lr, keep_prob: dropout} 
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            total_loss += loss
            print('--------- batch: ', batch)
        losses.append(total_loss)
        print('total loss: ', total_loss)
    
    print('losses: ', losses)
    #plt.plot(losses)    
    return losses
#tests.test_train_nn(train_nn)

#def build_model(sess):
#  
#    num_classes = 2
#    image_shape = (160, 576)
#    data_dir = './data'
#    vgg_path = os.path.join(data_dir, 'vgg')
#  
#    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
#    learning_rate = tf.placeholder(tf.float32)
#        
#    # input_image will be fed later in train_nn
#    input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
#                                               load_vgg(sess, vgg_path)
#
#    nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
#        
#    logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, 
#                                                        learning_rate, num_classes)
#    return logits
    
# taken from: https://github.com/karolmajek/CarND-Semantic-Segmentation/blob/master/main.py
#def gen_test_output_video(sess, logits, keep_prob, image_pl, video_file, image_shape):
#    """
#    Generate test output using the test images
#    :param sess: TF session
#    :param logits: TF Tensor for the logits
#    :param keep_prob: TF Placeholder for the dropout keep robability
#    :param image_pl: TF Placeholder for the image placeholder
#    :param image_shape: Tuple - Shape of image
#    :return: Output for for each test image
#    """
#    cap = cv2.VideoCapture(video_file)
#    counter=0
#    while True:
#        ret, frame = cap.read()
#        if frame is None:
#            break
#        image = scipy.misc.imresize(frame, image_shape)
#
#        im_softmax = sess.run(
#            [tf.nn.softmax(logits)],
#            {keep_prob: 1.0, image_pl: [image]})
#        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
#        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
#        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
#        mask_full = scipy.misc.imresize(mask, frame.shape)
#        mask_full = scipy.misc.toimage(mask_full, mode="RGBA")
#        mask = scipy.misc.toimage(mask, mode="RGBA")
#
#
#        street_im = scipy.misc.toimage(image)
#        street_im.paste(mask, box=None, mask=mask)
#
#        street_im_full = scipy.misc.toimage(frame)
#        street_im_full.paste(mask_full, box=None, mask=mask_full)
#
#        cv2.imwrite("4k-result/4k_image%08d.jpg"%counter,np.array(street_im_full))
#        counter=counter+1
#
#    # When everything done, release the capture
#    cap.release()
#    cv2.destroyAllWindows()
  
  
def run():   
    
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    model_path = runs_dir
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    
    epochs = 5 # 25
    batch_size = 5
    # learning_rate = tf.constant(0.001)
  
    tf.reset_default_graph()

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        
        # Build model: --------------------------------------------------------
        # TODO: Build NN using load_vgg, layers, and optimize function

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)
        
        # input_image will be fed later in train_nn
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
                                                   load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, 
                                                        learning_rate, num_classes)
        # ---------------------------------------------------------------------
        
        #_assert_tensor_shape(logits, [num_classes*image_shape[0]*image_shape[1], num_classes], 'Logits')
        
        sess.run(tf.global_variables_initializer())
      
        # To save and restore the model
        saver = tf.train.Saver()
        
        # Train: --------------------------------------------------------------
        # TODO: Train NN using the train_nn function
        print('Training...')
        losses = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                          input_image, correct_label, keep_prob, learning_rate)
        
        plt.plot(losses)
        '''
        ### Kiki:
        # https://stackoverflow.com/questions/40655010/how-to-test-my-neural-network-developed-tensorflow
    
        # Save it after training is done
        import time
        model_path = saver.save(sess, "./saved_models/model_" + str(time.time()) + ".ckpt")
    
    #   # To test (like in helper.py):
    #    tf_p = tf.nn.softmax(logits) # logits are the predictions
    #    [p] = sess.run([tf_p], feed_dict = {x : x_test, y : y_test})
    
    #   # or: load the saved the session before testing
    #    tf_x = tf.constant(X_test)  
    #    tf_p = tf.nn.softmax(neuralNetworkModel(tf_x)) 
    #
    #    with tf.Session() as sess:
    #        tf.initialize_all_variables().run()
    #        saver.restore(sess, model_path)
    #        p = tf_p.eval()
    
        ### ----------------------------------------------------------------------
        '''
        # Test and save: ------------------------------------------------------
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        print('Saving labeled test data...')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, 
                                      keep_prob, input_image)
    
        # Test and plot: ------------------------------------------------------
        sess.close()
#        from glob import glob
#        import scipy.misc
#        import numpy as np          
        
        sess = tf.Session()
        # Build the model but not train it
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, './saved_models/model.ckpt')
        
        data_folder = './data/data_road/testing'
        for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
            #image_file = glob(os.path.join(data_folder, 'image_2', '*.png'))[0]
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        
            im_softmax = sess.run([tf.nn.softmax(logits)],
                                  {keep_prob: 1.0, input_image: [image]})
        
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
        
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)
            # to save:
            # image_path = os.path.basename(image_file)
            # im_array_to_save = np.array(street_im)
            plt.imshow(street_im)
            plt.pause(0.1)

        
        # OPTIONAL: Apply the trained model to a video
        
       

if __name__ == '__main__':
    
    run()
    
    # Test the saved model:
    from glob import glob
    import scipy.misc
    import numpy as np
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    saver.restore(sess, './saved_models/model.ckpt')
    
    image_shape = (160, 576) 
    data_folder = './data/data_road/testing'
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        #image_file = glob(os.path.join(data_folder, 'image_2', '*.png'))[0]
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        # to save:
        # image_path = os.path.basename(image_file)
        # im_array_to_save = np.array(street_im)
        plt.imshow(street_im)
        plt.pause(0.1)

    
    
#tf_x = tf.constant(X_test)  
#tf_p = tf.nn.softmax(neuralNetworkModel(tf_x)) 
#model_path = './saved_models/model.ckpt'
#
#with tf.Session() as sess:
#   tf.initialize_all_variables().run()
#   saver.restore(sess, model_path)
#   p = tf_p.eval()
        
    
        
        
