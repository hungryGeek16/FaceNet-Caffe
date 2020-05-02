from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import tensorflow as tf

with tf.Graph().as_default() as graph: # Set default graph as graph
           with tf.Session() as sess:
                # Load the graph in graph_def
                # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
                with gfile.FastGFile("facenet_frozen.pb",'rb') as f:

                                graph_def = tf.GraphDef()
                                graph_def.ParseFromString(f.read())
                                sess.graph.as_default()

                                # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

                                tf.import_graph_def(
                                graph_def,
                                input_map=None,
                                return_elements=None,
                                name="",
                                op_dict=None,
                                producer_op_list=None
                                )
                                
                                #initialize_all_variables
                                tf.global_variables_initializer()

                                wts = [n for n in graph.get_operations() if n.type=='Const']    
                                constant_values = {}
                                for constant_op in wts:
                                    constant_values[constant_op.name] = sess.run(constant_op.outputs[0])
weights_dict = {}
for i in constant_values.keys():
    if i.count('Conv2d') > 0 or i.count('Bottleneck') > 0 and i.count('Reshape') == 0:
        weights_dict[i] = constant_values[i]
np.save('final_weights.npy',weights_dict)                                    

