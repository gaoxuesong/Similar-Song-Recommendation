import tensorflow as tf
import numpy as np

def cross_entropy_loss(inputs, true_w):
    print ("Entered Here")
    
    mat = tf.matmul(true_w,tf.transpose(inputs))
    A = tf.diag_part(mat)

    mat = tf.exp(mat)
    B = tf.reduce_sum(mat,1)
    B = tf.log(B)
        

    #split0, split1 = tf.split(split_dim = 1,num_split = 2, value = inputs,name = 'split')
    #split0, split1 = tf.split(0,2, inputs)
    # split_inputs = tf.split(inputs, 128, 0)
    # split_labels = tf.split(true_w, 128, 0)

    # v_c = tf.split(inputs, 128, 0)
    # u_o = tf.split(true_w, 128, 0)

    # A = tf.zeros([1,], tf.float32)
    # B = tf.zeros([1,], tf.float32)
    # for i in range(128):
    #     prod = tf.matmul(v_c[i],tf.transpose(u_o[i]))
    #     prod = tf.reshape(prod,[-1])
    #     #print prod
    #     if i == 0:
    #          A = tf.add(A,prod)
    #     else:
    #          A = tf.concat([A,prod],0)
        # sum_per_context = tf.zeros([1,1], tf.float32)
        # for j in range(128):
        #     prod = tf.matmul(u_o[j], tf.transpose(v_c[i]))
        #     prod_exp = tf.exp(prod)
        #     sum_per_context = tf.add(sum_per_context, prod_exp)            
        # sum_per_context_log = tf.log(sum_per_context)
        # sum_per_context_log = tf.reshape(sum_per_context_log,[-1])
        # if i == 0:
        #      B = tf.add(B, sum_per_context_log)
        # else:
        #      B = tf.concat([B,sum_per_context_log],0)
        

    # with tf.Session() as sess:
    #     res = sess.run(A)
    #     print(A)
    
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    return tf.subtract(B, A)
    #return A,B,mat
    
def nce_loss(inputs, weights, biases, labels, sample):
    true_w = tf.nn.embedding_lookup(weights, labels)
    true_w = tf.reshape(true_w, [-1, 128])
    sample_vec = tf.nn.embedding_lookup(weights, sample)
    ones = tf.ones([128, 64], tf.float32)

    mat = tf.matmul(inputs, tf.transpose(true_w))
    mat = tf.diag_part(mat)
    A = tf.log(tf.sigmoid(mat))

    mat1 = tf.matmul(inputs, tf.transpose(sample_vec))
    mat1 = tf.subtract(ones, tf.sigmoid(mat1))
    mat1 = tf.log(mat1)
    B = tf.reduce_sum(mat1, 1)

    return -tf.add(A, B)
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

