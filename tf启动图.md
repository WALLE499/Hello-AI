# Hello-AI
Tensorflow中启动图：
介绍 TensorFlow 中启动图： tf.Session()，tf.InteractivesSession()，tf.train.Supervisor().managed_session()  用法的区别：
（1）tf.Session()构造阶段完成后, 才能启动图. 启动图的第一步是创建一个 Session 对象, 如果无任何创建参数, 会话构造器将启动默认图.
代码示例：
    #coding:utf-8  
      
    import tensorflow as tf  
      
    matrix1 = tf.constant([[3., 3.]])  
    matrix2 = tf.constant([[2.], [2.]])  
      
    preduct = tf.matmul(matrix1, matrix2)  
    # 使用 "with" 代码块来自动完成关闭动作.  
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())  
        print sess.run(preduct)  
  （2）tf.InteractivesSession()为了便于使用诸如 IPython之类的 Python 交互环境, 可以使用InteractiveSession 代替 Session 类, 使用 Tensor.eval()和 Operation.run()方法代替Session.run(). 这样可以避免使用一个变量来持有会话。
具体代码：
    #coding:utf-8  
      
    import tensorflow as tf  
      
    matrix1 = tf.constant([[3., 3.]])  
    matrix2 = tf.constant([[2.], [2.]])  
      
    preduct = tf.matmul(matrix1, matrix2)  
      
    sess_ = tf.InteractiveSession()  
      
    tf.global_variables_initializer().run()  
    print preduct.eval()  
      
    sess_.close()  
  （3）tf.train.Supervisor().managed_session()
    与上面两种启动图相比较来说，Supervisor() 帮助我们处理一些事情：
    (a) 自动去 checkpoint 加载数据或者初始化数据
   （b) 自动有一个 Saver ，可以用来保存 checkpoint
    eg: sv.saver.save(sess, save_path)
    (c) 有一个 summary_computed 用来保存 Summary
    因此我们可以省略了以下内容：（a）手动初始化或者从 checkpoint 中加载数据 （b）不需要创建 Saver 类， 使用 sv 内部的就可以
    （c）不需要创建 Summary_Writer()  
    具体代码：
        import tensorflow as tf  
      
    matrix1 = tf.constant([[3., 3.]])  
    matrix2 = tf.constant([[2.], [2.]])  
      
    preduct = tf.matmul(matrix1, matrix2)  
      
    sv = tf.train.Supervisor(logdir=None, init_op=tf.global_variables_initializer())  
      
    with sv.managed_session() as sess:  
        print sess.run(preduct)  
另举例子：  
    import tensorflow as tf  
      
    a = tf.Variable(1)  
    b = tf.Variable(2)  
    c = tf.add(a, b)  
      
    update = tf.assign(a, c)  
      
    init = tf.global_variables_initializer()  
      
    sv = tf.train.Supervisor(logdir="./tmp/", init_op=init)  
    saver = sv.saver  
    with sv.managed_session() as sess:  
        for i in range(1000):  
            update_ = sess.run(update)  
            #print("11111", update)  
      
            if i % 100 == 0:  
                sv.saver.save(sess, "./tmp/", global_step=i)  
