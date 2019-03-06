# gnmt  
google neural network tranlation comment  
google gnmt 代码阅读和注解  
相互调用关系  

nmt.py
	main()->run_main(train_fn, inference_fn)
	其中,train_fn指train.py中的train()
	run_main中，根据参数：flags.inference_input_file决定是走train逻辑还是走infer逻辑
	如果是infer，则取最新的checkpoint，执行inference_fn
	如果是train，则走train.py的train()


train.py
	train()
	解析参数，attention选择不同的model_creator
	利用model helper 和 model_creator 构建train_model, eval_model, infer_model


		model_helper.py
			create_train_model
			根据输入的参数以及训练数据生成文件迭代器
			选择要生成的模型：一般我们选择AttentionModel，AttentionModel继承了Model类，重写了_build_decoder_cell方法，


			这里主要是添加了attention_mechanism，使用tf.contrib.seq2seq.AttentionWrapper生生成了lstm cell，这里使用了model_helper.py里的create_rnn_cell()


			create_rnn_cell()->_cell_list(){
				根据num_layers，调用最终的single_cell_fn生成单个cell，具体的single_cell的方法为model_helper.py里的_single_cell


				single_cell里，
					(1)根据不同的参数，可以生成不同的cell，这里包括BasicLSTMCell, GRUCell, LayerNormBasicLSTMCell,
					(2)使用DropoutWarapper为每一层添加dropout
					(3)根据参数添加residualwrapper
					(4)添加device wrapper
			}


			Model里的主要方法：
				_build_encoder()
				_build_bidirectional_rnn()
				_build_decoder_cell
			Model继承自BaseModel,BaseModel的构造方法里：
				init方法是真正的模型构造器，
				主要包括以下几个步骤：
					初始化，
					embedding, 
					projection(output_layer Dense layer)
					build train graph: build encoder, build decoder, compute loss (其中build encoder/decoder都是各个子类做具体的实现)
    
train.py构建完成开始train loop
