## Documentation

<h1>build_model</h1>

<p>This is a function that creates a model for image classification using a pre-trained model (specified by base_model - <code>Xception</code> from Keras Applications in particular) as a base model, with some additional inner layers and a final output layer. The model takes a tuple of input data input_shape (e.g. (150, 150, 3) for images with given resolution and 3 color channels) as input and returns a compiled model.</p>

<p>The pre-trained base model (base_model) is first set to be not trainable. Then, an input layer is defined using the input shape. The base model is then applied to the input data, but training is set to False so that the base model's weights are not updated during training.</p>

<p>The output of the base model is passed through a global average pooling layer, which reduces the spatial dimensions of the output by taking the average value over all spatial dimensions. This is followed by an inner dense layer with size_inner units and a ReLU activation function.</p>

<p>If include_dropout is set to True, a dropout layer with a dropout rate of droprate is applied to the output of the inner dense layer. Otherwise, the output of the inner dense layer is passed directly to the output layer. The output layer is a dense layer with a single unit and a sigmoid activation function, which will produce a binary output (either 0 or 1).</p>

<p>Finally, the model is compiled with an Adam optimizer (using the specified learning_rate) and a binary cross-entropy loss function. The model's summary is then printed and the compiled model is returned.</p>


<h1>checkpoint_weight</h1>


<p>The <code>checkpoint_weights</code> function is a utility function that sets up several callbacks for use during model training. The main purpose of the callbacks is to save the best model to a file during training and to stop the training if the validation accuracy does not improve after a certain number of epochs. These callbacks include:</p>
<ul>
  <li><code>ModelCheckpoint</code>: This callback saves the best model weights to a file, with a name that includes the epoch number and the validation accuracy.</li>
  <li><code>EarlyStopping</code>: This callback stops the training if the validation accuracy does not improve after a specified number of epochs (in this case, 3 epochs).</li>
  <li><code>TensorBoard</code>: This callback is used to visualize the training progress in TensorBoard, a tool for analyzing and debugging machine learning models.</li>
</ul>
<h3>Usage</h3>
<p>To use the <code>checkpoint_weights</code> function, you need to provide the following arguments:</p>
<ul>
  <li><code>model_name</code>: The name of the model. This name will be used to name the checkpoint files that are saved.</li>
  <li><code>checkpoint_dir</code>: The directory where the model checkpoint files will be saved.</li>
  <li><code>log_dir</code>: The directory where the TensorBoard logs will be saved.</li>
  <li><code>delete_files</code> (optional, default=True): A flag to indicate whether to delete existing checkpoints. If this flag is set to <code>True</code>, the function will delete all files in the <code>checkpoint_dir</code> directory that contain the <code>model_name</code> in their names. If this flag is not set or is set to <code>False</code>, the function will append an underscore and a number to the end of <code>model_name</code> if there are any files in the <code>checkpoint_dir</code> directory that contain <code>model_name</code> in their names.</li>
  <li><code>restore_from_checkpoint</code> (optional, default=False): A flag to indicate whether to restore the model weights from a checkpoint file. If this flag is set to <code>True</code>, the function will find the latest checkpoint file in the <code>checkpoint_dir</code> directory and load the model weights from it.</li>
  <li><code>callbacks</code> : a list of callbacks to use during training. If this parameter is not provided, the function will create the ModelCheckpoint, EarlyStopping, and TensorBoard callbacks.</li>
</ul>

<p>The function first checks if the <code>checkpoint_dir</code> and <code>log_dir</code> directories exist, and creates them if they do not. It then checks the value of the <code>restore_from_checkpoint</code> flag and, if it is set to True, searches for the latest checkpoint file in the <code>checkpoint_dir</code> directory with the same model name and restores the model weights from it.</p>
<p>The function then checks the value of the <code>delete_files</code> flag. If it is set to True, the function deletes all existing checkpoint files with the same model name in the <code>checkpoint_dir</code> directory. If <code>delete_files</code> is not set or is set to False, the function appends an underscore and a number to the end of <code>model_name</code> if there are any existing checkpoint files with the same model name in the <code>checkpoint_dir</code> directory. The number is equal to the number of such files in the directory.</p>
<p>The function then creates a <code>ModelCheckpoint</code> callback and specifies the <code>checkpoint_dir</code> directory and the <code>model_name</code> as the file name for the checkpoint file. The callback is set to save the best model. It is also set to monitor the validation accuracy and save the model when the validation accuracy improves.</p>
<p>The function also creates an <code>EarlyStopping</code> callback and specifies the number of epochs to wait before stopping the training if the validation accuracy does not improve. Finally, the function creates a <code>TensorBoard</code> callback and specifies the <code>log_dir</code> directory as the location for the <code>TensorBoard</code> logs.</p>

<p>The <code>checkpoint_weights</code> function returns a list containing the <code>ModelCheckpoint</code>, <code>EarlyStopping</code>, and <code>TensorBoard</code> callbacks, which you can pass to the <code>fit</code> function of your model to use during training.</p>
<p>Here is an example of how to use the <code>checkpoint_weights</code> function to set up the callbacks for model training:</p>
<pre><code>callbacks = checkpoint_weights(model_name='my_model', checkpoint_dir='checkpoints', log_dir='logs')

model.fit(train_generator,
	  epochs=100,
	  validation_data=validation_generator,
          callbacks=callbacks,
          steps_per_epoch=len(train_generator))
</code></pre>
<p>This will train the model using the provided training and validation data, and save the best model weights to a file in the <code>checkpoints</code> directory with a name that includes the epoch number and the validation accuracy. If the validation accuracy does not improve after 3 epochs, the training will be stopped and the best model will be restored. In addition, the training progress will be logged to the <code>logs</code> directory and can be visualized in TensorBoard.</p>


<h1>train</h1>

<p>The <code>train</code> function allows you to train and evaluate a model for multiple learning rates.</p> The <code>build_model</code> function is used to construct the model using the supplied learning rate, dropout rate, and other parameters. The model is then trained using the supplied training data and epochs, as well as the specified callbacks. The training history is saved and returned as a dictionary, with the hyperparameters (learning rate and dropout rate) as keys and the training history objects as values. In addition, the function returns the trained model with the highest validation accuracy.

It appears that the model is trained using a static graph defined by the train_step function. The train_step function performs a single training step, using a gradient tape to compute the gradients of the loss function with respect to the model's trainable variables and applying the gradients using the optimizer. The loss value resulting from the training step is returned by the function.

<h2>Parameters:</h2>
<ul>
  <li><code>rates</code>: a list of float values representing the learning rates to use for training the model.</li>
  <li><code>checkpoint_weights</code>: a function that creates a ModelCheckpoint callback for saving the model weights during training.</li>
  <li><code>epochs</code>: an integer representing the number of epochs to train the model.</li>
  <li><code>input_shape</code>: a tuple representing the shape of the input data.</li>
  <li><code>include_dropout</code>: a boolean indicating whether or not to include dropout layers in the model.</li>
</ul>
<h2>Returns:</h2>
<ul>
  <li><code>scores</code>: a dictionary containing the training history for each learning rate. The keys of the dictionary are the learning rates and the values are the training history objects returned by the <code>fit</code> method.</li>
  <li><code>model</code>: a trained <code>tf.keras.Model</code> with the best validation accuracy.</li>
</ul>
<h2>Example</h2>
<pre><code>scores, model = train(rates=[0.001, 0.01, 0.1],
                      callbacks=create_checkpoint_callback,
                      epochs=10,
                      input_shape=(128, 128, 3),
                      include_dropout=True)
</code></pre>

<h1>plot_metrics</h1>

<p>We define a function <code>plot_metrics</code> that takes in a history object, label, maximum number of epochs, and figure number, and plots the accuracy and loss for training and validation data on separate subplots of a single figure.</p>
<p>The <code>history</code> object is expected to have the keys <code>'accuracy'</code>, <code>'val_accuracy'</code>, <code>'loss'</code>, and <code>'val_loss'</code>, which represent the accuracy and loss for training and validation data, respectively. These are plotted using the <code>plot</code> function of the <code>matplotlib</code> library, with the <code>x</code>-axis representing the epoch number and the <code>y</code>-axis representing either the accuracy or loss.</p>
<p>The <code>label</code> parameter is a string that describes the hyperparameters used in the model, such as the learning rate and dropout rate. This label is used as the label for the plotted lines.</p>
<p>The <code>max_epochs</code> parameter is used to set the <code>x</code>-axis limits of the plot to ensure that all subplots have the same <code>x</code>-axis range.</p>
<p>The <code>fig_num</code> parameter specifies the figure number, which is used to save the figure to a file with a corresponding name.</p>
<p>Finally, the code sets the size of the figure to be twice the original size, and loops through the <code>scores</code> dictionary, calling the <code>plot</code> function for each item in the dictionary and passing in the corresponding history object, label, maximum number of epochs, and figure number. The resulting plots are then displayed using the <code>show</code> function of the <code>matplotlib</code> library.</p>



<h1>plot_image_extension_frequency</h1>

<p>This function plots the frequency of different image extensions in a given directory.</p>

<h2>Usage</h2>

<pre><code>plot_image_extension_frequency(path)
</code></pre>

<h2>Parameters</h2>

<ul>
  <li><code>path</code>: (required) The path to the directory to be scanned.</li>
</ul>

<h2>Example</h2>

<pre><code>plot_image_extension_frequency('./Images/train')
</code></pre>

<h2>Output</h2>

<p>A bar chart showing the frequency of each image extension in the given directory.</p>

<h2>Notes</h2>

<ul>
  <li>The function uses the <code>os.walk</code> function to iterate over the files and subfolders in the given <code>path</code>.</li>
  <li>The function extracts the extension of each file using the <code>os.path.splitext</code> function and adds it to a list called <code>extensions</code>.</li>
  <li>The function uses the <code>collections.Counter</code> function to count the frequency of each extension in the <code>extensions</code> list.</li>
  <li>The function sets a list of colors for the different extensions and uses a <code>for</code> loop to iterate over the unique extensions.</li>
  <li>For each extension, the function plots a bar chart using <code>matplotlib</code>'s <code>bar</code> function. The function uses the <code>extension</code> as the x-axis label and the frequency of the extension as the y-axis label. The color of the bar is chosen from the <code>colors</code> list using the index <code>i</code> modulo the length of the <code>colors</code> list.</li>
  <li>The function adds labels to the x-axis, y-axis, and the chart title using <code>matplotlib</code>'s <code>xlabel</code>, <code>ylabel</code>, and <code>title</code> functions.</li>
  <li>The function displays the plot using <code>matplotlib</code>'s <code>show</code> function.</li>
</ul>

<h1>disp_samples</h1>

<ul>
  <li>It gets the list of class names from the train_generator object, which is a generator that returns batches of images and labels for training a machine learning model. The <code>class_indices</code> attribute of <code>train_generator</code> is a dictionary that maps class names to integer labels.</li>
  <li>It defines a function <code>disp_images</code> that takes in a generator and a list of class names as arguments. The generator is expected to return batches of images and labels.</li>
  <li>Inside the <code>disp_images</code> function, the code creates a figure with a size of 10x10 and retrieves the first batch of images and labels from the generator using <code>next(iter(generator))</code>.</li>
  <li>The labels are converted to integers and the first 9 images in the batch are plotted in a 3x3 grid using a loop. For each image, the code displays the image using <code>plt.imshow</code> and sets the title of the subplot to be the corresponding class name using <code>plt.title</code>. The axis labels are turned off using <code>plt.axis("off")</code>.</li>
  <li>Finally, the <code>disp_images</code> function is called with the <code>validation_generator</code> and <code>class_names</code> as arguments, which plots the images using the class names.</li>
</ul>


<h1>show_augmented_images</h1>

  <p>
   We define a function called <code>show_augmented_images</code> that takes in three arguments: <code>directory</code>, <code>img_input</code>, and <code>class_name</code>.
  </p>
  <p>
    The function first constructs a path to the specified <code>class_name</code> directory by joining <code>directory</code> and <code>class_name</code> using <code>os.path.join</code>. It then uses a list comprehension to get a list of file names within the <code>class_name</code> directory.
  </p>
  <p>
    Next, the function randomly selects one of the file names from the list using <code>random.choice</code> and loads the corresponding image using the <code>image.load_img</code> function from the <code>keras.preprocessing.image</code> module. It then converts the image to a numpy array using the <code>image.img_to_array</code> function.
  </p>
  <p>
    The numpy array is then reshaped to be compatible with the <code>train_datagen.flow</code> method, which generates augmented versions of the input image. The reshaped array is passed as an argument to <code>train_datagen.flow</code>, along with a batch size of 1.
  </p>
  <p>
    The function then iterates through the generated augmented images using a for loop and displays each image using the <code>plt.imshow</code> function from the <code>matplotlib</code> module. The loop is set to break after showing two images. Finally, the function calls <code>plt.show</code> to display all of the generated images.
  </p>