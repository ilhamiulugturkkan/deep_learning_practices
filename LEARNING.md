# 09-06-2026
I created a venv and installed some libraries (torch, torchvision, numpy, matplotlib, wandb). Then I decided to use Google Colab when I need GPU in the future as I used it in the past and familiar with its usage and plans. Basically I set up the environment that I'll work.

ILH-154: First I watched 3Blue1Brown NN series' first 4 videos which was a good reminder for me as I've seen that class a year ago but I realized that I remember every fundamental thing which felt really good. Then I read the PyTorch official docs on Tensors and Autograd which was pretty cool. Because tensors are basically numpy data structures but GPU usable it's amazing. Also autograd saves our ass to compute every parameters' gradients which is very cool. Altough I couldn't understand the implementation of the autogra, docs were pretty explaining. To understand the working logic of the autograd I started to watch Karpathy, The spelled-out intro to neural networks and backpropagation: building micrograd (first 54 mins) and it is REALLY good. My perspective to NN's and all the hard work changed instantly cuz I weirdly always thought calculating all that gradients etc. would be a huge pain in the ass for custom NNs but it logic is pretty simple and genius. So far everything is going pretty good.

ILH-155: I played a bit with common tensor operations such as reshape, view, sum etc. The difference between reshape and view was interesting. I learnt the term contiguous. I think it means storing the tensor in memory as we show them sequentially. Let's say there's a tensor like Q=[[1,2,3],[4,5,6]] storing this tensor in memory like [1,2,3,4,5,6] is a contiguous state. So using view() causes no issue. However if we Q.T and perturbate the shape of the Q, it becomes non-contiguous and using view throws an error. On the other hand reshape can be used for both states but we cannot be sure whether it copied or just viewed the data.

## "not doable for me just now" followed by "Update: done," is the whole roadmap
ILH-156: I exploded my brain for this task. However, I think it's not doable for me just now. Because although I know the conecpt theoretically, I am not familiar with the implementation and it is really hard to sense what to code and with which attributes etc. for now. Instead of it, I am examining the code and will finish the video.

# 09-07-2026
ILH-156 Update: I've finish implementing my version of micrograd. Ofc it's not that clean etc. but it was a great way to understand how real torch works. The importance of zero grad, the oop way of networks are pretty cool to implement and see at the same time. I think this exarcise is a bit difficult as the 'first DL exercise' and a bir exhausting to do but at the end of the day I am pretty satisfied with the results and all the thing my brain digest.

## A bad loss diagnostic doesn't just fail to help, it actively covers for other bugs.
ILH-157: This part seemed pretty easy but it took a lot of time tbh. I learnt a lot because I really started to think in a DL mindset instead of theoretical ML mindset. Instead of thinking these processes directly on a graph, I must start to think everything as neurons. When I think every variable as a neuron I start to code easily. Also, I learnt a couple of important details. One is updating the parameters with with torch.no_grad():. I didn't read any documentation about it but its reason is obvious. We are tracking updated parameters grads and for that, torch add every operation taken on a parameter to its dag (just like we did with Value on minigrad). As parameter updates are seperate from the process we must update them under that. Also, resetting the grads after updating is pretty important as well as grads are updated with += or -= signs.
Update reasons: Took the derivative wrong. Broadcasting caused every point to have the same noise during the sample init, I changed it. Also, loss functions were wrong (I forgot to add **2 etc.). Mistakes are trivial and made because of 2.54 am tiredness. Nothing to worry about.

# 09-08-2026
ILH-158: Logistic regression task. After linear regression it was pretty easy with autograd. My only problem is comprehending the size of elements that I am working with or my confidency on Torch operations which will be improved as I continue to write code like this exercise. There are not a lot things to mention here. I forgot the cross entropy function and checked it online. I plotted everything with AI (that's what I do). Also, linear regression wouldn't work with this data cuz this data distrbiution is not linear so linear regression model would not be able to cover this dataset.

From feedbacks: Storing loss in linreg_autograd was a huge mistake as I was storing a torch entitity. Instead of it from now on I'll use x.item() to store additional things or ops that are nothing to do with model etc.

# 09-09-2026
ILH-160: Read CS231n NNs Part 1 and I kenw all that stuff. I only didn't know that ReLU is one of the best activation functions to use. For the second reading (same site NN part 2) I actually spent a good amount of time. I relearnt some stuff and honestly, didn't dive deep on some stuff (some calculations etc.). I got the fundamental message though. I didn't know about batch normalization (I still don't cuz I didn't read the paper yet.) and hot to set initial weights. I think it was a great introduction and I'll learn more on these topics as I continue to code stuff. Other than that, I knew everything except dropout which is pretty cool. Ofc I still don't know them in detail but I know the general concepts such as how to use, why to use, and how it works.

# 09-11-2026
I've finished watching Karpathy Zero to Hero video 2, makemore part 1: the spelled-out intro to language modeling. While doing that, I tried to implement all things that Andrej would implement beforehand and checked my work's quality. That took a pretty long time and was pretty hard at first. But I feel like getting more and more comfortable with PyTorch day by day. As I have the fundamentals (loss funcs, algorithms, etc.) the only struggle is to remember everything and somehow implementing it with PyTorch.
Also there were some tricks in this exercise. We first trained the biagram based on the counts and then we created a NN with randomized weights initally. When we optimized our weights (by minimizing the loss function) our NN converged exactly to our first implemented count model. Which is interesting because the first one was pretty easy to implement. But as the dataset grows and our goal differs (biagram is extremely easy to count) NNs are much more effective. Also smoothing the N array with +1 was equivalent to L2 regularization of our loss function which was pretty cool to see.

# 09-12-2026
I've finished implementing MLP on MNIST with pure numpy. It was a challenging task for me especially when it came to gradient calculations. First of all I normalized and standardized the training data. There I've made a mistake while normalizing as I did smh like data/np.sum(data) whereas I should've divide the data with the max possible value (which is 255 for this dataset). Then I initialized the weights and feed the MLP with inputs. Writing the forward() was pretty straightforward and easy. The only thing additional was subtracting a rows max value before feeding it into exp() func. Otherwise a lot of overflow errors were happening as we feed a lot of neurons data to a much smaller layer and values can go massive pretty quick. But while calculating the gradients there were some tricks. For example, normally calulcating gradients one by one is the classic method but especially for softmax, directly calculating dL/dZ2 was much more easy. Lastly the shape rule. I think this rule is one of the most important things I've learn in this section. Naturally all the grads must be in the same shape with their data and it is a great way to understand if we are calulcating a gradient correctly or not.

# 09-13-2026 
I've finish implementing the mlp_torch script yesterday but was too tired to write a learning.md section so here it is. First, I've read two pages of the PyTorch docs whcih were Dataloaders, NN, and Optimization Loop. These things were not that hard to understand as they are implementing the same things that we did before, they're just easier to code. For example instead of shuffling the dataset, dividing it to batches and iterating them and also doing this again for each epoch we just use dataloaders. Or instead of writing forward and backward functions, we just use NN and Optimizers. These libraries has some cool functions to use and easily implement but it is obvious that I can't start to code with them just by reading the docs. I know that I gotta code much more, and say 'Shit, this is much slower that I except' and search for a function from docs, and implement that. I think this will be the case for most of the functions etc. 

Some things to emphasize:
- I used 'device' for the first time and the training loop was really slow. Because I had a print function in there and the process was switching between gpu and cpu again and again. After I erased that print, the runtime was almost 3 times faster. 
- While using NNs there's no need to flatten data by hand nn has a good function for that but we gotta be careful about the shapes of the data.
- Again I normalized and standardized the data myself. I don't know if there's a funtion for that but it's pretty easy so no problem.


About Building makemore part2:
Embeddings: It's a way to represent anything as vectors. Doing it for chars, words, etc. is pretty impressive. I actually knew this term before but now by thinking the nn structure, it makes MUCH MORE sense.
PS: I started to guess the next lines of code before Andrej writes it which makes me feel pretty good.

# 09-14-2026
Training set is to train parameters, val set is for hyper-parameters, and test set is to evaluate.

Notes on the most important video of Andrej's series (prt 3 of building makemore):
- Initialization is wrong in bunch of aspects and we can tell it by looking at hte huge difference of loss between the epoc 0 and epoch 10000 (27.9->2.85). We can see this wwhen we plot the loss function by epochs. It looks like a hockey stick.

How to calculate the expected loss value at first?
for our case the prob of each class initially is 1/27 so the expected loss is -torch.tensor(1/27.0).log()= 3.3.


1- With extreme logit values, it is very likely to have a high loss. To minimize the loss initially, we gotta prevent extreme logits and try to make them as close as possible to each other. To make it possible:
- Init biases as 0 or very small values as we dont need a bias initially.
- With default init the last layer produces large logits, so the softmax is very peaked on random classes and the initial loss is way above ln(27) ≈ 3.3. The first few hundred steps of training are then wasted just shrinking those logits down, which is the hockey-stick shape at the start of the loss curve. So setting the last layer's weights to very small values is a good idea.
- 

2- Most of the activations (values of tanh(smh)) are 1.0 or -1.0 or really close to these values initially. We can see it by plotting a histogram of h values. This is extremely bad as it forces the gradient of itself to be 0 cuz when we derivate tanh we can see that values close to 1 or -1's slopes are very close to 0. This is dangerous because under the hood we are killing the self.grad by assigning it ~0 and therefore killing every grad value which are dependent to it. This can lead to never activating neuron problem. So we gotta solve this *'sturated tanh'* problem. The root of the problem is tanh(root). If root values are too extreme then we get this problem. So if root = inputs @ w + b, just make w = w * 0.1 and b = b * 0 or 0.01 etc.

AGAIN: To detect this problem we gotta plot h's (activation values) to see extreme values (1 and -1 for tanh case or 0 and 1 for sigmoid or negative values for ReLU ) and also their inputs to see the reason of these extreme values.

BUT how do we come up with numbers like *0.01 or *0.02 to scale down b's or w's? One thing to notice is when we do x @ w with x and w are normally dist. output's mean doesnt change but its std increases by sqrt(fan_in). When we do w * 5 the std increases much more so it's obvious that we gotta scale down w. ANSWER: we gotta / fan_in**0.5 or smh like that. Luckly pytorch has a good function to init weights wrt this problem called nn.init.kaiming_normal_(tensor, mode='fan_in', nonLinearity='activation func'). Andrej basically divides his weights to the square root of fenin.


Hoewever, setting the init weights optimal is not important as it was before thanks to inovations.

## Batch Normalization
It makes it possible to train deep nns much more reliably. It works. Standardizin tnah(hpreact)'s hpreact. It is a differentiable operation so it's alright. So we calculate the mean and the std of the minibatch and standardize the batch. With this hpreact is unit gaussian which prevents extreme values and helps activation functions to prevent the saturation and dead neurons. 

However, we want hprevact to be unit gaussian only in initialization. Because during the training, we want our values to be approximate to the ground truth values. So we should introduce scale and shift. So we have to scale standardized inputs by some gain and then add a value bias to shift them. We train gain and bias by backpropagating to offset and scale the unit dist.
So the bottom line is: Standardize and scale before the activation function to get rid of most of the problem that activation functions can cause. We have to keep this attitude while predicting as well.
This method is very helpful because as the networks gets deeper and deeper it will be very hard to adjust all the weights etc. initially. With batch normalization we are solving this problem inside the network in a generalized way and instead of updating maybe millions of weights we are adding additional parameters to optimize. So with batch normalization we don't need to brainstorm to adjust initial weights.
*So we just append a batch normalization layer to linear layers to control the scales of the activations (layers that has multiplications).* It also has a regularizing affect which is a side effect.
So after we're done with training, we might feed the network with a single input and we cannot estimate the mean and std of it. Therefore furing the training, it is a common choice to estimate the general means and std and then using those values while feeding the network with val or test data.
Some  details: 
-It is a wise choice to add a small fixed value to add std as it can be zero and lead to a div by zero error.
- Also adding bias to a layer comes just before a normalization layer is a waste of resource as we subtract it during the normalization. Batch normalization's bias does that job instead of prev layer's bias.

Last Note: This layer is an extremely dangerous layer to play with and causes a lot of bugs. Try to avoid it as much as possible and I believe that much more reliable and optimized techniques are introduced in 3 years (the video is post 3 years ago)

Why we dont want saturation values to shrink to zero? Isnt it good?

We can use plots to keep track of saturation rates, the parameters' grad values, the outputs of tanh's etc. to detect the problems of our network and add additional layers or might want to initialize the weights a little different.
With update:data ratio plot we can see if our lr is too large or small by comparing the lines with the constant -3 line.
Bottom line is plotting some attributes of the parameters is a very good way to understand what went wrong and what to do to cure the network.

# 09-16-2026

For the last 2 days I've worked on src/train which can be executed with config files. It's a general function to train models and save their parameters. It was kind of a challenge for me but I think I used all the things I've learnt for the last 2 weeks and it was a good review for me. Btw I am trying to understand the traditional ways to implement these kind of scripts therefore I am getting advice from various sources such as docs, internet, and AI. But I use these AFTER I code. ALSO, early stopping was a new term for me. It stops the training after loss stops decreasing for 'patience' times. I set patience to 3 in config.

3.2: For this exercise I dropped the early stopping and tried out 6 different configurations and their wandb logs. I logged train_loss, val_loss, and val_accuracy.

Baseline: Nothing to mention. It's the baseline with lr=0.01.

lr*100: With this model's graphs I relaized that 0.01 is a way too small learning rate for this model. Because both losses and accuracies seemed much more better than the baseline. The weird thing was after an epoch, val_loss increased but accuracy continued to increase. When I thought about it and checked it online I saw that this was not a weird thing at all. Because those are differnt metrics, one is binary and one is a lot of multiplication etc.

lr/100: As expected this performed really really bad because our baseline already had a small lr. Maybe for future, it can be a good idea to set lr as 1 and update it to 0.01 close to convergence.

shuffle_false: This was a interesting graph. All of the graphs were very smooth. With shuffling, loss could have a spike or a pit for an epoch or two but with shuffle=False those movements disappeared. Which is logical as the same batches are loaded to the trainer in the same order.

batch_size=2: One of the worst models. All noisy graphs and it takes AGES to converge.

batch_size=4096: Performs the worst for the first epochs but it would perform good if we would increase the epoch number. It is not the best batch size for MNIST dataset as we have a relatively small number of training samples. Because batches helps us to train the model with random samples and unique batches help pur models to converge much better. However when we take the batch 1/4 of the set all the batches have much bigger probability to be much similar. This is one of the most important facts why big batches (relative to the dataset size) doesnt perfom that well. But obviously much less training is the #1 reason of bad performance. AI:"big batches take fewer, more accurate steps per epoch, so at a fixed epoch budget they've simply had less opportunity to update; the standard fix is scaling LR up with batch size, not just running longer." (pff basically the same thing...)


3.3:
Dropout: This technique hurts the performance of the model significantly. Our model has a small amount of neurons and we apllied dropout to the most crowded layer (first layer) with 0.2 rate. That meant a significant amount of the parameters were disabled and the capacity of the model dropped. This was an expected result for me as our model is a baby.

Weight_decay: Weight decay didn't do a lot as we didn't have any overfitting problem and a regularization was not needed at all. With regularization, some of the weights tried to pull to zero even though they were extremely needed and useful. As I said before, even though MNIST dataset is relatively small our model is smaller and dumber therefore it cannot even overfit. So extra regularization hurts the performance.

CosineAnnealingLR: This is an empirically effective technique. Andrej also mentioned it in his video. It increases the performance by starting fast (as init weights are mostly need big updates) and acting careful as the model's weights become more 'fragile'. As said, our model performed its best performance with this technique.

# 09-17-2026
Histograms:
Weight hists layer: We want to them to start near 0 and expand and then stabilize. 
Gradient: We want the dist mean to be close to 0 and stable not extreme std.
Update:Weight Ratio: We want them close to -3.

Throughout the hists, the most text-book hists were lr-0.01's. However we didn't scored the best performance with it. For a small dataset and a model forgiving as MLP, it is pretty casual. Because large lr's can actually find a better minima than small ones. However from the graphs, we understand that lr=0.01 is the most reliable lr and we would want to use it for complex problems and it would probably perform better. Hopefully we'll try this in the future.

## Adam
Momentum: It basically considers both current and previous movement while updating the weight. It is useful to think the model's loss in a graph. It is dangerous to move based on the current slope only as there can be some noise etc. but considering past too gives us an idea where the loss was heading to. Of course finding a good beta (how much past actions to consider 1/(1-beta)) is critical. We want our model to consider past but also consider the current slope too. SOURCES: it specifically cancels oscillation and amplifies consistent motion

Adam: It's the mixed solution of momentum and RMSprop. We utilize them both and add an additional  (1-b1) to momentum like v = b1 * prev + 1-b1 * current.We apply bias correction to both v and s (divide by 1-beta^t) because both accumulators start at zero, which biases them low for the first several steps — without correction, early updates would be artificially small. Bias correction compensates for exactly this, boosting early updates back to their true scale; as t grows, 1-beta^t approaches 1 and the correction fades out on its own. While momentum prevents oscillation, RMSprop decreases the effect of highly moving grads and stabilizes the grads. 

# 09-18-2026

## CNN
Assumption: input type is image. Therefore we can add some properties and make forward process more efficient and reduce learnable parameter count. Other than that it is nearly the same as regular neural networks that I dealt with for the last 2 weeks. Assigning a neuron to every single input especially for images is pretty waseful. Images generally have larger number of inputs as they have vertical and horizontal pixel numbers * RGB values (height, width, depth). Therefore we can take advantage of knowing the input array type and its fundamentals.

Three main types of layers: Convolutional, Pooling, and Fully-Connected. We'll stack these layers. While convolutional and fully-connected layers perform transformations that are a function of not only the activations (values after the activation function/layer) in the input volume, but also of the parameters; RELU and Pool layers implement a fixed function. So the parameters in convolutional and fuilly-connected layers will be trained. Each layer accepts an input 3D volume and transforms it to an output 3D volume through a differentiable func.

### Convolutional Layer:
MOst heavy computations are done by this guy. Here, we have a set of learnable FILTERS. Each filter has a a width and height, and extends through the full depth of the input volume. So, if the input volume has a depth 3 a filter's shape may be (h,w,3). During the forward pass we slide each filter across the input volume  and compute DOT PRODUCTS between the entries of the filter and the input at any position. As we slide a filter, we produce a 2d activation map. Then we stack all teh filters and get a 3d volume. So we are connecting each neuron to only a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the receptive field of the neuron (equivalently this is the filter size).

So let's say we now fitler size etc. How many neurons are there in the output volume or how they are arranged? They depend on some hyperparameters such as DEPTH, STRIDE, and ZERO-PADDING.

DEPTH: Depth of the output volume is a hyperparameter. It corresponds to the filter count that we have each learning to look something different in the input. Some filters may be learning edges and some may be learning colors etc. We will call the neurons looking to the same region of the input as DEPTH COLUMN.

STRIDE: The amount of how many pixels we slide the filter each step. If stride is 1, we slide the filter one-by-one. If it's 2, we jump 2 pixels etc.

ZERO-PADDING: Sometimes it is efficient to pad the input volume with zeros like a frame around the border. The size of it is a hyperparameter. As we can be seen on figures etc. in Google or smh, the edges of the image can be learnt only one time while other pixels are learnt by filters multiple times. Therefore can be a good implementation to pad the input and therefore 'scan' the edges by multiple neurons (filter 'scans' it multiple times). Also a common useage: In general, setting zero padding to be P=(F−1)/2 when the stride is S=1 ensures that the input volume and output volume will have the same size spatially.

We can compute the output spatial size with given input size (W), filter size (F), stide (S) and zero-padding (P) with: (W - F + 2P)/(S) + 1

PARAMETER SHARING: We constrain the neurons in each depth slice to use the same weights and bias. So regardless of the input size let's say we have a 5x5 filter. Our learnable parameters size is 5 * 5 * 3 (weights of each pixel the filter can scan)+1(bias)=76. If we have depth 5 then the learnable parameter number is 76 * 5 = 380. Notice that if all neurons in a single depth slice are using the same weight vector, then the forward pass of the CONV layer can in each depth slice be computed as a convolution of the neuron’s weights with the input volume (Hence the name: Convolutional Layer). This is why it is common to refer to the sets of weights as a filter (or a kernel), that is convolved with the input. SO all the neurons in a depth slice shares the same parameters (weights and bias) and when we think about a filter (say 5x5) and an image (say 27x27x3) we know that our filters can get info from 5x5x3 pixels and it has a bias so the learnable parameter count estimation is pretty easy. This sharing technique is very effective when we think like if a filter learns some feature in x,y it should be able to learn the same thing in x2,y2. But it doesnt work that well if the image has a specific centered structure (a portrait of a face). In this case we should relax the constraint and call the layer a locally-connected layer(?).

Calculation (READ CAREFUL)
1-The local regions in the input image are stretched out INTO COLUMNS in an operation commonly called im2col. For example, if the input is [227x227x3] and it is to be convolved with 11x11x3 filters at stride 4, then we would take [11x11x3] blocks of pixels in the input and stretch each block into a column vector of size 11* 11 * 3 = 363. Iterating this process in the input at stride of 4 gives (227-11)/4+1 = 55 locations along both width and height, leading to an output matrix X_col of im2col of size [363 x 3025], where every column is a stretched out receptive field and there are 55*55 = 3025 of them in total. Note that since the receptive fields overlap, every number in the input volume may be duplicated in multiple distinct columns.
2- The weights of the CONV layer are similarly stretched out INTO ROWS. For example, if there are 96 filters of size [11x11x3] this would give a matrix W_row of size [96 x 363].
3- The result of a convolution is now equivalent to performing one large matrix multiply np.dot(W_row, X_col), which evaluates the dot product between every filter and every receptive field location. In our example, the output of this operation would be [96 x 3025], giving the output of the dot product of each filter at each location.
4- The result must finally be reshaped back to its proper output dimension [55x55x96].

### Pooling Layer
It is used to reduce the spatial size of the representation to reduce the number of learnable parameters (also controls overfitting). Pooling layer operates independently on each depth slice and reduces the size by using MAX funvction. 

It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with F=3,S=2 (also called overlapping pooling), and more commonly F=2,S=2. Pooling sizes with larger receptive fields are too destructive.

General pooling. In addition to max pooling, the pooling units can also perform other functions, such as average pooling or even L2-norm pooling. Average pooling was often used historically but has recently fallen out of favor compared to the max pooling operation, which has been shown to work better in practice.

It has a decreasing popularity as people start to use larger stride values instead of pooling layers.  Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs) too.

Some advices:
Prefer a stack of small filter CONV to one large receptive field CONV layer. 

Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.

“don’t be a hero”: Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch.

Second, if we suppose that all the volumes have C channels, then it can be seen that the single 7x7 CONV layer would contain C×(7×7×C)=49C2 parameters, while the three 3x3 CONV layers would only contain 3×(C×(3×3×C))=27C2 parameters

 # 09-19-2026
 Cont'
 Common sizes: 
 input must be divisable to 2
 filter size must be small (commonly 3x3, at most 5x5) using stride 1
 Padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. That is, when F=3, then using P=1 will retain the original size of the input. When F=5, P=2. For a general F, it can be seen that P=(F−1)/2 preserves the input size.

To sum up, we would like to leave the job of downsizing to POOL layers and would like conv layer to preserve the input size. It is very common to use F=2, S=2 POOL layers. Using F>3 is very uncommong as we lose significant amount of info.

## First coding exercise errk4/conv_numpy.py
im2col is OP! I am proud of what I implemented by just looking at LEARNING.md! I just got some help from Claude when I increased the image_count and I'm glad I did cuz I wouldn't solve that. I think adding image_count was a little extra so I don't worry that much. It was a fun exercise. I also feel pretty good as I can easily understand the shape of everything and do matmuls or reshape (With axis=) operations with much less error.

# 09-20-2026
I added CNN model and CIFAR10 dataloader to the src/ and then trained 6 models. Baseline, model with batch norm after every conv, model that is trained with augmented data, training with weight decay, and training with cosine LR schedule and lastly, all of them together.

Base model achieved ~70 accuracy after 30 epochs. It has 3 conv,relu,pool layers and a fc layer at the end.

Batch norm (normalizes activation so that not activation explosion happens even if weight become too large during training) and augmentation improved the model performance really well. I already knew that both of them were used for generalization and to prevent overfitting. They both achieved ~77 accuracy in 30 epochs which is pretty awsome. From there, I understood that generalization in CNNs are crucial.

Weight decay and cosLR achieved slightly better performance than the baseline with ~72 accuracy. They were pretty ineffective for MLP in fact they decreased the accuracy of the model back then. weight decay only helps when there's overfitting to counteract; my MLP had too little capacity to overfit, so weight decay was pure harm, while the CNN has enough capacity that overfitting is real and weight decay earns its keep. I can see that as the epcoh and learnable parameter counts increase their affect will be much more observable.

Lastly I used all of them and trained the CNN for 30 epochs. The model achieved ~81 accuracy and I suspect that the epoch was the bottleneck ther because in the graphs I can see that val loss is still dropping and accuracy is still increasing. I had 2 techniques to prevent overfitting and 2 techniques that are effective as the epoch and the parameter count goes up. Therefore I think if I trained the model much more those techniques would pay much more.

# 09-21-2026
ResNet: ResNet main idea is to solve degradation problem which is the case when we make nn's deeper but training loss increases. Which seems pretty odd because if we were to add identity mapping layers to a model it wouldn't perform worse. Therefore, these guys make it much easy to converge some layers to identity mappings (kinda like if necessary) by doing F(x) = H(x) - x. This way a model layer can easily converge to a identity mapping (the authors hypothesize that very close to an optimal mapping in many cases). Also it is much more easier for model to converge to an optimal way as it must push the output F(x) to zero. This matches what's seen in experiments: plain deep networks struggle to find good values on their own.

PyTorch: We can load pretrained models with PyTorch like model = resnet50(weights=ResNet50_Weights.DEFAULT) but we gotta be careful to preprocess the data before training/evaluating the model. We can observe the transformations by doing preprocess = weights.transforms().

Also we can freeze parameters by setting their require_grad to False and replace the head of the model by doing model.fc = 'our desired head'. Since the newly replaced head require_grad is True by default, replacing the head after freezing all the parameters in ResNet is a common way of transfer leraning (feature extraction. used when dataset is small and similar to pretrained model's training data.). That way we can use a pre-trained model and tune it sometime to utilize it for our purpose. (If we were doing fine-tuning than we would unfreeze all the backbone and train everything.)

So far I've tried to understand the logic of the ResNet, and examined some examples that loads pretrained ResNet models.

# 09-22-2026

I've imported the pre-trained ResNet50 weights played with it using CIFAR10.

First I froze all the parameters and updated it the pre-trained model's head with nn.Linear(2048, 10) so that I could train the model for CIFAR10. Then I imported the CIFAR10 data and transformed it to fit to ResNet layers. That fit included resizing the shape of the image to 224x224. 10 epochs training took ~30 mins and I achieved maximum of 80% accuracy. I tried to increase the accuracy by adding more layers to the FC but it didn't work out which meant the bottleneck isn't the FC. The parameters in the backbone need some updates specific to CIFAR10. 
Then I tried to exclude reshaping from the transforms to keep the accuracy and shorten the training time but accuracy dropped significantly. I am not sure what does ResNet do when it recieves smaller input so I have no comments on that.

Secondly, I unfroze everything and replace FC again. I fine-tuned model for 10 epochs with seperate LR's for backbone and head. This time training took much more time (~70 mins) but accuracy rised significantly to ~97% which is amazing.

Lastly, I've saved the best model's weights.

# 09-23-2026

I redid my dog breed classifier project's resnet part again. I got a 91.4% accuracy score which beat my prev SOTA. I think it's because of the way I've implemented the model. Previously I was using the same LR value for noth the backbone and head parameters which was a mistake because of the fact that backbone params are pretrained with images already. I also printed the Confusin Matrix and saw that the class 1 adn 22 are the problematic ones. They were confused a lot previously as well. Well both american pitbull terrier and staffordshire bull terrier are scary and it is hard to tell which one is which for me as well. So I am not that surprised. 

## Week 6: RNN & LSTM 

RNNs passes hte previous information to the nn. They are networks with loops in them.

They inherit the info but how much info? Sometimes the needed info for the next output can be pretty close to us (maybe 3 or 4 words) but sometimes the gap is pretty large and we need a large history of outputs to generate an accurate output. As the gap grows RNN become pretty useless in practice.

LSTMs (Long short term memory) solve this problem. They are a special kind of RNN and capable of learning long-term dependencies. Remembering information for long periods of time is practically their default behavior.

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. (like h(t) <- tanh(concat(h(t-1), x(t))))

On the other hand, LSTMs have a relatively more complex structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

### The core idea behind LSTMs

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram (see an image in web if you don't remember it). All the operations are done to decide how much info to keep of the prev cell state, what to add additionally.

The first step in LSTM is the 'forget gate layer', here we multiply the prev cell state with the output of sigmoid(W[h(T-1),x(t)] + B). Sigmoid gives a value between 0 and 1. If the output is 1 it means 'teh prev info is so important keep it' and if it's 0 we just say 'forget it'.

The next steps can be understood by looking at hte great visualizations https://colah.github.io/posts/2015-08-Understanding-LSTMs/ provided. It is pretty hard to explain it without the vis. Definetely read that blog again if you forget anything about LSTMs.

If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over programs.

even if your data is not in form of sequences, you can still formulate and train powerful models that learn to process it sequentially. You’re learning stateful programs that process your fixed-sized data.

RNNs are neural networks and everything works monotonically better (if done right) if you put on your deep learning hat and start stacking models up like pancakes. In other words we have two separate RNNs: One RNN is receiving the input vectors and the second RNN is receiving the output of the first RNN as its input. Except neither of these RNNs know or care - it’s all just vectors coming in and going out, and some gradients flowing through each module during backpropagation.

# 09-24-2026

Off-day

# 09-25-2026

Off-day

# 09-26-2025

Off-day

# 09-27-2026

Started to implement RNN from scratch. It's a bit hard because I know how RNN's work but I really struggle to visualize the code structure in my mind. I implemented the RNNCell without a problem but when dataloaders etc. are included to the equation, I started to have a tough time. But no worries, I am on it.

I spent some time on RNNScratch and I think I did a lot. First, I finally understood the logic of dataloaders because I implemented one myself. Then I continued step-by-step and implemented the RNNCell which is the smallest component of the RNN structure that I am building. Then I implemented the RNN scratch which takes data from the dataloader, embeds the chars and processes it with RNNCell and lastly outputs a vocab_size vector. I had some difficulties with understanding the general structure and some shape manipulations. I think misunderstanding the general structure was because I had 3 days off and forgot most of the RNN implmentation details. However shape problems are pretty bad as I need to be comfortable with torch shapes by now.

# 09-28-2026

I've finished the training code. Also, I added a sampler as well. Then I fed the model with Monte Cristo and generated sample text wit hlength 500. Temperature matter a lot with the results. With temp 0.5 I got very good structured results but with temp 1.5 I got some weird words such as V02αnBt.

Of course the purpose of this task was not to just implemnet smh and sample from it. I learnt from my mistakes such as:

-> I tried to sample by feeding all the generated chars but my model was trained to get at most 64 of the history. So I started to feed the model with outputs[:, -64:]. Which improved the model's performance.

-> I've disgard all the redundant batches and adjusted my dataloader's getitem function to stride as seq_length. This improved the training time a lot and training loss didn't increased.

-> From my experience, I set all the weights woth randn and multiplied them with 0.01. While training I didn't have the exploiding grad problem even without clipping the gradients.

Lastly I couldn't understand what clip_grad_norm_ was returning so I asked Claude. I think grad_norm is a great way to check how's model doing 'roughly' . It doesn't give the exact answer but if something is awfully wrong we can easily see it from grad_norm. However we shouldn't rely on it too much as it can be deceiving as well. I would like to write my understanding of grad_norm but I think Claude gave me a pretty good summary, so here it is:

Every parameter (W_xh, W_hh, b_h, embeddings, output layer — thousands of individual
numbers) gets its own gradient after backward(). clip_grad_norm_ flattens ALL of them
across the whole model into one giant vector and takes its L2 norm (sqrt of sum of
squares) — one number summarizing "how big is the total update about to be, across
every weight in the model."

If total_norm > max_norm, every gradient gets rescaled by (max_norm / total_norm),
same direction, smaller step. Calling it with max_norm=inf clips nothing — it's a
read-only probe to just observe the norm.

What to read from it: huge + spiky → exploding risk. Near-zero + flat → vanishing,
barely learning. My RNN (seq_len=64, lr=0.1) stayed stable at ~0.5–0.78 the whole
run — no explosion, no clipping needed.

**Caveat (the important part):** a healthy aggregate norm does NOT rule out
vanishing along the time axis. It's a sum over every parameter's gradient, including
all 64 timesteps' contributions to W_hh. Late-timestep contributions (closer to the
loss, pass through the W_hh product fewer times) can dominate the sum and hide
early-timestep contributions that are near zero (passed through the product ~63
times). The aggregate can't tell "everything's medium" apart from "half are huge,
half are ~0." Confirming vanishing for real means comparing gradient magnitude at
individual timesteps, not just the total.

# 09-30-2026

Today I confidently implemented LSTM from scratch by checking RNNScratch code and of course architecture of LSTM. It was a good experience again. I forgot to put weights inside nn.Parameter and had a huge heart attack. Then I fixed it and trained the RNNScratch, LSTMScratch, and nn.LSTM with the same hyperparameters and same training data. Then I had another heart attack. Because RNN was taking much less time to train (as it has much less parameters and operations) and it was over performing LSTMScratch. It was a HUGE disappointment for me as I was really hoping to get amazing results with LSTM (even more amazing than the result I got with RNN). 

Then I switched my optimizer to Adam and got that amazingER result. Today's lesson was 'Never use SGD when your architecture has a lot of things to tune AND some past information matters more than other past information for predicting what comes next, and that's exactly what gates are for (learning what to keep vs. discard). Because SGD has a fixed lr and it updates all parameters in a fixed way. However ADAM is the best. It combines RMSProp and Momentum and does some cool things internally (god knows what but it rocks). Also using RNN or LSTM doesn't matter too much when we are guessing the next char as it predictable from a short term of memory (like a couple of chars), however when we do complex stuff that requires a much more large window of past LSTM will be much more useful and worth the effort to train (hopefully),

# 10-01-2026

Lastly, I had a quiz kinda thing and failed to answer why there's a vanish/explode problem with RNNs. It's because to calculate the gradient of a previous timestamps weights, we have to apply the chain rule and estimate it by mutliplying every next weights gradient. If those gradients are close to 0, we basically end up in 0 after a while (depends on determined sequence length as well), if next gradients are more than 1, it explodes and become a very large value. That's why we want to keep gradients arund zero and use torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0). END OF WEEK 6

# 10-02-2026 

I did a class assignment and it was EASY. It would probably kill me if I didn't study this roadmap. However, I easily implemented different classes inheriting nn.Module, set the dataloaders etc. and easily trained models. It's nothing to do with this roadmap but it was cool to see the impact.

# 10-05-2026

## Transformer

### Self-Attention

The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in text case the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

The second step in calculating self-attention is to calculate a score. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position. The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.

The third and fourth steps are to divide the scores by 8 (or smh else), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.

The fifth step is to multiply each value vector by the softmax score. The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

The sixth step is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

The resulting vector is one we can send along to the feed-forward neural network. In the actual implementation, however, this calculation is done in matrix form for faster processing.

The paper further refined the self-attention layer by adding a mechanism called “multi-headed” attention. If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices. But we need a way to condense these eight down into a single matrix because feed forward network expects a single vector. So we concat the matrices then multiply them by an additional weights matrix WO.

### Positional Encoding

One thing that’s missing from the model as we have described it so far is a way to account for the order of the words in the input sequence.

To address this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention. The formula for positional encoding is described in the paper (section 3.5).

One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a layer-normalization step.

### Decoder Side

The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence.

The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to -inf) before the softmax step in the self-attention calculation.

The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.

Decoders repeat the process until a special symbol is reached indicating the transformer decoder has completed its output.

Then we pass produced vector to a fully connected linear layer (vectr_size, vocab_size) to obtain logits vector and then we apply softmax.