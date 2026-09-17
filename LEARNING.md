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