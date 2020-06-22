# Pytorch Coding Problems Records

This repository records all  problems I have encountered when coding with [Pytorch](https://pytorch.org/).

1. What is the difference between `mm()` and `matmul()`? They are the same!!
2. For the learnable parameters, remember to add `requires_grad=True`. But if a parameter is a function of another parameter which is grad trackable, then pytorch will automatically trace the gradient for this parameter. Another way is to use `Parameter()` which is stated [here](https://pytorch.org/docs/stable/nn.html?highlight=parameter#torch.nn.Parameter). The example is [here](https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py).
3. Before the backpropagation step, remember to zero all the gradient buffer using `optimizer.zero_grad()` at the **beginning** of each iteration
4. For MNIST dataset, although the images are grayscale, we also need to use tuples (0.5, ) (0.5,) to define means and stds.
5. For the operations which don't affect the gradient, use with `with torch.no_grad():`
6. Two ways to get the parameter of a specific layer: (1) `net.parameters()[index]` (2)`net.layer_name.weight(or bias).data`
7. `torch.Tensor(2,3)` constructs a 2-by-3 tensor, `torch.Tensor((2,3))` constructs a tensor([2., 3.])
8. `b = a.view(,)` if we change one of the elements in a, the corresponding element in b will change.
9. Note that for index, `[1:4]` does not include the fourth element. 
10. Tensor and Numpy: if some operations are not supported by the Tensor, then convert it to Numpy, afterwards convert it back. Note that if the type of them are the same, they share the same storing position.
11. The size of input for CNN: `[Batch,  Channels,  Height,  Width]`
12. Image transforms in Dataloader: 1. note the order of the numpy (H,W,C) and tensor(C,H,W) is different, can use `.transpose((2,0,1))` to convert 2. for gray images, if it is two dimentional in numpy, first use `array[:,:,newaxis] `to extend it to 3d. 3. some operation in `torchvision.transforms` are for PIL images. To convert from tensor to PILImage: `import torchvision.transforms.functional as F  mask = F.to_pil_image(mask)`    4. note if we want to use `plt` to show the image, first it needs to be numpy array, and for gray image, we need to first squeenze it, but don't need to transpose it
10. Use `nn.LogSoftmax` with `NLLLoss` or use `nn.Softmax` with `CrossEntropyLoss`.(if binary, use `nn.Sigmoid` and `BCELoss`)
11. When customizing the Dataset, the return value must contain **tensor, list, number or dict**, cannot return other objects, otherwise, when we use enumerate to get the batch, it will be wrong.
12. Difference between `torch.Tensor` and` torch.tensor`: torch.Tensor includes all functions in torch.tensor in addition with torch.empty.
13. `torch.ones(5)` is different from `torch.ones(1, 5)`. the former construct a row vector, while the latter constructs a 1-by-5 matrix.
14. `torch.tensor` will always copy the original tensor to a new tensor, so that they don't share memory. if you want to convert from numpy to tensor or tensor to numpy and want them to share the memory, use `.numpy()` and` .from_numpy()`.
15. Note that when construct a new tensor, pytorch will not store the gradient for it. If we want to keep it, set `.require_grad = True`. 
16. After each `.backward()`, the gradient will be accumulated, so if we want to re-compute it, then remove the history first.
17. For the input to `torch.nn module`, it must contain mini-batch, even there is only one sample. Use `unsqeeze(0)` to get the correct dim.
18. To compute ROC_AUC score using sklearn, Please Note that y_score is the predictive probability of **POSITIVE** classes!!
20. Don't apply Dropout to the output layer!
21. When you pad the batched inputs of various length for RNN and do the prediction label for the last hidden states, please notice that for sequences of different lengths, we need to get the output from different time step!! Otherwise, for some sequences of shorter length, it will get the output from the padded time step!!
22. When `.view(-1)` a tensor, please check whether its `requires_grad == True`
25. [This blog](https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66) states a question that in Bidirectional RNN, when use the output from the last step, the reversed RNN only see the last step, which can lose a lot of representation ability. Through experiment, it found that output and hidden states are different: `hn` from Bidirecitional RNN contains both last step of normal and reversed RNN (after both of them see the whole sequence), `output` only contains the hidden state by feeding the input of this time step into both forward and backward RNN cell at this time (until this time, forward RNN has seen $(w_{1}, \dots, w_{t})$ and backward RNN has seen $(w_{N}, \ldots, w_{t})$). 