快速开始
===========================

以下说明假设您已经安装了PyTorch-NPU环境，有关环境安装，请参考 :doc:`./install`

一般来说，要在代码中使用NPU进行训练推理，需要做以下更改：

#. 导入torch_npu扩展包 ``import torch_npu``
#. 将模型，以及模型输入上传到NPU上

:: 

    device= torch.device("npu")
    model = model.to(device)
    input = input.to(device)

下面的实例演示了如何使用NPU进行训练和推理任务：

1. 单卡训练
-----------------------
以下代码使用了cifar10数据集在NPU上训练模型（截取自 `PyTorch tutorials <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_），请关注高亮的内容。

.. code-block:: python
    :linenos:
    :emphasize-lines: 20,21,23,24,25,84,85,109,110,146,147,171,172

    """
    Training an image classifier
    ----------------------------

    We will do the following steps in order:

    1. Load and normalize the CIFAR10 training and test datasets using
    ``torchvision``
    1. Define a Convolutional Neural Network
    2. Define a loss function
    3. Train the network on the training data
    4. Test the network on the test data

    5. Load and normalize CIFAR10
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Using ``torchvision``, it’s extremely easy to load CIFAR10.
    """
    import torch
    # 引入torch-npu包
    import torch_npu

    # 定义device
    device = torch.device('npu:0' if torch.npu.is_available() else 'cpu')
    print(device)

    import torchvision
    import torchvision.transforms as transforms

    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ########################################################################
    # 2. Define a Convolutional Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Copy the neural network from the Neural Networks section before and modify it to
    # take 3-channel images (instead of 1-channel images as it was defined).
    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    # 将模型加载到NPU上
    net.to(device)

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # 将input数据发送到NPU上
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    ########################################################################
    # 5. Test the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # We have trained the network for 2 passes over the training dataset.
    # But we need to check if the network has learnt anything at all.
    #
    # We will check this by predicting the class label that the neural network
    # outputs, and checking it against the ground-truth. If the prediction is
    # correct, we add the sample to the list of correct predictions.
    #
    # Let us look at how the network performs on the whole dataset.
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # 将input数据发送到NPU上
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    ########################################################################
    # That looks way better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            # 将input数据发送到NPU上
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

2. 使用DeepSpeed多卡并行训练
---------------------------
以下代码使用了cifar10数据集，使用DeepSpeed训练模型在多张NPU卡上进行模型训练（来自 `DeepSpeed Examples <https://github.com/microsoft/DeepSpeedExamples/blob/master/training/cifar/cifar10_deepspeed.py>`_），自DeepSpeed v0.12.6之后，代码无需任何修改，即可自动检测NPU并进行训练。

.. rli:: https://raw.githubusercontent.com/microsoft/DeepSpeedExamples/master/training/cifar/cifar10_deepspeed.py
    :language: python
    :linenos:


3. 使用Transforms进行模型微调
---------------------------------
以下代码使用了Transforms对LLM进行微调（来自 `transforms examples <https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py>`_），自transforms xxx版本以及accelerator 0.21.0版本以后，代码无需任何修改，即可自动检测NPU并进行。

.. rli:: https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/language-modeling/run_clm.py
    :language: python
    :linenos:


.. code-block:: shell
    :linenos:

    python run_clm.py \
        --model_name_or_path openai-community/gpt2 \
        --train_file path_to_train_file \
        --validation_file path_to_validation_file \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --do_eval \
        --output_dir /tmp/test-clm

4. 使用Diffusers进行模型微调
---------------------------------
以下代码使用了Diffusers对文生图模型进行微调（来自 `diffusers examples <https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py>`_），自diffusers v0.27.0版本以后，代码无需任何修改，即可自动检测NPU并进行。


.. rli:: https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image.py
    :language: python
    :linenos:


.. code-block:: shell
    :linenos:

    export MODEL_NAME="CompVis/stable-diffusion-v1-4"
    export DATASET_NAME="lambdalabs/naruto-blip-captions"

    accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --use_ema \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=15000 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="sd-pokemon-model"