import numpy as np
import torch
from torch.autograd import Variable
import logging

logger = logging.getLogger(__file__)


def get_odin_score(args,model, inputs, temperature, epsilon, criterion,flag_discrete=False,device="cuda"):

    if flag_discrete:
        inputs = inputs[0]
        if args.model=="transformer":
            input_embeds = model.transformer.embeddings(inputs)
        else:
            input_embeds = model.embedding(inputs)

        input_embeds = Variable(input_embeds, requires_grad=True).to(device)
        hidden,_,_ = model(inputs, inp_embeds=input_embeds)
        outputs = model.classifier(hidden.to(model.classifier.weight.dtype))
    else:
        inputs = Variable(inputs, requires_grad=True)
        outputs, _ = model(inputs)

    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    nnOutputs = outputs.data.cpu()
    # Using temperature scaling
    outputs = outputs / temperature

    # Calculating the perturbation we need to add, that is,f
    # the sign of gradient of cross entropy loss w.r.t. input
    maxIndexTemp = np.argmax(nnOutputs, axis=-1)
    labels = Variable(torch.LongTensor(maxIndexTemp)).to(device)
    loss = criterion(outputs, labels)
    # inputs.retain_grad()
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}

    if flag_discrete:
        gradient = torch.ge(input_embeds.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        tempInputs = torch.add(input_embeds.data, -epsilon, gradient)

        with torch.no_grad():
            hidden,_,_ = model(inputs, inp_embeds=tempInputs)
            hidden = hidden.to(model.classifier.weight.dtype)
            outputs = model.classifier(hidden)

    else:
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        with torch.no_grad():
            tempInputs = torch.add(inputs.data, -epsilon, gradient)
            outputs, _ = model(Variable(tempInputs))
    #
    # outputs = model(Variable(tempInputs))
    outputs = outputs / temperature
    nnOutputs = torch.max(torch.softmax(outputs, -1), -1)[0]
    # Calculating the confidence after adding perturbations
    nnOutputs = nnOutputs.cpu()

    return nnOutputs

def sample_odin_estimator(model, criterion, valid_loader, epsilons: list, temper=1000, flag_discrete=False,device="cuda"):

    model.to(device)
    if not isinstance(epsilons, list):
        raise ValueError

    odin_score = {}

    for epsilon in epsilons:
        odin_score[epsilon] = []
        for inputs, _ in valid_loader:
            if flag_discrete:
                inputs = inputs[0]
                input_embeds = model.transformer.embeddings(inputs)
                input_embeds = Variable(input_embeds, requires_grad=True).to(device)
                out = model(inputs, inp_embeds=input_embeds)
                outputs = model.classifier(out[0])
            else:
                inputs = Variable(inputs, requires_grad=True).to(device)
                # inputs = inputs.cuda()
                inputs.retain_grad()
                outputs, _ = model(inputs)

            # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
            nnOutputs = outputs.data.cpu()
            # Using temperature scaling
            outputs = outputs / temper

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            maxIndexTemp = torch.argmax(nnOutputs, dim=-1)
            labels = Variable(maxIndexTemp).to(device)
            loss = criterion(outputs, labels)

            loss.backward()
            # print([w.grad for w in model.parameters()][0])

            # Normalizing the gradient to binary in {0, 1}
            # Normalizing the gradient to the same space of image
            # Adding small perturbations to images

            if flag_discrete:
                gradient = torch.ge(input_embeds.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2
                tempInputs = torch.add(input_embeds.data, -epsilon, gradient)
                out = model(inputs, inp_embeds=tempInputs)
                outputs = model.classifier(out[0])
            else:
                gradient = torch.ge(inputs.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2
                tempInputs = torch.add(inputs.data, -epsilon, gradient)
                outputs, _ = model(Variable(tempInputs))
            # outputs = model(Variable(tempInputs))
            outputs = outputs / temper
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu()
            nnOutputs = torch.max(torch.softmax(nnOutputs, -1), -1)[0]
            nnOutputs = nnOutputs.numpy()
            odin_score[epsilon].extend(nnOutputs)
    score_func = [(k, np.sum(v)) for k, v in odin_score.items()]

    print(score_func)
    score_func = sorted(score_func, key=lambda x: x[-1])[-1]

    print("Optimized Epsilon {0}".format(score_func[0]))
    return score_func[0]