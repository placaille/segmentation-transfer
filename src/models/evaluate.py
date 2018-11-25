import numpy as np
import torch
from utils import logit_to_img


def evaluate(model, data_iterator, device):

  # evaluation
  model.eval()
  count = 0
  total = 0
  outputs = dict()
  for images, labels in data_iterator:
    logits = model(images.to(device))

    preds = torch.argmax(logits, dim=1)

    count += torch.sum(labels.to(device) == preds)
    total += labels.nelement()

  images, labels = iter(data_iterator).next()
  labels = labels.to('cpu').numpy()
  logits = model(images.to(device))

  outputs['images'] = images.to('cpu').numpy()
  outputs['targets'] = logit_to_img(labels).transpose(0, 3, 1, 2)

  classes = logits.to('cpu').detach().numpy().argmax(1)
  outputs['segmented'] = logit_to_img(classes).transpose(0, 3, 1, 2)

  # compute accuracy
  with np.errstate(divide='ignore', invalid='ignore'):
    acc = np.divide(count.cpu().numpy(), total)

  outputs['accuracy'] = acc
  return outputs
