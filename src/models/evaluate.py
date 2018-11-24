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
  logits = model(images)
  outputs['image'] = images
  outputs['target'] = logit_to_img(labels)
  outputs['segmented'] = logit_to_img(logits)

  # compute accuracy
  with np.errstate(divide='ignore', invalid='ignore'):
    acc = np.divide(count.cpu().numpy(), total)

  outputs['accuracy'] = acc
  return outputs
