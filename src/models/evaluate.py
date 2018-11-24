import numpy as np
import torch


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

  # compute accuracy
  with np.errstate(divide='ignore', invalid='ignore'):
    acc = np.divide(count.cpu().numpy(), total)

  outputs['accuracy'] = acc
  return outputs
