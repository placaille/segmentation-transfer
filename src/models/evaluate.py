import numpy as np
import torch
from utils import logit_to_img


def evaluate(model, partition_provider, device):

  # evaluation
  model.eval()
  count = 0
  total = 0
  batch_count = 0
  outputs = dict()
  with torch.no_grad():
      for partition in partition_provider.valid_partition_iterator:
        valid_data_iterator = partition_provider.get_valid_iterator(partition)
        for images, labels in valid_data_iterator:
          batch_count += 1
          logits = model(images.to(device))

          preds = torch.argmax(logits, dim=1).cpu()

          count += torch.sum(labels == preds)
          total += labels.nelement()

          if batch_count == 1:
            outputs['images'] = images.cpu().numpy()
            outputs['targets'] = logit_to_img(labels).transpose(0, 3, 1, 2)
            outputs['segmented'] = logit_to_img(preds.cpu().numpy()).transpose(0, 3, 1, 2)

  del logits
  torch.cuda.empty_cache()

  # compute accuracy
  with np.errstate(divide='ignore', invalid='ignore'):
    acc = np.divide(count.cpu().numpy(), total)

  outputs['accuracy'] = acc
  return outputs
