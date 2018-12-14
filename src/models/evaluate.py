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


def evaluate_transfer(model_seg, model_transfer, data_iterator, device):

  # evaluation
  model_seg.eval()
  model_transfer.eval()
  batch_count = 0
  outputs = dict()
  with torch.no_grad():
    for images in data_iterator:
      batch_count += 1

      # get transformed image in simulator domain
      transformed = model_transfer(images.to(device))

      # get segmentation
      seg_logits = model_seg(transformed)
      seg_preds = torch.argmax(seg_logits, dim=1).cpu()

      if batch_count == 1:
        outputs['images'] = images.cpu().numpy()
        outputs['transformed'] = transformed.cpu().numpy()
        outputs['segmented'] = logit_to_img(seg_preds.cpu().numpy()).transpose(0, 3, 1, 2)
        break

  del seg_logits
  del seg_preds
  torch.cuda.empty_cache()

  return outputs


def evaluate_segtransfer(model_seg, sim_data_iterator, real_data_iterator, device):

  # evaluation
  model_seg.eval()
  outputs = dict()
  with torch.no_grad():
    real = iter(real_data_iterator).next()
    # get segmentation
    seg_logits = model_seg(real.to(device))
    seg_preds = torch.argmax(seg_logits, dim=1).cpu()

    outputs['real'] = real.cpu().numpy()
    outputs['real_segmented'] = logit_to_img(seg_preds.cpu().numpy()).transpose(0, 3, 1, 2)

    sim = iter(sim_data_iterator).next()[0]
    # get segmentation
    seg_logits = model_seg(sim.to(device))
    seg_preds = torch.argmax(seg_logits, dim=1).cpu()

    outputs['sim'] = real.cpu().numpy()
    outputs['sim_segmented'] = logit_to_img(seg_preds.cpu().numpy()).transpose(0, 3, 1, 2)

  torch.cuda.empty_cache()
  return outputs
