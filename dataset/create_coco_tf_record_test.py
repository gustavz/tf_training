#
"""Convert raw COCO dataset to TFRecord for object_detection.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      train_category_index=None,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    train_category_index: Second Category Index with reduced Classes
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  # find really used category ids
  if train_category_index:
      used_category_ids = []
      for id in category_index:
          for idt in train_category_index:
              if category_index[id]['name'] in train_category_index[idt].values():
                  used_category_ids.append(id)

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  num_annotations_skipped = 0
  for object_annotations in annotations_list:
    if object_annotations['category_id'] in used_category_ids:

        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
          num_annotations_skipped += 1
          continue
        if x + width > image_width or y + height > image_height:
          num_annotations_skipped += 1
          continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])

        # Find Train ID matching the name
        if train_category_index:
            name = category_index[object_annotations['category_id']]['name']
            for id in train_category_index:
                if train_category_index[id]['name'] == name:
                    category_id = id
                    break
        else:
            category_id = int(object_annotations['category_id'])

        category_ids.append(category_id)
        category_names.append(train_category_index[category_id]['name'].encode('utf8'))
        area.append(object_annotations['area'])

        if include_masks:
          run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                              image_height, image_width)
          binary_mask = mask.decode(run_len_encoding)
          if not object_annotations['iscrowd']:
            binary_mask = np.amax(binary_mask, axis=2)
          pil_image = PIL.Image.fromarray(binary_mask)
          output_io = io.BytesIO()
          pil_image.save(output_io, format='PNG')
          encoded_mask_png.append(output_io.getvalue())
  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/label':
          dataset_util.int64_list_feature(category_ids),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped



def _create_tf_record_from_coco_annotations(
    annotations_file, image_dir, output_path, include_masks,label_path=None):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  """
  with tf.gfile.GFile(annotations_file, 'r') as fid:
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']

    # Workaround Category Index to train on reduced classes
    train_category_index = None
    if label_path:
        label_map = label_map_util.load_labelmap(label_path)
        categories=label_map_util.convert_label_map_to_categories(label_map,90,use_display_name=True)
        train_category_index = label_map_util.create_category_index(categories)

    category_index = label_map_util.create_category_index(groundtruth_data['categories'])

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      tf.logging.info(
          'Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    tf.logging.info('%d images are missing annotations.',
                    missing_annotation_count)

    tf.logging.info('writing to output path: %s', output_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    total_num_annotations_skipped = 0
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]
      _, tf_example, num_annotations_skipped = create_tf_example(
          image, annotations_list, image_dir, category_index, train_category_index, include_masks)
      total_num_annotations_skipped += num_annotations_skipped
      writer.write(tf_example.SerializeToString())
    writer.close()
    tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)


if __name__ == '__main__':
  CWD = os.getcwd()
  train_image_dir=CWD+"/coco/train2017"
  val_image_dir=CWD+"/coco/val2017"
  test_image_dir=CWD+"/coco/test2017"
  val_annotations_file=CWD+"/coco/annotations/instances_val2017.json"
  train_annotations_file=CWD+"/coco/annotations/instances_train2017.json"
  testdev_annotations_file=CWD+"/coco/annotations/image_info_test-dev2017.json"
  include_masks=True

  output_name="red_"
  label_path=CWD+"/red_coco_label_map.pbtxt"
  output_dir=CWD

  if not tf.gfile.IsDirectory(output_dir):
    tf.gfile.MakeDirs(output_dir)
  val_output_path = os.path.join(output_dir, '{}val.record'.format(output_name))
  train_output_path = os.path.join(output_dir, '{}train.record'.format(output_name))
  testdev_output_path = os.path.join(output_dir, '{}testdev.record'.format(output_name))


  _create_tf_record_from_coco_annotations(
      train_annotations_file,
      train_image_dir,
      train_output_path,
      include_masks,
      label_path)

  _create_tf_record_from_coco_annotations(
      val_annotations_file,
      val_image_dir,
      val_output_path,
      include_masks,
      label_path)

  _create_tf_record_from_coco_annotations(
      testdev_annotations_file,
      test_image_dir,
      testdev_output_path,
      include_masks,
      label_path)
