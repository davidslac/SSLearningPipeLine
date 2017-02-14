from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import json
import matplotlib.pyplot as plt
from . import util

plt.ion()


class SSLearnPipeline(object):
  def __init__(self, outputdir, output_prefix, max_boxes_in_one_image, total_to_label=250):
    self.outputdir = outputdir
    assert os.path.exists(outputdir)
    self.output_prefix = output_prefix
    self.labeled_dir = os.path.join(outputdir, 'labeled')
    if not os.path.exists(self.labeled_dir):
      os.mkdir(self.labeled_dir)
    self.jpegs_to_label = os.path.join(outputdir, 'jpegs_to_label')
    if not os.path.exists(self.jpegs_to_label):
      os.mkdir(self.jpegs_to_label)
    self.total_to_label = total_to_label
    self.max_boxes_in_one_image = max_boxes_in_one_image

  def get_category(self, boxes_labeled):
    assert len(boxes_labeled) <= self.max_boxes_in_one_image
    assert len(boxes_labeled) >= 0
    if boxes_labeled == []: return 0
    if boxes_labeled == [1]: return 1
    if boxes_labeled == [2]: return 2
    if boxes_labeled == [3]: return 3
    raise Exception("not fully implemented")


  def labeling_not_done(self):
    number_labeled = len(glob.glob(os.path.join(self.labeled_dir, '%s*.json' % self.output_prefix)))
    if number_labeled >= self.total_to_label:
      return False
    return True

  def make_labelme_command_line(self, input_jpeg_fname, output_label_fname):
    import labelme
    import labelme.app as labelme_app
    cmd = 'python %s' % labelme_app.__file__
    cmd += ' --output %s' % output_label_fname
    cmd += ' %s' % input_jpeg_fname
    return cmd

  def validate_label_file(self, label_file):
    label_info = json.load(file(label_file,'r'))
    shapes = label_info['shapes']
    assert len(shapes)<=self.max_boxes_in_one_image
    unique_labels = set()
    for shape in shapes:
      try:
        shape_label = int(shape['label'])
      except:
        raise Exception('all shape labels must be an integer, i.e, 1,2, etc, but this label is %s' % shape['label'])
      assert shape_label >= 1 and shape_label <= self.max_boxes_in_one_image
      unique_labels.add(shape_label)
      assert util.is_closed_five_point_box(shape['points'])
    assert len(unique_labels)==len(shapes), "all box labels must be unique"
      
  def update_label_file(self, label_file, keystr):
    label_info = json.load(file(label_file,'r'))
    del label_info['imageData']
    category = 0
    for shape in label_info['shapes']:
      box_id = int(shape['label'])
      category += 1<<box_id
    label_info['category'] = str(category)
    label_info['imgkey'] = keystr
    fout = file(label_file,'w')
    fout.write(json.dumps(label_info, sort_keys=True, indent=4, separators=(',' , ':' )))
    fout.close()
               
  def label(self, img, keystr):
    plt.figure(1)
    plt.imshow(img)
    plt.show()
    plt.pause(.1)

    output_label_fname = os.path.join(self.labeled_dir, self.output_prefix + '_' + keystr + '.json')
    output_jpeg_fname  = os.path.join(self.jpegs_to_label, self.output_prefix + '_' + keystr + '.jpeg')

    if os.path.exists(output_label_fname):
      print("This image has already been labeled - skipping: exists: %s" % output_label_fname)
      return None
    
    while True:
      #   then user can skip classes already labeled
      ans = raw_input("Hit enter to label this image, or n to skip it: ")
      if ans.lower().strip()=='':
        break
      if ans.lower().strip()=='n':
        return None

    util.create_jpeg(img, output_jpeg_fname)
    labelme_command_line = self.make_labelme_command_line(input_jpeg_fname=output_jpeg_fname,
                                                          output_label_fname=output_label_fname)
    assert 0 == os.system(labelme_command_line)
    assert os.path.exists(output_label_fname)
    
    self.validate_label_file(output_label_fname)
    self.update_label_file(output_label_fname, keystr)

    # TODO: print category given for this image
    #       print balance - i.e, how many of each category are currently labeled
    #       then user can skip over-sampled categories

    
  
  

  
  
