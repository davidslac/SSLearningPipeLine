from sslearnpipeline import SSLearnPipeline
import random
import psana
import numpy as np  

def make_dark_if_needed():
    
def calc_log_thresh(img, thresh):
    img = img.astype(np.float32, copy=True)
    replace = img >= thresh
    newval = np.log(1.0 + img[replace] - thresh)
    img[replace]=thresh + newval
    return img


def calc_vproj_roi(img, window_len=224):
    assert len(img.shape)==2
    assert img.shape[1] >= window_len
    vproj = np.mean(img, axis=0)
    assert len(vproj)==img.shape[1]
    cumsum = np.cumsum(vproj)
    sliding_cumsum_over_windowlen = cumsum.copy()
    sliding_cumsum_over_windowlen[window_len:] -= cumsum[0:img.shape[1]-window_len]
    right = max(window_len, np.argmax(sliding_cumsum_over_windowlen))
    left = right - window_len
    adjust = 0
    if left < 0:
        adjust = -left
    if right >= img.shape[1]:
        adjust = img.shape[1] - right
    left += adjust
    right += adjust
    return [left, right]


def prepare_image(img, is_present_threshold=510000, log_thresh=300, window_len=224):
    logsum = np.sum(np.log(np.maximum(1.0, img.astype(np.float))))
    if logsum < is_present_threshold:
        return None
    log_img = calc_log_thresh(img, log_thresh)
    left, right = calc_vproj_roi(log_img, window_len=window_len)
    roi_img = log_img
    roi_img  =log_img[:,left:right]
    roi_img = np.flipud(roi_img) 
    return roi_img


def get_dark(fname, dark_run_idx, num_to_average=300):
    if not os.path.exists(fname):
        ds = psana.DataSource(dark_run_idx)
        det = psana.Detector('xtcav')
        run = ds.runs().next()
        event_times = list(run.times())
        random.seed(458921)
        random.shuffle(event_times)
        
    return npy.load(file(fname,'r'))
def main():    
    sslearn = SSLearnPipeline(outputdir='/reg/d/psdm/amo/amo86815/scratch/davidsch',
                              output_prefix='amo86815',
                              total_to_label=250,
                              max_boxes_in_one_image=2)
    dark_filename = '/reg/d/psdm/amo/amo86815/scratch/davidsch/dark_run68.npy'
    dark = get_dark(fname=dark_filename, dark_run_idx='exp=amo86815:run=68:idx')
    ds = psana.DataSource('exp=amo86815:run=72:idx')
    det = psana.Detector('xtcav')
    run = ds.runs().next()
    event_times = list(run.times())
    random.seed(394901)
    random.shuffle(event_times)
    event_times = event_times[0:10]
    while sslearn.labeling_not_done() and len(event_times):
        tm = event_times.pop()
        evt = run.event(tm)
        orig_img = det.image(evt)
        prep_img = prepare_image(orig_img)
        img_id = evt.get(psana.EventId)
        sslearn.label(prep_img, img_id)


if __name__ == '__main__':
    main()

