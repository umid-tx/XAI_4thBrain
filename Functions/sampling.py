import numpy as np
from random import randint
import openslide
from tqdm import tqdm
import os
#from data_loader import *
from SlideRunner.dataAccess.annotations import *

class SlideContainer():

    def __init__(self, file: Path, annotations:dict, y, level: int=0, width: int=256, height: int=256, sample_func: callable=None):
        self.file = file
        self.slide = openslide.open_slide(str(file))
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.y = y
        self.annotations = annotations
        self.sample_func = sample_func
        self.classes = list(set(self.y[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int=0, y: int=0):
             return np.array(self.slide.read_region(location=(int(x * self.down_factor),int(y * self.down_factor)),
                                          level=self.level, size=(self.width, self.height)))[:, :, :3]


    @property
    def shape(self):
        return (self.width, self.height)

    def __str__(self):
        return 'SlideContainer with:\n sample func: '+str(self.sample_func)+'\n slide:'+str(self.file)

    def get_new_train_coordinates(self):
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.y, **{"classes": self.classes, "size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "annotations" : self.annotations,
                                               "level": self.level, "container" : self})

        # use default sampling method
        class_id = np.random.choice(self.classes, 1)[0]
        ids = self.y[1] == class_id
        xmin, ymin, _, _ = np.array(self.y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
        return int(xmin - self.shape / 2), int(ymin - self.height / 2)

def sampling_func(y, **kwargs):
    y_label = np.array(y[1])
    h, w = kwargs['size']

    _arbitrary_prob = 0.1
    _mit_prob = 0.5
    
    sample_prob = np.array([_arbitrary_prob, 1-_arbitrary_prob-_mit_prob, _mit_prob])
    
    case = np.random.choice(3, p=sample_prob)
    
    
    
    bg_label = [0] if y_label.dtype == np.int64 else ["bg"]
    classes = bg_label + kwargs['classes']
    level_dimensions = kwargs['level_dimensions']
    level = kwargs['level']
    if ('bg_label_prob' in kwargs):
        _bg_label_prob = kwargs['bg_label_prob']
        if (_bg_label_prob>1.0):
            raise ValueError('Probability needs to be <= 1.0.')
    else:
        _bg_label_prob = 0.0  # add a backgound label to sample complete random
    
    if ('strategy' in kwargs):
        _strategy = kwargs['strategy']
    else:
        _strategy = 'normal'
        
    if ('set' in kwargs):
        _set = kwargs['set']
    else:
        _set = 'training'

    if ('negative_class' in kwargs):
        _negative_class = kwargs['negative_class']
    else:
        _negative_class = 7 # hard examples

        
    _random_offset_scale = 0.5  # up to 50% offset to left and right of frame
    xoffset = randint(-w, w) * _random_offset_scale
    yoffset = randint(-h, h) * _random_offset_scale
    coords = np.array(y[0])

    slide_width, slide_height = level_dimensions[level]
    
    if (case==0):
        if (_set == 'training'): # sample on upper part of image
            xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h)
        elif (_set == 'validation'): # sample on lower part of image
            xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h) + int(slide_height/2)
        elif (_set == 'test'):
            xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), slide_height - h)
    if (case==2): # mitosis
        
        ids = y_label == 1

        if (_set == 'training'):
            ids[coords[:,1]>slide_height/2] = 0 # lower part not allowed
        elif (_set == 'validation'):
            ids[coords[:,1]<slide_height/2] = 0 # upper part not allowed

        if (np.count_nonzero(ids)<1):
            if (_set == 'training'): # sample on upper part of image
                xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h)
            elif (_set == 'validation'): # sample on lower part of image
                xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h) + int(slide_height/2)
        else:
            xmin, ymin, xmax, ymax = np.array(y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
    if (case==1): #nonmitosis
            annos = kwargs['annotations']
            coords = np.array(annos[_negative_class]['bboxes'])
            
            ids = np.arange(len(coords))

            if (_set == 'training'):
                ids[coords[:,1]>slide_height/2] = 0 # lower part not allowed
            elif (_set == 'validation'):
                ids[coords[:,1]<slide_height/2] = 0 # upper part not allowed

            if (np.count_nonzero(ids)<1):

                if (_set == 'training'): # sample on upper part of image
                    xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h)
                elif (_set == 'validation'): # sample on lower part of image
                    xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h) + int(slide_height/2)
            else:
                xmin, ymin, xmax, ymax = coords[ids][randint(0, np.count_nonzero(ids) - 1)]
        
    return int(xmin - w / 2 + xoffset), int(ymin - h / 2 +yoffset)


def get_slides(slidelist_test:list, database:"Database", positive_class:int=2, negative_class:int=7, basepath:str='WSI', size:int=256):


    lbl_bbox=list()
    files=list()
    train_slides=list()
    val_slides=list()

    getslides = """SELECT uid, filename FROM Slides"""
    for idx, (currslide, filename) in enumerate(tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
        if (str(currslide) in slidelist_test): # skip test slides
            continue

        database.loadIntoMemory(currslide)

        slide_path = basepath + os.sep + filename

        slide = openslide.open_slide(str(slide_path))

        level = 0#slide.level_count - 1
        level_dimension = slide.level_dimensions[level]
        down_factor = slide.level_downsamples[level]

        classes = {positive_class: 1} # Map non-mitosis to background

        labels, bboxes = [], []
        annotations = dict()
        for id, annotation in database.annotations.items():
            if annotation.deleted or annotation.annotationType != AnnotationType.SPOT:
                continue
            annotation.r = 25
            d = 2 * annotation.r / down_factor
            x_min = (annotation.x1 - annotation.r) / down_factor
            y_min = (annotation.y1 - annotation.r) / down_factor
            x_max = x_min + d
            y_max = y_min + d
            if annotation.agreedClass not in annotations:
                annotations[annotation.agreedClass] = dict()
                annotations[annotation.agreedClass]['bboxes'] = list()
                annotations[annotation.agreedClass]['label'] = list()

            annotations[annotation.agreedClass]['bboxes'].append([int(x_min), int(y_min), int(x_max), int(y_max)])
            annotations[annotation.agreedClass]['label'].append(annotation.agreedClass)

            if annotation.agreedClass in classes:
                label = classes[annotation.agreedClass]

                bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                labels.append(label)

        if len(bboxes) > 0:
            lbl_bbox.append([bboxes, labels])
            files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels], sample_func=partial(sampling_func, set='training', negative_class=negative_class)))
            train_slides.append(len(files)-1)

            lbl_bbox.append([bboxes, labels])
            files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels], sample_func=partial(sampling_func, set='validation', negative_class=negative_class)))
            val_slides.append(len(files)-1)

    return lbl_bbox, train_slides,val_slides,files
