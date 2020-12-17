from torch.utils.data import Dataset
import torchio as tio
from torchio import DATA
import numpy as np 
import SimpleITK as sitk
from pathlib import Path 
from .visualization import show_subject 
import os 

class CT_Dataset:
    def __init__(self, img_dir, show_example=False):

        self.img_dir = img_dir

        self.image_dir = self.img_dir + 'imgs'
        self.mask_dir = self.img_dir + 'masks'
        #self.image_paths = image_dir.glob('*.nii.gz')
        #self.mask_paths = mask_dir.glob('*.nii.gz')
        self.show_example = show_example

    def retrieve_data(self):

        
        subjects = []

        for image in os.listdir(self.image_dir):
            mask_tot = []
            if image.endswith('.nii.gz'):
                id = image.split(' ')[0]
                for mask in os.listdir(self.mask_dir):
                    if mask.endswith('.nii.gz'):
                        if id in mask:
                            mask_tot.append(mask)


                for i in range(0, len(mask_tot)):

                    subject = tio.Subject(
                        ct = tio.Image(self.image_dir + '/' + image),
                        segm = tio.Image(self.mask_dir + '/' + mask_tot[i])
                    )

                    subjects.append(subject)
        dataset = tio.SubjectsDataset(subjects) 

        return dataset, subjects

    def transform(self, training_split_ratio):

        training_trainsform = tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(),
            tio.RandomAffine()
        ])

        validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])

        dataset, subjects = self.retrieve_data()
        num_subjects = len(dataset)
        num_training_subjects = int(training_split_ratio*num_subjects)
        training_subjects = subjects[:num_training_subjects]
        validation_subject = subjects[num_training_subjects:]

        training_set = tio.SubjectsDataset(training_subjects, transform=training_trainsform)

        validation_set = tio.SubjectsDataset(validation_subject, transform=validation_transform)

        print('Training set:', len(training_set), 'subjects')
        print('Validation set:', len(validation_set), 'subjects')

        if self.show_example:
            one_subject = dataset[0]
            show_subject(tio.ToCanonical()(one_subject), 'ct', label_name='segm')


        return training_set, validation_set

