#!/usr/bin/env python
# coding: utf-8

# In[3]:


from setuptools import setup

setup(
   name='diagnostic',
   version='1.0',
   description='Kumoh National Institute of Technology KLE 445 diagnosis diseases project model',
   author='MinJong Kim',
   author_email='alswhd1113@gmail.com',
   packages=['diagnostic'],  #same as name
   install_requires=['transformers', 'sentence_transformers','annoy'], #external packages as dependencies
)


# In[ ]:




