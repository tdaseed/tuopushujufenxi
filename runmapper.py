# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:47:54 2015

@author: tieqiangli
"""

import mapper
import numpy as np
import matplotlib.pyplot as plt

import os.path as op

def mapper_cluster(in_file = '/Users/tieqiangli/mapperinput/CorrelationArray1d_0708.csv',
                   out_path = '/Users/tieqiangli/mapperinput/output/'):
                        
    data = np.loadtxt(str(in_file), delimiter=',', dtype=np.float)
    
#    metricpar = {'metric': 'euclidean'}
#    
#    point_labels = np.array(['a','b','c','d'])
##    point_labels = np.array([600001,600002,600003,600004])
#    mask = [1,2,3,4]
    point_labels = None
    mask = None

#    data, point_labels = mapper.mask_data(data, mask, point_labels)
    
    '''
        Step 2: Metric
    '''
    intrinsic_metric = False
    if intrinsic_metric:
        is_vector_data = data.ndim != 1
        if is_vector_data:
            metric = Euclidean
            if metric != 'Euclidean':
                raise ValueError('Not implemented')
        data = mapper.metric.intrinsic_metric(data, k=1, eps=1.0)
    is_vector_data = data.ndim != 1
    '''
        Step 3: Filter function
    '''
    if is_vector_data:
        metricpar = {'metric': 'euclidean'}
        f = mapper.filters.Gauss_density(data,
            metricpar=metricpar,
            sigma=1.0)
    else:
        f = mapper.filters.Gauss_density(data,
            sigma=1.0)
    '''
        Step 4: Mapper parameters
    '''
    cover = mapper.cover.cube_cover_primitive(intervals=5, overlap=90.0)
    cluster = mapper.single_linkage()
    if not is_vector_data:
        metricpar = {}
    mapper_output = mapper.mapper(data, f,
        cover=cover,
        cluster=cluster,
        point_labels=point_labels,
        cutoff=None,
        metricpar=metricpar)
    mapper.scale_graph(mapper_output, f, cover=cover,
                       weighting='inverse', maxcluster=100, expand_intervals=False, exponent=10,
                       simple=False)
#    cutoff = mapper.cutoff.first_gap(gap=0.1)
#    mapper_output.cutoff(cutoff, f, cover=cover, simple=False)
    
    '''
        Step 5: Save results
    '''
    t = op.basename(in_file)    
#    date_stamp = t[len(t)-8:len(t)-4]    
    
#    mapper_output.draw_scale_graph()
#    out_file = out_path + 'scale_graph_'  + '_' + t + '.pdf'
#    plt.savefig(out_file)
    
    minsizes = []
    mapper_output.draw_2D(minsizes=minsizes)
    out_file = out_path + 'mapper_output_' + '_' + t + '.pdf'
    plt.savefig(out_file)
