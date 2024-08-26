import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm ##qq plots

def clusters_plots(final_data, assignments, prototypes, model_name):

    min_samples= 500
    max_samples= len(final_data)

    colors_c = [plt.colormaps['tab20'](c) for c in np.linspace(0, 1, num=prototypes)]
    test_data_list = np.asarray(final_data)

    random_samples_ids = np.random.choice(range(test_data_list.shape[0]),size=min_samples) #random samples to QQPLOT
    colors_c = [plt.colormaps['tab20'](c) for c in np.linspace(0, 1, num=prototypes)]

    fig, ax = plt.subplots(figsize=(15, 15))

    pplot  = sm.ProbPlot(data=test_data_list[random_samples_ids].ravel())
    pplot.qqplot(other=pplot,ax=ax, marker='', linestyle='dashed',label= 'self') #45 line

    for num, cluster in enumerate(range(prototypes)): #iterates over each cluster (uses num to get new color)
        
        cur_cluster = test_data_list[assignments==cluster].copy()
        cur_cluster_shape = cur_cluster.shape

        if (cur_cluster_shape[0]>min_samples) and (cur_cluster_shape[0]<max_samples):

            random_samples_ids = np.random.choice(range(cur_cluster_shape[0]),size=min_samples) #random select samples from current cluster

            cur_color = colors_c[num] 
            pplot.qqplot(other=cur_cluster[random_samples_ids].ravel(),ax=ax,marker='', linestyle='solid', color=cur_color,
                            label= f'{cluster}: {cur_cluster_shape[0]}')

    fig.savefig(f'figs/{model_name}_qqplot.png')
    print('Done QQ plot')

    #HISTOGRAM
    n_bins = 50
    fig, ax = plt.subplots(figsize=(15, 15))

    plt.hist(test_data_list.ravel(),density=True,bins=n_bins,alpha = 0.5,label= 'self')

    for num, cluster in enumerate(range(prototypes)):
        

        cur_cluster = test_data_list[assignments==cluster].copy()
        cur_cluster_shape = cur_cluster.shape

        if (cur_cluster_shape[0]>min_samples) and (cur_cluster_shape[0]<max_samples):

            plt.hist(cur_cluster.ravel(),density=True,bins=n_bins,
                     histtype='step',color=colors_c[num],
                     label= f'{cluster}: {cur_cluster_shape[0]}',linewidth=1.5)
            
    plt.legend()
    plt.savefig(f'figs/{model_name}_hist.png')
    print('Done histogrma')

    #SAMPLES
    num_samples =  6

    fig, ax = plt.subplots(prototypes,num_samples,figsize=(20,4*prototypes))

    for cluster in range(prototypes):
            
        cluster_indexes = np.where(assignments==cluster)[0]
        samples_indexes = np.random.choice(cluster_indexes,num_samples)

        fig.subplots_adjust(wspace=0.2, hspace=0.01)

        for cur_sample in range(num_samples):
            ax[cluster,cur_sample].imshow(final_data[samples_indexes[cur_sample]],cmap='bwr',vmin=0, vmax=1)

    fig.savefig(f'figs/{model_name}_samples.png')
    print('Done Samples')
        

